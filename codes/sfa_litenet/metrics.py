"""逐图区域指标及欧氏容差 BF1；不使用方形膨胀替代欧氏距离。"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def mask_boundary(mask):
    """前景减去 3×3 腐蚀得到内边界；图像之外视为背景。"""
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def boundary_f1(prediction, target, tolerance=2.0):
    predicted, truth = mask_boundary(prediction), mask_boundary(target)
    n_predicted, n_truth = int(predicted.sum()), int(truth.sum())
    if n_predicted == 0 or n_truth == 0:
        return 1.0 if n_predicted == n_truth else 0.0
    precision = np.count_nonzero(distance_transform_edt(~truth)[predicted] <= tolerance)/n_predicted
    recall = np.count_nonzero(distance_transform_edt(~predicted)[truth] <= tolerance)/n_truth
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0


def region_metrics(tp, fp, fn, tn):
    """双方空前景视为完全正确；单方空前景时区域重合指标为零。"""
    both_empty = tp+fp+fn == 0
    return {
        "iou": tp/(tp+fp+fn) if not both_empty else 1.0,
        "accuracy": (tp+tn)/(tp+fp+fn+tn),
        "precision": tp/(tp+fp) if tp+fp else float(both_empty),
        "recall": tp/(tp+fn) if tp+fn else float(both_empty),
        "f1": 2*tp/(2*tp+fp+fn) if not both_empty else 1.0,
    }


class SegmentationMetrics:
    """主结果是逐图宏平均；附加 micro 字段明确区分像素累计结果。"""

    def __init__(self, threshold=0.5, tolerance=2.0):
        self.threshold, self.tolerance = threshold, tolerance
        self.rows = []

    def update(self, probabilities, targets, ids):
        probabilities, targets = np.asarray(probabilities), np.asarray(targets)
        if probabilities.shape != targets.shape or probabilities.ndim != 4 or probabilities.shape[1] != 1:
            raise ValueError("指标输入必须为相同尺寸的 B×1×H×W 数组")
        if len(ids) != len(probabilities):
            raise ValueError("样本编号数量与批大小不同")
        for probability, target, item in zip(probabilities[:, 0], targets[:, 0], ids):
            prediction, truth = probability >= self.threshold, target >= 0.5
            tp, fp = int((prediction & truth).sum()), int((prediction & ~truth).sum())
            fn, tn = int((~prediction & truth).sum()), int((~prediction & ~truth).sum())
            row = {"id": item, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                   **region_metrics(tp, fp, fn, tn), "bf1": boundary_f1(prediction, truth, self.tolerance)}
            self.rows.append(row)

    def compute(self):
        if not self.rows:
            raise ValueError("没有样本，不能计算指标")
        macro = {key: float(np.mean([row[key] for row in self.rows]))
                 for key in ("iou", "accuracy", "precision", "recall", "f1", "bf1")}
        counts = {key: sum(row[key] for row in self.rows) for key in ("tp", "fp", "fn", "tn")}
        return {**macro, "images": len(self.rows), "micro": region_metrics(**counts), "pixel_counts": counts}
