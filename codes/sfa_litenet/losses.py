"""Focal、Dice、距离边界和可微 soft-clDice 联合目标。"""

import torch
from torch import nn
from torch.nn import functional as F


def soft_erode(image):
    """轴向最小池化近似二值十字结构元腐蚀，保留概率梯度。"""
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_skeleton(image, iterations):
    """反复腐蚀及开运算提取软骨架，输入输出均为 B×1×H×W。"""
    opened = F.max_pool2d(soft_erode(image), 3, stride=1, padding=1)
    skeleton = F.relu(image-opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = F.max_pool2d(soft_erode(image), 3, stride=1, padding=1)
        delta = F.relu(image-opened)
        skeleton = skeleton + F.relu(delta-skeleton*delta)
    return skeleton


class JointLoss(nn.Module):
    """权重默认 1/1/0.01/1；边界项使用 GT 有符号距离的区域积分。

    距离边界项可以为负，这符合有符号距离定义，不应用截断破坏梯度。
    空 GT 的 soft-clDice 回退为平均前景概率，避免零骨架的不确定比值。
    """

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)

    def forward(self, logits, target, distance):
        if logits.shape != target.shape or target.shape != distance.shape:
            raise ValueError("损失函数要求 logits、mask、distance 均为 B×1×H×W 且尺寸一致")
        c = self.config
        probability = logits.sigmoid()
        dimensions = (1, 2, 3)
        smooth = c["smooth"]
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = c["focal_alpha"]*target + (1-c["focal_alpha"])*(1-target)
        focal = (alpha*(1-pt).pow(c["focal_gamma"])*bce).mean()
        intersection = (probability*target).sum(dimensions)
        dice = (1-(2*intersection+smooth)/(probability.sum(dimensions)+target.sum(dimensions)+smooth)).mean()
        boundary = (probability*distance).mean()
        predicted_skeleton = soft_skeleton(probability, c["skeleton_iterations"])
        target_skeleton = soft_skeleton(target, c["skeleton_iterations"])
        topology_precision = ((predicted_skeleton*target).sum(dimensions)+smooth)/(predicted_skeleton.sum(dimensions)+smooth)
        topology_recall = ((target_skeleton*probability).sum(dimensions)+smooth)/(target_skeleton.sum(dimensions)+smooth)
        cldice_per_image = 1-(2*topology_precision*topology_recall+smooth)/(topology_precision+topology_recall+smooth)
        cldice_per_image = torch.where(target.sum(dimensions) > 0, cldice_per_image, probability.mean(dimensions))
        cldice = cldice_per_image.mean()
        parts = {"focal": focal, "dice": dice, "boundary": boundary, "cldice": cldice}
        total = sum(c[key+"_weight"]*value for key, value in parts.items())
        return total, parts
