"""集中读取配置；路径以项目根目录为基准，避免工作目录改变语义。"""

from copy import deepcopy
from pathlib import Path
import math

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def resolve_path(value):
    """将配置中的路径解析为绝对路径。"""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _merge(base, update, prefix=""):
    for key, value in update.items():
        name = f"{prefix}{key}"
        if key not in base:
            raise ValueError(f"未知配置字段：{name}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"配置 {name} 必须为映射")
            _merge(base[key], value, name + ".")
        else:
            previous = base[key]
            if previous is None:
                valid = value is None or isinstance(value, str)
            elif isinstance(previous, bool):
                valid = isinstance(value, bool)
            elif isinstance(previous, int):
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(previous, float):
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            else:
                valid = isinstance(value, type(previous))
            if not valid:
                raise ValueError(f"配置字段类型错误：{name}")
            base[key] = value


def load_config(path=None):
    """默认配置与用户 YAML 合并；未知字段直接报错，防止拼写错误静默生效。"""
    with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if path is not None and resolve_path(path) != DEFAULT_CONFIG:
        with resolve_path(path).open(encoding="utf-8") as handle:
            update = yaml.safe_load(handle)
        if not isinstance(update, dict):
            raise ValueError("用户配置必须为 YAML 映射")
        _merge(config, update)
    validate_config(config)
    return deepcopy(config)


def validate_config(c):
    """在真正读取数据或创建模型之前校验关键参数。"""
    d, m, t, a, e = (c[k] for k in ("data", "model", "training", "augmentation", "evaluation"))
    if len(d["image_size"]) != 2 or any(int(v) != v or v < 16 for v in d["image_size"]):
        raise ValueError("image_size 必须为两个不小于 16 的整数")
    if len(m["encoder_channels"]) != 4 or any(not isinstance(v, int) or v < 1 for v in m["encoder_channels"]):
        raise ValueError("编码器必须包含四个正通道数")
    for key in ("bottleneck_channels", "blocks_per_stage", "frequency_reduction",
                "coordinate_reduction", "coordinate_min_channels", "boundary_channels"):
        if not isinstance(m[key], int) or m[key] < 1:
            raise ValueError(f"model.{key} 必须为正整数")
    if m["eca_kernel"] < 1 or m["eca_kernel"] % 2 != 1:
        raise ValueError("ECA 卷积核必须为正奇数")
    if len(d["mean"]) != 3 or len(d["std"]) != 3 or min(d["std"]) <= 0:
        raise ValueError("RGB 归一化均值和标准差需要三个分量，标准差必须为正")
    ratios = d["split_ratios"]
    if len(ratios) != 3 or min(ratios) <= 0 or not math.isclose(sum(ratios), 1.0):
        raise ValueError("划分比例必须为三个正数且总和为 1")
    if d["fallback_group_size"] < 1 or not 0 <= d["near_duplicate_hamming"] <= 64:
        raise ValueError("默认分组大小或近重复阈值无效")
    if not 0 < d["mask_threshold"] <= 255:
        raise ValueError("掩膜阈值必须在 (0,255] 内")
    for key in ("horizontal_flip", "vertical_flip", "photometric_probability"):
        if not 0 <= a[key] <= 1:
            raise ValueError(f"增强概率 {key} 必须在 [0,1] 内")
    if len(a["scale_range"]) != 2 or not 0 < a["scale_range"][0] <= a["scale_range"][1]:
        raise ValueError("增强缩放范围无效")
    if any(not 0 <= a[k] <= 1 for k in ("brightness", "contrast", "saturation")):
        raise ValueError("颜色增强幅度必须在 [0,1] 内")
    for value in (t["batch_size"], t["epochs"], t["patience"], e["batch_size"]):
        if not isinstance(value, int) or value < 1:
            raise ValueError("批大小、轮数和早停耐心值必须为正整数")
    if not 0 <= t["warmup_epochs"] < t["epochs"]:
        raise ValueError("预热轮数必须小于总轮数")
    if not 0 < t["warmup_start_factor"] <= 1:
        raise ValueError("预热起始比例必须在 (0,1] 内")
    if not 0 <= t["minimum_lr"] <= t["learning_rate"] or t["learning_rate"] <= 0:
        raise ValueError("学习率范围无效")
    if t["weight_decay"] < 0 or t["gradient_clip_norm"] < 0 or t["min_delta"] < 0:
        raise ValueError("训练正则项和早停改善量不得为负")
    if not t["seeds"] or len(set(t["seeds"])) != len(t["seeds"]):
        raise ValueError("训练种子列表不能为空或重复")
    if any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in t["seeds"]):
        raise ValueError("训练种子必须为 [0,2^32) 内的整数")
    if not isinstance(d["split_seed"], int) or not 0 <= d["split_seed"] < 2**32:
        raise ValueError("划分种子必须为 [0,2^32) 内的整数")
    if c["project"]["num_workers"] < 0:
        raise ValueError("工作进程数量不得为负")
    if not 0 < e["threshold"] < 1 or e["boundary_tolerance"] < 0:
        raise ValueError("分割阈值或边界容差无效")
    loss = c["loss"]
    if any(loss[k] < 0 for k in ("focal_weight", "dice_weight", "boundary_weight", "cldice_weight")):
        raise ValueError("损失权重不得为负")
    if sum(loss[k] for k in ("focal_weight", "dice_weight", "boundary_weight", "cldice_weight")) <= 0:
        raise ValueError("至少启用一项损失")
    if not 0 <= loss["focal_alpha"] <= 1 or loss["focal_gamma"] < 0 or loss["smooth"] <= 0:
        raise ValueError("Focal 或平滑参数无效")
    if loss["skeleton_iterations"] < 1:
        raise ValueError("软骨架迭代次数必须为正")
    if any(c["benchmark"][k] < 1 for k in ("warmup", "iterations", "repeats")):
        raise ValueError("基准计时次数必须为正")
    for key in ("experiment",):
        value = c["project"][key]
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError("实验名称必须为单个目录名")
    # 所有生成文件必须离开原始数据目录，避免输出污染甚至覆盖数据。
    root = resolve_path(d["root"])
    for output in (resolve_path(c["project"]["output_dir"]), resolve_path(d["split_manifest"])):
        if output == root or root in output.parents:
            raise ValueError("输出目录和划分清单不得写入原始数据目录")
