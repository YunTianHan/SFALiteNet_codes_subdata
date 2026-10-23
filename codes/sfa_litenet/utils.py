"""随机状态、设备选择、结构化日志及安全的模型检查点接口。"""

import csv
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("本项目支持 auto、cpu、cuda 或 cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但当前 PyTorch 未检测到可用 CUDA")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("所选 CUDA 设备编号超出范围")
    return device


def make_logger(directory):
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(directory.resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(directory/"run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_json(path, value):
    """用临时文件替换完成写入，避免中断留下半个清单或指标文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("没有可写出的 CSV 行")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture_rng():
    """只保存 weights_only 加载器可处理的张量和 Python 基础类型。"""
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"name": state[0], "keys": state[1].tolist(),
            "position": state[2], "has_gauss": state[3], "cached_gauss": state[4]},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns["name"], np.asarray(ns["keys"], dtype=np.uint32), ns["position"], ns["has_gauss"], ns["cached_gauss"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("恢复训练的可见 CUDA 设备数量与保存时不同")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    """只接受本项目的结构化检查点；不回退到执行任意 pickle 的加载方式。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"检查点不存在：{path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("检查点格式不匹配；请使用本项目 train.py 产生的 best.pt 或 last.pt")
    for key in ("model", "config", "manifest", "seed", "epoch"):
        if key not in checkpoint:
            raise ValueError(f"检查点缺少字段：{key}")
    return checkpoint
