"""计算模型复杂度并测量 FP32 推理速度。"""

import argparse
import io
import os
import time

import numpy as np
import torch

from sfa_litenet.config import load_config, resolve_path, validate_config
from sfa_litenet.data import CrackDataset, load_manifest
from sfa_litenet.models import SFALiteNet
from sfa_litenet.utils import choose_device, load_checkpoint, seed_everything, write_json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None, help="可选；省略则只分析随机初始化的架构")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="outputs/benchmark.json")
    args = parser.parse_args()
    checkpoint = load_checkpoint(resolve_path(args.checkpoint)) if args.checkpoint else None
    config = load_config(args.config)
    if checkpoint:
        for key in ("model", "loss", "augmentation"):
            config[key] = checkpoint["config"][key]
        for key in ("image_size", "mean", "std", "mask_threshold"):
            config["data"][key] = checkpoint["config"]["data"][key]
    validate_config(config)
    if config["project"]["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(42, config["project"]["deterministic"])
    device = choose_device(args.device or config["project"]["device"])
    output = resolve_path(args.output)
    root = resolve_path(config["data"]["root"])
    if output == root or root in output.parents:
        raise ValueError("复杂度报告不得写入原始数据目录")
    manifest = load_manifest(config["data"])
    # 使用数据集中的图像，确保计时反映实际输入尺寸。
    dataset = CrackDataset(config, manifest, "val", with_distance=False)
    sample = dataset[0]["image"].unsqueeze(0).to(device)
    model = SFALiteNet(config["model"]).to(device).eval()
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    result = {"parameters": sum(p.numel() for p in model.parameters()),
              "state_dict_size_mb": buffer.getbuffer().nbytes/1e6,
              "device": str(device), "torch": str(torch.__version__), "precision": "FP32",
              "batch_size": 1, "input_shape": list(sample.shape), "sample_id": dataset.records[0]["id"],
              "checkpoint": str(resolve_path(args.checkpoint)) if checkpoint else None}
    try:
        from fvcore.nn import FlopCountAnalysis
        analysis = FlopCountAnalysis(model, sample)
        result["fvcore_flops"] = int(analysis.total())
        result["unsupported_ops"] = dict(analysis.unsupported_ops())
        result["uncalled_modules"] = sorted(analysis.uncalled_modules())
        result["flops_note"] = "fvcore 默认算子口径，乘加按一次计算；未支持的算子未计入总量，不能视作完整硬件操作数。"
    except ImportError:
        result["flops_error"] = "缺少可选 fvcore，未计算 FLOPs。"
    except (RuntimeError, ValueError, NotImplementedError) as error:
        result["flops_error"] = str(error)
    durations = []
    for _ in range(config["benchmark"]["repeats"]):
        for _ in range(config["benchmark"]["warmup"]):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(config["benchmark"]["iterations"]):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append((time.perf_counter()-start)/config["benchmark"]["iterations"])
    result.update({"benchmark": config["benchmark"], "latency_ms_per_run": [v*1000 for v in durations],
                   "fps_per_run": [1/v for v in durations], "fps_mean": float(np.mean([1/v for v in durations])),
                   "fps_std": float(np.std([1/v for v in durations], ddof=1)) if len(durations)>1 else None,
                   "timing_scope": "只包含模型前向传播，不包含文件读取、预处理、后处理及模型加载"})
    write_json(output, result)
    print(f"分析结果已写入：{output}")


if __name__ == "__main__":
    main()
