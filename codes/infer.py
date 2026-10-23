"""单图或目录推理：保存原始尺寸的概率图、二值掩膜与裂隙叠加图。"""

import argparse
import json
import os

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from sfa_litenet.config import resolve_path, validate_config
from sfa_litenet.data import image_tensor
from sfa_litenet.models import SFALiteNet
from sfa_litenet.utils import choose_device, load_checkpoint, seed_everything, write_json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="单张图像或图像目录（不递归）")
    parser.add_argument("--output", default="outputs/inference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None, help="省略时使用检查点阈值")
    args = parser.parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    validate_config(config)
    threshold = args.threshold if args.threshold is not None else config["evaluation"]["threshold"]
    if not 0 < threshold < 1:
        parser.error("阈值必须在 (0,1) 内")
    if config["project"]["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(checkpoint["seed"], config["project"]["deterministic"])
    device = choose_device(args.device)
    model = SFALiteNet(config["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    source, output = resolve_path(args.input), resolve_path(args.output)
    root = resolve_path(config["data"]["root"])
    if output == root or root in output.parents or output == source or (source.is_dir() and source in output.parents):
        raise ValueError("推理输出必须离开原始数据目录和输入目录")
    extensions = {value.lower() for value in config["data"]["image_extensions"]}
    files = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not files:
        raise FileNotFoundError("输入中没有可推理图像")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("输入存在同名不同扩展名图像，输出会冲突，请先分开目录")
    output.mkdir(parents=True, exist_ok=True)
    existing = [output/(path.stem+suffix) for path in files for suffix in ("_mask.png", "_probability.png", "_overlay.png")]
    if any(path.exists() for path in existing):
        raise FileExistsError("输出中已有同名预测，请选择新的 --output，避免覆盖")
    h, w = config["data"]["image_size"]
    records = []
    for path in files:
        with Image.open(path) as handle:
            original = handle.convert("RGB")
        resized = original.resize((w, h), Image.Resampling.BILINEAR)
        tensor = image_tensor(resized, config["data"]).unsqueeze(0).to(device)
        probability = model(tensor).sigmoid()
        # 先恢复连续概率到原图尺寸再阈值化，避免直接缩放二值掩膜的锯齿。
        probability = F.interpolate(probability, size=(original.height, original.width), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        if not np.isfinite(probability).all():
            raise FloatingPointError(f"模型产生非有限预测概率：{path}")
        mask = probability >= threshold
        Image.fromarray((mask*255).astype(np.uint8)).save(output/f"{path.stem}_mask.png")
        # 16 位 PNG 概率值除以 65535 即为 [0,1]，保留比 8 位更高精度。
        Image.fromarray(np.rint(probability*65535).astype(np.uint16)).save(output/f"{path.stem}_probability.png")
        overlay = np.asarray(original, dtype=np.float32).copy()
        overlay[mask] = 0.55*overlay[mask]+0.45*np.array([255, 0, 0], dtype=np.float32)
        Image.fromarray(np.rint(overlay).astype(np.uint8)).save(output/f"{path.stem}_overlay.png")
        records.append({"input": str(path), "width": original.width, "height": original.height,
                        "foreground_pixels": int(mask.sum())})
    write_json(output/"inference.json", {"checkpoint": str(checkpoint_path), "threshold": threshold,
               "network_image_size": [h, w], "probability_scale": 65535, "images": records})
    print(json.dumps({"images": len(files), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
