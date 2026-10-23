"""训练、逐图评估、早停及支持继续训练的检查点管理。"""

import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .config import resolve_path, validate_config
from .data import CrackDataset, load_manifest, stable_digest
from .losses import JointLoss
from .metrics import SegmentationMetrics
from .models import SFALiteNet
from .utils import (capture_rng, choose_device, load_checkpoint, make_logger, restore_rng,
                    save_checkpoint, seed_everything, write_csv, write_json)


def run_directory(config, seed):
    return resolve_path(config["project"]["output_dir"])/config["project"]["experiment"]/f"seed_{seed}"


def make_loader(dataset, config, seed, training=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=config["training" if training else "evaluation"]["batch_size"],
                      shuffle=training, num_workers=config["project"]["num_workers"],
                      pin_memory=torch.cuda.is_available(), drop_last=False, generator=generator,
                      persistent_workers=False)


def epoch_learning_rate(epoch, config):
    """轮编号从零开始；前五轮线性预热，第六轮起余弦下降至最后一轮的最低值。"""
    base, minimum = config["learning_rate"], config["minimum_lr"]
    warmup = config["warmup_epochs"]
    if epoch < warmup:
        progress = epoch/max(1, warmup-1)
        return base*(config["warmup_start_factor"]+(1-config["warmup_start_factor"])*progress)
    progress = (epoch-warmup)/max(1, config["epochs"]-warmup-1)
    return minimum+(base-minimum)*0.5*(1+math.cos(math.pi*progress))


def _batch_loss(model, batch, criterion, device):
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    distance = batch["distance"].to(device, non_blocking=True)
    logits = model(image)
    total, parts = criterion(logits, mask, distance)
    if not torch.isfinite(total):
        raise FloatingPointError(f"出现非有限损失，样本：{batch['id']}")
    return logits, mask, total, parts


def train_epoch(model, loader, criterion, optimizer, device, clip_norm):
    model.train()
    sums, count = {}, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        _, _, total, parts = _batch_loss(model, batch, criterion, device)
        total.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
        optimizer.step()
        number = len(batch["id"])
        for key, value in {"total": total, **parts}.items():
            sums[key] = sums.get(key, 0.0)+float(value.detach())*number
        count += number
    return {key: value/count for key, value in sums.items()}


@torch.inference_mode()
def evaluate_loader(model, loader, criterion, device, config, output=None):
    """在统一输入尺寸计算指标，预测结果也按该尺寸保存以便审计 BF1 容差。"""
    model.eval()
    metrics = SegmentationMetrics(config["evaluation"]["threshold"], config["evaluation"]["boundary_tolerance"])
    sums, count = {}, 0
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        logits, mask, total, parts = _batch_loss(model, batch, criterion, device)
        probabilities = logits.sigmoid().cpu().numpy()
        metrics.update(probabilities, mask.cpu().numpy(), batch["id"])
        number = len(batch["id"])
        for key, value in {"total": total, **parts}.items():
            sums[key] = sums.get(key, 0.0)+float(value)*number
        count += number
        if output is not None and config["evaluation"]["save_predictions"]:
            prediction_dir = output/"predictions"
            prediction_dir.mkdir(exist_ok=True)
            for item, probability in zip(batch["id"], probabilities[:, 0]):
                binary = (probability >= config["evaluation"]["threshold"]).astype(np.uint8)*255
                Image.fromarray(binary).save(prediction_dir/f"{item}.png")
    result = metrics.compute()
    result["loss"] = {key: value/count for key, value in sums.items()}
    result["threshold"] = config["evaluation"]["threshold"]
    result["boundary_tolerance"] = config["evaluation"]["boundary_tolerance"]
    result["image_size"] = config["data"]["image_size"]
    if output is not None:
        write_csv(output/"per_image.csv", metrics.rows)
        write_json(output/"metrics.json", result)
    return result


def train_run(config, seed, manifest=None, resume=None):
    """固定数据划分，种子只影响初始化、训练样本顺序与在线增强。"""
    validate_config(config)
    # 确定性 CUDA 矩阵乘法要求在创建 CUDA 上下文之前设置该变量。
    if config["project"]["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(seed, config["project"]["deterministic"])
    device = choose_device(config["project"]["device"])
    manifest = manifest if manifest is not None else load_manifest(config["data"])
    directory = run_directory(config, seed)
    checkpoint = load_checkpoint(resolve_path(resume)) if resume else None
    if checkpoint is None and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"运行目录非空，避免覆盖；请更改 experiment 或使用 --resume：{directory}")
    if checkpoint is not None and directory.exists() and any(directory.iterdir()):
        if resolve_path(resume) != (directory/"last.pt").resolve():
            raise FileExistsError("非空运行目录只能从该目录的 last.pt 继续；迁移检查点请使用新的 experiment")
    if checkpoint is not None:
        if checkpoint["seed"] != seed or stable_digest(checkpoint["manifest"]) != stable_digest(manifest):
            raise ValueError("恢复训练必须使用相同种子及完整划分清单")
        # 路径和设备可迁移，任何影响优化/增强/模型的字段必须保持一致。
        for key in ("model", "loss", "training", "augmentation", "evaluation"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"恢复训练时不允许修改 {key} 配置")
        for key in ("image_size", "mean", "std", "mask_threshold"):
            if checkpoint["config"]["data"][key] != config["data"][key]:
                raise ValueError(f"恢复训练的数据处理配置 {key} 已改变")
        if checkpoint["config"]["project"]["deterministic"] != config["project"]["deterministic"]:
            raise ValueError("恢复训练时确定性设置必须一致")
    logger = make_logger(directory)
    logger.info("设备=%s；随机种子=%s；数据划分=%s；FP32 训练", device, seed, manifest["counts"])
    model = SFALiteNet(config["model"]).to(device)
    logger.info("当前实现参数数量=%s（实际构建值，不预填论文参数量）", sum(p.numel() for p in model.parameters()))
    criterion = JointLoss(config["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    training = CrackDataset(config, manifest, "train", seed)
    validation = CrackDataset(config, manifest, "val", seed)
    validation_loader = make_loader(validation, config, seed)
    best_iou, best_epoch, bad_epochs, start, history = -1.0, -1, 0, 0, []
    best_model = None
    if checkpoint is not None:
        for key in ("optimizer", "rng", "best_iou", "best_epoch", "bad_epochs", "history", "best_model"):
            if key not in checkpoint:
                raise ValueError(f"该检查点不支持继续训练，缺少 {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_iou, bad_epochs = checkpoint["best_iou"], checkpoint["bad_epochs"]
        best_epoch = checkpoint["best_epoch"]
        start, history = checkpoint["epoch"]+1, checkpoint["history"]
        best_model = checkpoint["best_model"]
        restore_rng(checkpoint["rng"])
        logger.info("已恢复到第 %s 轮结束，最佳验证 IoU=%.6f", start, best_iou)
    if start >= config["training"]["epochs"] or bad_epochs >= config["training"]["patience"]:
        raise ValueError("该检查点已完成全部轮数或触发早停，没有待继续的训练轮次")
    write_json(directory/"config.json", config)
    write_json(directory/"split.json", manifest)
    write_json(directory/"environment.json", {"torch": str(torch.__version__), "numpy": np.__version__,
               "device": str(device), "cuda_version": torch.version.cuda,
               "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    for epoch in range(start, config["training"]["epochs"]):
        training.set_epoch(epoch)
        loader = make_loader(training, config, seed+epoch, training=True)
        learning_rate = epoch_learning_rate(epoch, config["training"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train_loss = train_epoch(model, loader, criterion, optimizer, device, config["training"]["gradient_clip_norm"])
        validation_result = evaluate_loader(model, validation_loader, criterion, device, config)
        iou = validation_result["iou"]
        improved = iou > best_iou + config["training"]["min_delta"]
        if improved:
            best_iou, bad_epochs = iou, 0
            best_epoch = epoch
            # CPU 快照避免保留计算图，并允许只用 last.pt 在新目录恢复最佳模型。
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        row = {"epoch": epoch+1, "lr": learning_rate,
               **{f"train_{key}": value for key, value in train_loss.items()},
               **{f"val_{key}": validation_result[key] for key in ("iou", "accuracy", "precision", "recall", "f1", "bf1")},
               **{f"val_loss_{key}": value for key, value in validation_result["loss"].items()}}
        history.append(row)
        write_csv(directory/"history.csv", history)
        write_json(directory/"history.json", history)
        payload = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "best_model": best_model, "config": config, "manifest": manifest, "seed": seed,
                   "epoch": epoch, "best_iou": best_iou, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                   "history": history, "rng": capture_rng()}
        save_checkpoint(directory/"last.pt", payload)
        if improved:
            # best.pt 用于评估和推理；继续训练使用包含优化器和随机状态的 last.pt。
            save_checkpoint(directory/"best.pt", {key: payload[key] for key in
                ("format_version", "model", "config", "manifest", "seed", "epoch", "best_iou")})
            save_checkpoint(directory/"weights.pt", best_model)
        elif not (directory/"best.pt").exists():
            save_checkpoint(directory/"best.pt", {"format_version": 1, "model": best_model,
                "config": config, "manifest": manifest, "seed": seed, "epoch": best_epoch,
                "best_iou": best_iou})
            save_checkpoint(directory/"weights.pt", best_model)
        logger.info("轮次 %d/%d lr=%.7f train=%.6f val=%.6f IoU=%.6f BF1=%.6f 未改善=%d",
                    epoch+1, config["training"]["epochs"], learning_rate, train_loss["total"],
                    validation_result["loss"]["total"], iou, validation_result["bf1"], bad_epochs)
        if bad_epochs >= config["training"]["patience"]:
            logger.info("验证 IoU 连续 %d 轮未改善，触发早停", bad_epochs)
            break
    model.load_state_dict(best_model, strict=True)
    result = evaluate_loader(model, validation_loader, criterion, device, config, directory/"validation")
    result["seed"] = seed
    result["checkpoint"] = str(directory/"best.pt")
    write_json(directory/"validation"/"metrics.json", result)
    logger.info("训练结束，最佳验证 IoU=%.6f；测试集保留给独立 evaluate.py", best_iou)
    return result


def evaluate_checkpoint(config, checkpoint_path, split="test", output=None, manifest=None):
    """使用检查点内模型/预处理设置，当前配置仅提供路径、设备和评估控制。"""
    checkpoint = load_checkpoint(resolve_path(checkpoint_path))
    effective = json.loads(json.dumps(checkpoint["config"]))
    effective["project"] = config["project"]
    effective["evaluation"] = config["evaluation"]
    for key in ("root", "split_manifest", "metadata_csv"):
        effective["data"][key] = config["data"][key]
    validate_config(effective)
    manifest = manifest if manifest is not None else load_manifest(effective["data"])
    if stable_digest(manifest) != stable_digest(checkpoint["manifest"]):
        raise ValueError("当前数据划分与检查点训练划分不同，拒绝混用")
    if effective["project"]["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(checkpoint["seed"], effective["project"]["deterministic"])
    device = choose_device(effective["project"]["device"])
    model = SFALiteNet(effective["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = CrackDataset(effective, manifest, split, checkpoint["seed"])
    # 即使手动评估 train 集，也不使用随机增强。
    dataset.training = False
    loader = make_loader(dataset, effective, checkpoint["seed"])
    directory = Path(output) if output else run_directory(config, checkpoint["seed"])/f"evaluation_{split}"
    data_root = resolve_path(effective["data"]["root"])
    if directory.resolve() == data_root or data_root in directory.resolve().parents:
        raise ValueError("评估输出不得位于原始数据目录")
    result = evaluate_loader(model, loader, JointLoss(effective["loss"]).to(device), device, effective, directory)
    result.update({"seed": checkpoint["seed"], "split": split, "checkpoint": str(resolve_path(checkpoint_path)),
                   "manifest_signature": manifest["signature"]})
    write_json(directory/"metrics.json", result)
    return result


def summarize_runs(results, output):
    """保存逐种子结果以及样本标准差；单次运行不伪造标准差。"""
    summary = {"runs": len(results), "seeds": [row["seed"] for row in results], "std_ddof": 1, "metrics": {}}
    if len(set(summary["seeds"])) != len(results):
        raise ValueError("汇总中出现重复种子")
    for key in ("iou", "accuracy", "precision", "recall", "f1", "bf1"):
        values = [row[key] for row in results]
        summary["metrics"][key] = {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    write_json(output, {**summary, "individual_runs": results})
    return summary
