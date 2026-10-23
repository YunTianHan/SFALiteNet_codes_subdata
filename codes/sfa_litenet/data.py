"""真实 JPG/PNG 配对、可追溯分组划分与同步图像掩膜增强。"""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import warnings

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import distance_transform_edt
import torch
from torch.utils.data import Dataset

from .config import resolve_path

SPLITS = ("train", "val", "test")


def file_sha256(path):
    """分块哈希不将大文件整体读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _inside(root, relative):
    """数据清单仅允许引用数据根目录之内的文件。"""
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"数据路径越界或文件不存在：{relative}")
    return path


def _sort_key(path):
    stem = path.stem
    return (0, int(stem), stem) if stem.isdigit() else (1, 0, stem)


def _dhash(image):
    """64 位差值哈希仅用于保守合并疑似近重复图像，不能证明来源独立。"""
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    return int.from_bytes(np.packbits(values[:, 1:] > values[:, :-1]).tobytes(), "big")


def _metadata(config, ids):
    path = config["metadata_csv"]
    if path is None:
        return {}, None
    rows = {}
    with resolve_path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"item_id", "source_id", "section_id"}.issubset(reader.fieldnames or []):
            raise ValueError("来源 CSV 必须包含 item_id,source_id,section_id 三列")
        for row in reader:
            item = row["item_id"].strip()
            if item in rows or not item or not row["source_id"].strip() or not row["section_id"].strip():
                raise ValueError("来源 CSV 存在重复编号或缺失来源/区段字段")
            rows[item] = (row["source_id"].strip(), row["section_id"].strip())
    if set(rows) != set(ids):
        raise ValueError("来源 CSV 必须恰好覆盖当前全部数据编号")
    return rows, file_sha256(resolve_path(path))


def discover_records(config):
    """读取实际像素核对配对、尺寸及二值标签，并建立防交叉分组。

    并查集合并同来源、同区段、重复图以及近重复图，合并关系可传递。
    无元数据时以连续编号块作保守起点；这不能替代真实采集来源记录。
    """
    root = resolve_path(config["root"])
    image_dir, mask_dir = root/config["images_dir"], root/config["masks_dir"]
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"需要实际图像与掩膜目录：{image_dir} 和 {mask_dir}")
    extensions = {v.lower() for v in config["image_extensions"]}
    files = sorted((p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions), key=_sort_key)
    if not files:
        raise ValueError("真实数据目录内没有支持的图像文件")
    ids = [p.stem for p in files]
    if len(set(ids)) != len(ids):
        raise ValueError("存在同名不同扩展名图像，无法唯一匹配掩膜")
    actual_masks = {p.stem for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == config["mask_extension"].lower()}
    if actual_masks != set(ids):
        raise ValueError(f"图像与掩膜编号不一致：缺失={sorted(set(ids)-actual_masks)}，孤立={sorted(actual_masks-set(ids))}")
    metadata, metadata_hash = _metadata(config, ids)
    records, parents = [], list(range(len(files)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        left, right = find(i), find(j)
        parents[max(left, right)] = min(left, right)

    reasons = []
    for index, path in enumerate(files):
        mask_path = mask_dir/(path.stem+config["mask_extension"])
        image_path = _inside(root, path.relative_to(root))
        mask_path = _inside(root, mask_path.relative_to(root))
        with Image.open(image_path) as original, Image.open(mask_path) as label:
            image, mask_rgb = original.convert("RGB"), np.asarray(label.convert("RGB"))
            if image.size != label.size:
                raise ValueError(f"图像与标签尺寸不同：{path.stem}")
            if not np.array_equal(mask_rgb[..., 0], mask_rgb[..., 1]) or not np.array_equal(mask_rgb[..., 0], mask_rgb[..., 2]):
                raise ValueError(f"标签不是灰度二值图：{path.stem}")
            values = np.unique(mask_rgb[..., 0])
            if config["strict_binary_masks"] and not set(values.tolist()).issubset({0, 255}):
                raise ValueError(f"标签应为黑 0/白 255：{path.stem} 的像素值为 {values.tolist()}")
            binary = mask_rgb[..., 0] >= config["mask_threshold"]
            pixel_hash = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            label_hash = hashlib.sha256(str(label.size).encode()+binary.tobytes()).hexdigest()
            source, section = metadata.get(path.stem, (None, None))
            records.append({"id": path.stem, "image": image_path.relative_to(root).as_posix(),
                            "mask": mask_path.relative_to(root).as_posix(), "width": image.width,
                            "height": image.height, "image_sha256": file_sha256(image_path),
                            "mask_sha256": file_sha256(mask_path), "pixel_sha256": pixel_hash,
                            "label_sha256": label_hash, "dhash": _dhash(image),
                            "foreground_pixels": int(binary.sum()), "mask_values": values.tolist(),
                            "source_id": source, "section_id": section})
        for previous in range(index):
            other, current = records[previous], records[index]
            reason = None
            if metadata:
                if current["source_id"] == other["source_id"] or current["section_id"] == other["section_id"]:
                    reason = "来源或采集区段相同"
            elif index//config["fallback_group_size"] == previous//config["fallback_group_size"]:
                reason = "连续编号块"
            if current["pixel_sha256"] == other["pixel_sha256"]:
                if current["label_sha256"] != other["label_sha256"]:
                    raise ValueError(f"完全相同图像出现冲突标签：{other['id']} 与 {current['id']}")
                reason = "解码像素重复"
            elif bin(current["dhash"] ^ other["dhash"]).count("1") <= config["near_duplicate_hamming"]:
                reason = "差值哈希近重复"
            if reason:
                union(previous, index)
                if reason != "连续编号块":
                    reasons.append({"left": other["id"], "right": current["id"], "reason": reason})
    for index, record in enumerate(records):
        record["group"] = "group_" + records[find(index)]["id"]
    provenance = {"metadata_sha256": metadata_hash, "mode": "source_and_section" if metadata else "consecutive_blocks",
                  "merges": reasons}
    return records, provenance


def split_signature(config, records, provenance):
    """关联实际文件内容、分组与划分参数，重定位目录不改变数据身份。"""
    keys = ("split_seed", "split_ratios", "fallback_group_size", "near_duplicate_hamming",
            "mask_threshold", "strict_binary_masks")
    return stable_digest({"settings": {k: config[k] for k in keys}, "records": records, "provenance": provenance})


def prepare_manifest(config, overwrite=False):
    """按组分配近似 70/15/15，清单写到独立 outputs；禁止拆散组追求比例。"""
    path = resolve_path(config["split_manifest"])
    if path.exists() and not overwrite:
        return load_manifest(config)
    records, provenance = discover_records(config)
    groups = defaultdict(list)
    for row in records:
        groups[row["group"]].append(row)
    if len(groups) < 3:
        raise ValueError("防泄漏合并后不足三个独立组；请提供真实来源信息并审阅近重复关系，不能拆组强行划分")
    rng = random.Random(config["split_seed"])
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda group: -len(groups[group]))
    targets = dict(zip(SPLITS, (r*len(records) for r in config["split_ratios"])))
    counts = dict.fromkeys(SPLITS, 0)
    assigned = []
    for index, group in enumerate(order):
        empty = [name for name in SPLITS if counts[name] == 0]
        candidates = empty if len(order)-index == len(empty) else list(SPLITS)
        size = len(groups[group])
        chosen = min(candidates, key=lambda name: sum(
            ((counts[k]+(size if k == name else 0)-targets[k])**2)/targets[k] for k in SPLITS))
        counts[chosen] += size
        assigned.extend(dict(row, split=chosen) for row in groups[group])
    manifest = {"version": 1, "signature": split_signature(config, records, provenance),
                "seed": config["split_seed"], "requested_ratios": config["split_ratios"],
                "counts": counts, "provenance": provenance,
                "records": sorted(assigned, key=lambda row: ids_key(row["id"]))}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return manifest


def ids_key(item):
    return (0, int(item), item) if item.isdigit() else (1, 0, item)


def load_manifest(config):
    """重新读取真实数据核验清单，数据变化时拒绝沿用旧划分。"""
    path = resolve_path(config["split_manifest"])
    if not path.is_file():
        raise FileNotFoundError(f"缺少划分清单，请先执行 prepare_data.py：{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records, provenance = discover_records(config)
    if manifest.get("version") != 1 or manifest.get("signature") != split_signature(config, records, provenance):
        raise ValueError("数据、标签、来源或划分设置已改变；请审阅后使用 prepare_data.py --overwrite 重新生成清单")
    expected = {row["id"]: row for row in records}
    rows = manifest["records"]
    if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("清单存在重复、遗漏或未知样本")
    seen, counts = {}, dict.fromkeys(SPLITS, 0)
    for row in rows:
        if {k: v for k, v in row.items() if k != "split"} != expected[row["id"]] or row["split"] not in SPLITS:
            raise ValueError("清单记录与实际数据不一致")
        key, split = row["group"], row["split"]
        if key in seen and seen[key] != split:
            raise ValueError(f"检测到组跨集合泄漏：{key}")
        seen[key] = split
        counts[split] += 1
    if min(counts.values()) < 1 or counts != manifest["counts"]:
        raise ValueError("划分清单集合为空或计数不一致")
    if provenance["mode"] == "consecutive_blocks":
        warnings.warn("缺少来源元数据：编号分组和近重复检测不能保证原图或采集区段彼此独立。", stacklevel=2)
    return manifest


def signed_distance(mask):
    """GT 内为负、外为正的像素距离场；空标签用正的图像对角线惩罚假阳性。"""
    foreground = mask.astype(bool)
    if not foreground.any():
        return np.full(mask.shape, np.hypot(*mask.shape), dtype=np.float32)
    if foreground.all():
        return np.full(mask.shape, -np.hypot(*mask.shape), dtype=np.float32)
    return (distance_transform_edt(~foreground)-distance_transform_edt(foreground)).astype(np.float32)


def image_tensor(image, data_config):
    """RGB [0,255] → float32 CHW，再按集中配置归一化。"""
    values = np.asarray(image, dtype=np.float32)/255.0
    values = (values-np.asarray(data_config["mean"], np.float32))/np.asarray(data_config["std"], np.float32)
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))


class CrackDataset(Dataset):
    """只读真实文件；训练每个样本按种子、轮数与索引独立确定增强。"""

    def __init__(self, config, manifest, split, seed=42, with_distance=True):
        if split not in SPLITS:
            raise ValueError(f"未知集合：{split}")
        self.data = config["data"]
        self.augment = config["augmentation"]
        self.root = resolve_path(self.data["root"])
        self.records = [row for row in manifest["records"] if row["split"] == split]
        self.training = split == "train"
        self.seed, self.epoch, self.with_distance = seed, 0, with_distance
        if not self.records:
            raise ValueError(f"{split} 集合为空")

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _transform(self, image, mask, rng):
        h, w = self.data["image_size"]
        # 先统一输入尺寸，再根据配置执行多尺度增强。
        image = image.resize((w, h), Image.Resampling.BILINEAR)
        mask = mask.resize((w, h), Image.Resampling.NEAREST)
        if not self.training:
            return image, mask
        if rng.random() < self.augment["horizontal_flip"]:
            image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        if rng.random() < self.augment["vertical_flip"]:
            image, mask = ImageOps.flip(image), ImageOps.flip(mask)
        scale = rng.uniform(*self.augment["scale_range"])
        new_w, new_h = max(1, round(w*scale)), max(1, round(h*scale))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        pad_w, pad_h = max(0, w-new_w), max(0, h-new_h)
        left, top = rng.randint(0, pad_w), rng.randint(0, pad_h)
        padding = (left, top, pad_w-left, pad_h-top)
        # 图像用归一化均值补边，标签用背景补边；两者几何位置完全一致。
        image = ImageOps.expand(image, padding, fill=tuple(round(v*255) for v in self.data["mean"]))
        mask = ImageOps.expand(mask, padding, fill=0)
        x, y = rng.randint(0, image.width-w), rng.randint(0, image.height-h)
        box = (x, y, x+w, y+h)
        image, mask = image.crop(box), mask.crop(box)
        if rng.random() < self.augment["photometric_probability"]:
            operations = [(ImageEnhance.Brightness, "brightness"), (ImageEnhance.Contrast, "contrast"),
                          (ImageEnhance.Color, "saturation")]
            rng.shuffle(operations)
            for enhancer, key in operations:
                magnitude = self.augment[key]
                image = enhancer(image).enhance(rng.uniform(1-magnitude, 1+magnitude))
        return image, mask

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(_inside(self.root, row["image"])) as handle:
            image = handle.convert("RGB")
        with Image.open(_inside(self.root, row["mask"])) as handle:
            mask = handle.convert("L")
        original_size = (image.height, image.width)
        rng = random.Random(f"{self.seed}:{self.epoch}:{row['id']}")
        image, mask = self._transform(image, mask, rng)
        binary = (np.asarray(mask) >= self.data["mask_threshold"]).astype(np.float32)
        item = {"image": image_tensor(image, self.data), "mask": torch.from_numpy(binary[None].copy()),
                "id": row["id"], "original_size": torch.tensor(original_size)}
        if self.with_distance:
            item["distance"] = torch.from_numpy(signed_distance(binary)[None])
        return item
