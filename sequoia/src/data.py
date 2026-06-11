"""Data layer: folder->label map (+hard assert), RGB cleaning, stratified k-fold,
timm transforms, and Dataset classes.

THE #1 FAILURE MODE is label scrambling. The folder name *is* the label
(`train/7/...` -> class 7). We build an explicit map and assert `"0"->0 ... "99"->99`
at import time. We NEVER rely on ImageFolder's alphabetical class order
(`"0","1","10","11",...,"2",...`), which scrambles labels -> ~random accuracy.
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from timm.data import create_transform, resolve_data_config
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Label map (the build gate)
# --------------------------------------------------------------------------- #
def build_label_map(train_dir: str, num_classes: int = 100) -> dict[str, int]:
    """folder_name -> int(folder_name). Asserts the map is exactly {0..99}.

    Raises if any folder isn't an int, if labels don't cover 0..num_classes-1,
    or if the identity `int(name) == label` is ever violated.
    """
    folders = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    label_map = {}
    for d in folders:
        if not d.isdigit():
            raise ValueError(f"train subfolder '{d}' is not an integer class name")
        label_map[d] = int(d)
    labels = sorted(label_map.values())
    assert labels == list(range(num_classes)), (
        f"label set must be exactly 0..{num_classes - 1}; got min={labels[0]} "
        f"max={labels[-1]} n={len(labels)}"
    )
    # identity check: "0"->0 ... "99"->99
    for name, lab in label_map.items():
        assert int(name) == lab, f"label map broken: {name} -> {lab}"
    return label_map


def build_samples(train_dir: str, num_classes: int = 100) -> tuple[list[str], np.ndarray]:
    """Explicit (paths, labels) built from the label map — never from dir-iteration order.

    Returns paths sorted (folder asc, then filename asc) for determinism.
    """
    label_map = build_label_map(train_dir, num_classes)
    paths: list[str] = []
    labels: list[int] = []
    for name in sorted(label_map, key=int):
        cls_dir = os.path.join(train_dir, name)
        for fn in sorted(os.listdir(cls_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fn))
                labels.append(label_map[name])
    return paths, np.asarray(labels, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Test IDs (Kaggle expects every image in test_dir — currently 1036)
# --------------------------------------------------------------------------- #
def load_all_test_ids(test_dir: str) -> list[str]:
    """All test filenames sorted numerically: 0.jpg, 1.jpg, …, 1035.jpg."""
    ids = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    ids.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return ids


def load_test_ids(sample_submission: str) -> list[str]:
    """IDs from sample_submission.csv (1000 rows). Legacy helper — do not use for Kaggle submit."""
    return pd.read_csv(sample_submission)["ID"].astype(str).tolist()


# --------------------------------------------------------------------------- #
# Stratified k-fold
# --------------------------------------------------------------------------- #
def stratified_folds(labels: np.ndarray, n_splits: int, seed: int, shuffle: bool = True):
    """Yield (fold_idx, train_idx, val_idx). Classes with < n_splits images warn
    (sklearn) and land in val for some folds — acceptable; the caller logs it.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    X = np.zeros(len(labels))
    for i, (tr, va) in enumerate(skf.split(X, labels)):
        yield i, tr, va


# --------------------------------------------------------------------------- #
# Transforms (per-model via timm)
# --------------------------------------------------------------------------- #
def _safe_open_rgb(path: str) -> Image.Image:
    """Open and force RGB (handles L/RGBA/P). Data is verified all-RGB; this is a guard."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def build_transforms(model, img_size: int | None, aug_cfg: dict, train: bool) -> Callable:
    """timm transforms using the model's resolved data config (correct mean/std/crop).

    Train: RandomResizedCrop + hflip + RandAugment + color jitter + RandomErasing.
    Eval:  resize -> center-crop at img_size.
    """
    data_cfg = resolve_data_config({}, model=model)
    if img_size:
        data_cfg["input_size"] = (3, img_size, img_size)
    if train:
        return create_transform(
            input_size=data_cfg["input_size"],
            is_training=True,
            mean=data_cfg["mean"],
            std=data_cfg["std"],
            scale=(aug_cfg.get("train_crop_min", 0.4), 1.0),
            hflip=aug_cfg.get("hflip", 0.5),
            color_jitter=aug_cfg.get("color_jitter", 0.4),
            auto_augment=aug_cfg.get("rand_augment", "rand-m9-mstd0.5-inc1"),
            re_prob=aug_cfg.get("reprob", 0.25),
            re_mode="pixel",
            interpolation=data_cfg.get("interpolation", "bicubic"),
        )
    return create_transform(
        input_size=data_cfg["input_size"],
        is_training=False,
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        crop_pct=data_cfg.get("crop_pct", 0.875),
        interpolation=data_cfg.get("interpolation", "bicubic"),
    )


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class TrainDataset(Dataset):
    """Indexed subset of (paths, labels) with a transform — used for train and val folds."""

    def __init__(self, paths: list[str], labels: np.ndarray, indices: np.ndarray, transform: Callable):
        self.paths = [paths[i] for i in indices]
        self.labels = labels[indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = self.transform(_safe_open_rgb(self.paths[i]))
        return img, int(self.labels[i])


class TestDataset(Dataset):
    """Test images for the given IDs (numeric filename order). Returns (img, id_str)."""

    def __init__(self, test_dir: str, ids: list[str], transform: Callable):
        self.paths = [os.path.join(test_dir, i) for i in ids]
        self.ids = ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img = self.transform(_safe_open_rgb(self.paths[i]))
        return img, self.ids[i]


def class_sample_weights(labels: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    """Per-sample weights = sqrt-inverse class frequency (for WeightedRandomSampler)."""
    sub = labels[indices]
    counts = np.bincount(sub, minlength=int(labels.max()) + 1).astype(np.float64)
    inv = 1.0 / np.sqrt(np.clip(counts, 1, None))
    return torch.as_tensor(inv[sub], dtype=torch.double)
