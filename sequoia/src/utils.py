"""Shared utilities: seeding/determinism, device selection, metrics, logging,
config loading, and the submission-format validator.

All training/inference correctness gates that aren't data-specific live here.
"""
from __future__ import annotations

import logging
import os
import random
import sys
import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch. With deterministic=True, make cuDNN deterministic.

    Residual nondeterminism on CUDA/MPS (atomic ops, some kernels) is documented
    in the README — bitwise reproducibility is not guaranteed on GPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(prefer: str | None = None) -> torch.device:
    """Device-agnostic selection: cuda -> mps -> cpu. `prefer` forces a choice."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_enabled(cfg_amp: bool, device: torch.device) -> bool:
    """AMP is only enabled on CUDA — MPS/CPU autocast is unreliable for this stack."""
    return bool(cfg_amp) and device.type == "cuda"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def resolve_backbone(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """Merge global train.* with a backbone entry's overrides; return a flat spec."""
    if key not in cfg["backbones"]:
        raise KeyError(f"backbone '{key}' not in config.backbones {list(cfg['backbones'])}")
    entry = copy.deepcopy(cfg["backbones"][key])
    train = copy.deepcopy(cfg["train"])
    train.update(entry.pop("overrides", {}) or {})
    return {"key": key, "timm": entry["timm"], "img_size": entry.get("img_size"), "train": train}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_dir: str | None = None, name: str = "run") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Submission validation
# --------------------------------------------------------------------------- #
def validate_submission(
    sub: pd.DataFrame,
    *,
    test_dir: str | None = None,
    template_path: str | None = None,
    num_classes: int = 100,
) -> None:
    """Hard-fail the build if submission shape/IDs/labels are wrong.

    Prefer test_dir (all images in Data/test/, sorted 0..N-1). template_path is legacy.
    Checks: expected row count, identical ID set & order, header ['ID','Label'],
    integer labels in 0..num_classes-1, no NaNs.
    """
    if test_dir:
        import data as D

        expected_ids = D.load_all_test_ids(test_dir)
    elif template_path:
        expected_ids = pd.read_csv(template_path)["ID"].astype(str).tolist()
    else:
        raise ValueError("validate_submission requires test_dir or template_path")

    assert list(sub.columns) == ["ID", "Label"], f"columns must be ['ID','Label'], got {list(sub.columns)}"
    assert len(sub) == len(expected_ids), (
        f"row count {len(sub)} != expected {len(expected_ids)} "
        f"(Kaggle needs every file in test_dir)"
    )
    assert list(sub["ID"].astype(str)) == expected_ids, (
        "ID column must list every test image, sorted numerically (0.jpg, 1.jpg, …)"
    )
    assert sub["Label"].notna().all(), "Label has NaNs"
    labels = sub["Label"].astype(int)
    hi = num_classes - 1
    assert ((labels >= 0) & (labels <= hi)).all(), f"labels must be in 0..{hi}"
    assert (labels == sub["Label"]).all(), "labels must be integers"
