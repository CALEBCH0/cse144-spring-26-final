import csv
import os
import time
from collections import Counter

# Load .env for HF_TOKEN (needed for gated models like DINOv3)
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    with open(_env) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from collections import Counter as _Counter
from datasets import Dataset as _Dataset, concatenate_datasets as _concat_ds, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, set_seed


def _upsample_to_minimum(raw_fold, min_count=20):
    labels = raw_fold["label"]
    counts = _Counter(labels)
    extras = {"image": [], "label": []}
    for cls, count in counts.items():
        needed = min_count - count
        if needed > 0:
            idx = [i for i, l in enumerate(labels) if l == cls]
            for j in range(needed):
                extras["image"].append(raw_fold[idx[j % len(idx)]]["image"])
                extras["label"].append(cls)
    if extras["image"]:
        extra_ds = _Dataset.from_dict(extras, features=raw_fold.features)
        upsampled = _concat_ds([raw_fold, extra_ds])
        n_added = len(extras["image"])
        print(f"  Upsampled {sum(1 for c in counts if counts[c]<min_count)} rare classes "
              f"(+{n_added} imgs, {len(raw_fold)}->{len(upsampled)})")
        return upsampled
    return raw_fold

from config import GPU_MEMORY_FRACTION, N_FOLDS, SEED, TRAIN_DIR
from models import MODELS
from pipeline import collate_fn, compute_metrics
from transforms import build_transforms
from utils import (
    CutMixCollator,
    CustomTrainer,
    EpochAccuracyCallback,
    MixupCollator,
    MixupCutMixCollator,
    apply_freeze as _apply_freeze,
    apply_lora,
    get_test_probs,
    predict_test_images,
    signal,
    top_confusion_pairs,
)

RESULTS_FILE = "search_results.csv"
SUBMISSIONS_DIR = "search_results"

# ── Configs to run ────────────────────────────────────────────────────────────
# Set to None to run everything not yet in the CSV (full resume mode).
# Set to a set of indices to run only those specific configs.
# Completed configs (already in search_results.csv) are always skipped regardless.
CONFIGS_TO_RUN = {
    # DINOv2 unfreeze=2 LR sweep
    # 44, 45, 46, 47,
    # DINOv2 unfreeze=2 LLRD sweep
    # 48, 49, 50,
    # DINOv2 unfreeze=2 augmentation ablation
    # 51, 52,
    # ViT unfreeze=4 augmentation ablation (best ViT: LLRD=0.85)
    # 53, 54, 55,
    # DINOv2-Large: unfreeze depth, LR sweep, epochs, augmentation
    # 56, 57, 58, 59, 60, 61, 62, 63,
    # DINOv2-base: best lr + best LLRD combos, more unfreeze, longer training
    # 64, 65, 66, 67, 68, 69,
    # DINOv2-Large + DINOv2-base: combine best findings, overnight run
    # 70, 71, 72, 73, 74, 75, 76, 77,
    # More epochs + LLRD sweep + cosine + deeper unfreeze for Large; base cosine
    # 78, 79, 80, 81, 82, 83, 84,  # done / currently running
    # 518px resolution + RandAugment combos
    # 85, 86, 87,  # currently running
    # SigLIP2 + upsampling
    121,
}

# fmt: off
SEARCH_CONFIGS = [
    # ── DINOv2 unfreeze depth ─────────────────────────────────────
    {"model": "dinov2", "unfreeze_blocks": 0,  "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 2,  "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 4,  "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 12, "lr": 1e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── ConvNeXt-Base freeze strategy ────────────────────────────
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 1e-3,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 1, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 2, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── ConvNeXt-Base LR sweep (full fine-tune) ───────────────────
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 1e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 3e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 1e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 3e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── ViT LR sweep ──────────────────────────────────────────────
    {"model": "vit", "unfreeze_blocks": 0, "lr": 1e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "vit", "unfreeze_blocks": 0, "lr": 3e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "vit", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "vit", "unfreeze_blocks": 0, "lr": 1e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── LR scheduler ──────────────────────────────────────────────
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 1e-4,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "vit",      "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "vit",      "unfreeze_blocks": 0, "lr": 1e-4,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── Label smoothing ───────────────────────────────────────────
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.0,  "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.1,  "weight_decay": 0.01},

    # ── Weight decay ──────────────────────────────────────────────
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.05},
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.1},
    {"model": "vit",      "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.05},

    # ── LLRD decay factor ─────────────────────────────────────────
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.65},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.75},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85},
    {"model": "vit",      "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.65},
    {"model": "vit",      "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.75},
    {"model": "vit",      "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85},

    # ── Augmentation ablation (convnext, unfreeze=4, cosine) ──────
    # baseline: no extra aug, no mixup/cutmix
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": False, "use_randaugment": False, "use_random_erasing": False, "collator": "none"},
    # +color jitter only
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True,  "use_randaugment": False, "use_random_erasing": False, "collator": "none"},
    # +RandAugment only
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": False, "use_randaugment": True,  "use_random_erasing": False, "collator": "none"},
    # full aug stack (CJ + RA + RE)
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True,  "use_randaugment": True,  "use_random_erasing": True,  "collator": "none"},
    # full aug + mixup
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True,  "use_randaugment": True,  "use_random_erasing": True,  "collator": "mixup"},

    # ── CutMix vs Mixup (convnext, full aug stack) ────────────────
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True,  "use_randaugment": True,  "use_random_erasing": True,  "collator": "cutmix"},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True,  "use_randaugment": True,  "use_random_erasing": True,  "collator": "mixup_cutmix"},

    # ── LoRA rank sweep — DINOv2 ──────────────────────────────────
    {"model": "dinov2", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 4},
    {"model": "dinov2", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 8},
    {"model": "dinov2", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 16},

    # ── LoRA rank sweep — ViT ─────────────────────────────────────
    {"model": "vit", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 4},
    {"model": "vit", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 8},
    {"model": "vit", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 16},

    # ── DINOv2 LR sweep (unfreeze=2, best depth) ──────────────────
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 3e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 3e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── DINOv2 LLRD (unfreeze=2, lr=5e-5, cosine) ────────────────
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.65},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.75},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85},

    # ── DINOv2 augmentation (unfreeze=2, lr=5e-5, linear) ─────────
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_randaugment": True},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True, "use_randaugment": True, "use_random_erasing": True},

    # ── ViT augmentation (unfreeze=4, LLRD=0.85, best ViT config) ─
    {"model": "vit", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85,
     "use_randaugment": True},
    {"model": "vit", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85,
     "use_color_jitter": True},
    {"model": "vit", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85,
     "use_color_jitter": True, "use_randaugment": True, "use_random_erasing": True},

    # ── DINOv2-Large: unfreeze depth (mirror best-of-Base findings) ─
    {"model": "dinov2_large", "unfreeze_blocks": 0,  "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2_large", "unfreeze_blocks": 2,  "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2_large", "unfreeze_blocks": 4,  "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── DINOv2-Large: LR sweep at unfreeze=2 ──────────────────────
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── DINOv2-Large: more epochs at best unfreeze ─────────────────
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "num_epochs": 20},
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "num_epochs": 30},

    # ── DINOv2-Large: RandAugment at best unfreeze ────────────────
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_randaugment": True},

    # ── DINOv2-base: combine best lr (1e-4) + best LLRD (0.85) ────  idx 64-69
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75},
    {"model": "dinov2", "unfreeze_blocks": 4, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85},
    {"model": "dinov2", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85},
    {"model": "dinov2", "unfreeze_blocks": 6, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 20},

    # ── DINOv2-Large: combine best findings for overnight run ──────  idx 70-77
    # A: best lr + best LLRD + best epochs (top priority)
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 30},
    # B: same but 20ep — faster sanity check
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 20},
    # C: alt LLRD at full power
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30},
    # D: aug helped Large — combine with best settings
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 30, "use_randaugment": True},
    # E: confirm LLRD effect at 30ep (no LLRD baseline)
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "num_epochs": 30},
    # F: add LLRD to current best config (lr=5e-5, 30ep)
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 30},
    # G: base best config + 20ep
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 20},
    # H: base best config + 30ep
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 30},

    # ── Next search: more epochs + LLRD sweep + cosine + deeper unfreeze ──  idx 78-84
    # 78: continue LLRD sweep — 0.85→0.75 gained +0.19%; does 0.65 continue the trend?
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.65, "num_epochs": 30},
    # 79: 40ep — Large still improving at 30ep (+1.2% over 20ep); expect +0.3-0.5% more
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 40},
    # 80: 50ep — queue alongside 40ep to avoid wasting a night if 40ep still gains
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 50},
    # 81: cosine at 30ep — long training commonly benefits from cosine LR decay
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30},
    # 82: unfreeze=4 for Large — 4/24 layers is only 17% (vs 33% for base); safe with LLRD=0.65
    {"model": "dinov2_large", "unfreeze_blocks": 4, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.65, "num_epochs": 30},
    # 83: push LR higher — needs even gentler LLRD to protect lower layers
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 2e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.65, "num_epochs": 30},
    # 84: base cosine — only untested axis for DINOv2-base
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 10},

    # ── 518px resolution + RandAugment combos ─────────────────────────  idx 85-87
    # 85: dinov2_large at native 518px (patch_size=14 → 37×37 patches vs 16×16 at 224px)
    #     batch_size=16 to stay within VRAM (5× more tokens → higher activation memory)
    {"model": "dinov2_large_518", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30, "batch_size": 16},
    # 86: RandAugment on 224px Large — +0.93% on base but never tested on Large; isolate aug effect
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30, "use_randaugment": True},
    # 87: 518px + RandAugment — combined; only run after 85/86 confirm both are individually positive
    {"model": "dinov2_large_518", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30, "use_randaugment": True, "batch_size": 16},

    # ── SigLIP baselines ───────────────────────────────────────────────────  idx 88-90
    # 88: SigLIP baseline — same conservative settings as best DINOv2-base config
    {"model": "siglip", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    # 89: + class weights — SigLIP has 768-dim features; minority classes may benefit most
    {"model": "siglip", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_class_weights": True},
    # 90: more epochs — SigLIP pretrained on broader data; may need more fine-tuning time
    {"model": "siglip", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "num_epochs": 20},

    # ── CLIP-ViT-Large optimised configs ──────────────────────────────────  idx 91-93
    # 91: unfreeze=2, LLRD=0.75, 30ep — DINOv2-Large sweet spot; image-text pretraining → shallow unfreeze
    {"model": "clip_vit", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30},
    # 92: unfreeze=4, LLRD=0.75, 30ep — DINOv3 (same ViT-L arch) peaked at unfreeze=4; tests if arch > pretraining
    {"model": "clip_vit", "unfreeze_blocks": 4, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30},
    # 93: unfreeze=4, LLRD=0.75, 50ep — extend best config; likely +0.5-1pp
    {"model": "clip_vit", "unfreeze_blocks": 4, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 50},

    # ── DINOv2-Large best config + class weights ───────────────────────────  idx 94-95
    # 94: cfg-80 best (50ep, LLRD=0.75) + class weights — does balancing help large model?
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 50, "use_class_weights": True},
    # 95: cfg-80 best + class weights + RandAugment — combined accessories
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 50, "use_class_weights": True, "use_randaugment": True},

    # ── SigLIP2-SO400M-384 (top-team backbone) ────────────────────────────  idx 96-102
    # Top team: frozen probe on this backbone alone → 94.5% OOF; fine-tuned → 95.4%
    # SO400M = 400M-param ViT-So400M, 27 transformer layers, pretrained at 384px on image-text pairs

    # 96: conservative baseline — 2 blocks, same settings as DINOv2 best
    {"model": "siglip2_so400m", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.8},
    # 97: their unfreeze depth — 6/27 blocks; batch_size=16 (unfreeze=6 OOMs at 32 on T4/local)
    {"model": "siglip2_so400m", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.8, "batch_size": 16},
    # 98: their exact scheduler + epoch combo — cosine 15ep; batch_size=16
    {"model": "siglip2_so400m", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.1, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 15, "batch_size": 16},
    # 99: + MixUp — top team used MixUp/CutMix successfully on SigLIP2; isolate MixUp effect
    {"model": "siglip2_so400m", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.1, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 15, "collator": "mixup", "batch_size": 16},
    # 100: + CutMix — isolate CutMix effect
    {"model": "siglip2_so400m", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.1, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 15, "collator": "cutmix", "batch_size": 16},
    # 101: + class weights — does balancing help SO400M tail classes?
    {"model": "siglip2_so400m", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.1, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 15, "use_class_weights": True, "batch_size": 16},
    # 102: SO400M-512px — larger spatial resolution; batch_size=8 for VRAM
    {"model": "siglip2_so400m_512", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.1, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 15, "batch_size": 8},

    # ── SigLIP2-SO400M hyperparameter sweep (local) ─────────────────────  idx 103-107
    # Kaggle covers: 96 (unfreeze=2,10ep), 97 (unfreeze=6,10ep), 98 (unfreeze=6,15ep,cosine)
    # Local fills: depth=4, depth=8, 20ep variants, conservative lr

    # 103: unfreeze=4 — intermediate depth between Kaggle's 2 and 6
    {"model": "siglip2_so400m", "unfreeze_blocks": 4, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 15},
    # 104: unfreeze=2, 20ep — baseline + more epochs; fold 1/3 still improving at ep10 in cfg 96
    {"model": "siglip2_so400m", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 20},
    # 105: unfreeze=6, 20ep — extend Kaggle cfg 97 to 20ep; batch_size=16 for unfreeze≥6
    {"model": "siglip2_so400m", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 20, "batch_size": 16},
    # 106: unfreeze=8, LLRD=0.75 — deeper unfreeze; batch_size=16
    {"model": "siglip2_so400m", "unfreeze_blocks": 8, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 15, "batch_size": 16},
    # 107: unfreeze=6, lr=5e-5 — conservative LR at top-team depth; batch_size=16
    {"model": "siglip2_so400m", "unfreeze_blocks": 6, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 15, "batch_size": 16},

    # ── SigLIP2 longer training sweep ──────────────────────────────────────  idx 108-111
    # cfg 104 (best): unfreeze=2, 20ep → 95.00%. Does training longer help?
    # 108: unfreeze=2, 30ep — +10ep beyond current best
    {"model": "siglip2_so400m", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 30},
    # 109: unfreeze=2, 40ep — +20ep; test for overfitting vs further gain
    {"model": "siglip2_so400m", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 40},
    # 110: siglip2_so400m_512, 20ep — baseline 512px used 15ep; try +5ep
    {"model": "siglip2_so400m_512", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.1, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 20, "batch_size": 8},
    # 111: siglip2_so400m_512, 30ep — longer training for 512px
    {"model": "siglip2_so400m_512", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.1, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 30, "batch_size": 8},
    # 112: siglip2_so400m_512, unfreeze=2 — fix: we used unfreeze=6 (competitor-inspired) but
    #      unfreeze=2 is best for 384px; test if it rescues 512px from bad config
    {"model": "siglip2_so400m_512", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.1, "weight_decay": 0.01,
     "llrd_factor": 0.8, "num_epochs": 20, "batch_size": 8},

    # ── DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m) ─────────  idx 113-118
    # ViT-L, 300M params, pretrained on 1.689B images (12× DINOv2-Large's 142M).
    # Distilled from ViT-7B teacher. Same size as DINOv2-Large → similar training time.
    # DINOv2-Large best: unfreeze=2, lr=1e-4, linear, 50ep, llrd=0.75 → 87.39%.
    # Ordered fastest → slowest (20ep before 30ep, smaller unfreeze first).
    # Key questions: (1) does stronger pretraining allow more unfreezing?
    #                (2) does it converge faster (fewer epochs needed)?
    #                (3) lower lr to preserve richer features?

    # 113: baseline — same as dinov2_large best but 20ep; establishes DINOv3 floor (~95min)
    {"model": "dinov3", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 20},
    # 114: lower LR — gentler updates to preserve richer DINOv3 features (~95min)
    {"model": "dinov3", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 20},
    # 115: unfreeze=4, 20ep — does stronger pretraining tolerate deeper fine-tuning? (~97min)
    {"model": "dinov3", "unfreeze_blocks": 4, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 20},
    # 116: unfreeze=6, 20ep — deep fine-tuning; risky on 1079 images but worth one test (~100min)
    {"model": "dinov3", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 20},
    # 117: unfreeze=2, 50ep — full training at conservative depth (matches dinov2_large best) (~46min)
    {"model": "dinov3", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 50},
    # 118: unfreeze=4, 50ep — deeper unfreeze at full training budget (~46min)
    {"model": "dinov3", "unfreeze_blocks": 4, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 50},
    # 119: unfreeze=4, lr=5e-5, 50ep — lower LR cross with deeper unfreeze + full epochs (~46min)
    {"model": "dinov3", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 50},

    # ── CLIP-ViT extended search ───────────────────────────────────────────  idx 120
    # 120: unfreeze=6, LLRD=0.65, 50ep — DINOv3-style deeper unfreeze; tests arch-driven behavior
    {"model": "clip_vit", "unfreeze_blocks": 6, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.65, "num_epochs": 50},

    # ── SigLIP2-SO400M + upsampling ───────────────────────────────────────  idx 121
    # 121: best SigLIP2 config (unfreeze=2, 20ep, LLRD=0.8) + upsample min=20
    #      DINOv2-Large got +2.32pp from upsampling; SigLIP2 effect unknown
    {"model": "siglip2_so400m", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05,
     "weight_decay": 0.01, "llrd_factor": 0.8, "num_epochs": 20, "batch_size": 16, "upsample": True},
]
# fmt: on

NUM_EPOCHS = 10
BATCH_SIZE = 32


def run_cv(cfg, model_cfg, full_dataset, fold_splits, class_names, num_labels, label2id, id2label, cfg_index=0):
    image_processor = model_cfg["get_processor"]()
    train_tf, val_tf = build_transforms(
        image_processor,
        use_color_jitter=cfg.get("use_color_jitter", False),
        use_randaugment=cfg.get("use_randaugment", False),
        use_random_erasing=cfg.get("use_random_erasing", False),
    )
    all_label_ids = list(range(num_labels))
    fold_per_class_f1 = []

    # Collator selection
    collator_type = cfg.get("collator", "none")
    if collator_type == "mixup":
        data_collator = MixupCollator(num_labels, alpha=0.2)
    elif collator_type == "cutmix":
        data_collator = CutMixCollator(num_labels, alpha=1.0)
    elif collator_type == "mixup_cutmix":
        data_collator = MixupCutMixCollator(num_labels, mixup_alpha=0.2, cutmix_alpha=1.0)
    else:
        data_collator = collate_fn

    fold_metrics = []
    fold_train_sec = []
    fold_eval_sec = []
    fold_models = []
    pooled_true = []
    pooled_pred = []
    oof_logits  = np.zeros((len(full_dataset), num_labels), dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        signal("fold_start", config=cfg_index, fold=fold_idx + 1, n_folds=N_FOLDS, model=cfg["model"])
        train_raw  = full_dataset.select(train_idx)
        if cfg.get("upsample", False):
            train_raw = _upsample_to_minimum(train_raw, min_count=20)
        train_fold = train_raw.with_transform(train_tf)
        val_fold   = full_dataset.select(val_idx).with_transform(val_tf)

        model = model_cfg["get_model"](num_labels, label2id, id2label)

        lora_r = cfg.get("lora_r", 0)
        if lora_r > 0:
            model = apply_lora(model, cfg["model"], r=lora_r)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            trainable = _apply_freeze(model, cfg["model"], cfg["unfreeze_blocks"])

        fold_batch_size = cfg.get("batch_size", BATCH_SIZE)
        training_args = TrainingArguments(
            output_dir=os.path.join(model_cfg["output_dir"], "search", f"fold_{fold_idx}"),
            num_train_epochs=cfg.get("num_epochs", NUM_EPOCHS),
            per_device_train_batch_size=fold_batch_size,
            per_device_eval_batch_size=fold_batch_size,
            learning_rate=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            label_smoothing_factor=cfg["label_smoothing"],
            lr_scheduler_type=cfg["scheduler"],
            warmup_steps=int(0.1 * cfg.get("num_epochs", NUM_EPOCHS) * (863 // fold_batch_size)) if cfg["scheduler"] == "cosine" else 0,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            seed=SEED,
            remove_unused_columns=False,
            logging_strategy="no",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        # LLRD disabled when LoRA active (PEFT renames params, breaking layer pattern matching)
        llrd_factor = cfg.get("llrd_factor", 1.0) if lora_r == 0 else 1.0
        use_soft_labels = collator_type in ("mixup", "cutmix", "mixup_cutmix")

        class_weights = None
        if cfg.get("use_class_weights", False):
            counts = Counter(full_dataset["label"])
            cw = torch.zeros(num_labels)
            for ci in range(num_labels):
                cw[ci] = 1.0 / max(counts.get(ci, 1), 1)
            class_weights = cw / cw.mean()

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            compute_metrics=compute_metrics,
            processing_class=image_processor,
            data_collator=data_collator,
            llrd_factor=llrd_factor,
            model_name=cfg["model"],
            use_mixup=use_soft_labels,
            class_weights=class_weights,
        )

        train_eval_fold = full_dataset.select(train_idx).with_transform(val_tf)
        trainer.add_callback(EpochAccuracyCallback(train_eval_fold, val_fold, collate_fn))

        try:
            t0 = time.perf_counter()
            trainer.train()
            fold_train_sec.append(time.perf_counter() - t0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    fold {fold_idx+1}: OOM — skipping fold")
                torch.cuda.empty_cache()
                continue
            raise

        try:
            trainer.data_collator = collate_fn  # plain collator — no mixup on val images
            t0 = time.perf_counter()
            preds_out = trainer.predict(val_fold)
            fold_eval_sec.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"    fold {fold_idx+1}: predict failed ({e}) — skipping fold")
            torch.cuda.empty_cache()
            continue

        preds = np.argmax(preds_out.predictions, axis=1)
        true  = preds_out.label_ids
        pooled_true.extend(true.tolist())
        pooled_pred.extend(preds.tolist())
        oof_logits[val_idx] = preds_out.predictions.astype(np.float32)

        per_class = f1_score(true, preds, labels=all_label_ids, average=None, zero_division=0)
        fold_per_class_f1.append(per_class)
        fold_metrics.append({
            "accuracy":  accuracy_score(true, preds),
            "precision": precision_score(true, preds, average="macro", zero_division=0),
            "recall":    recall_score(true, preds, average="macro", zero_division=0),
            "f1":        float(per_class.mean()),
        })
        fold_models.append(model.cpu())
        torch.cuda.empty_cache()
        tr = fold_train_sec[-1] if fold_train_sec else 0
        ev = fold_eval_sec[-1] if fold_eval_sec else 0
        print(f"    fold {fold_idx+1}: acc={fold_metrics[-1]['accuracy']:.4f}  f1={fold_metrics[-1]['f1']:.4f}"
              f"  train={tr:.0f}s  eval={ev:.0f}s  trainable={trainable:,}")
        signal("fold_done", config=cfg_index, fold=fold_idx + 1, model=cfg["model"],
               acc=round(fold_metrics[-1]["accuracy"], 4), f1=round(fold_metrics[-1]["f1"], 4),
               train_sec=round(tr))

    if not fold_metrics:
        raise RuntimeError("All folds failed — cannot compute CV metrics")

    mean_train_sec = float(np.mean(fold_train_sec)) if fold_train_sec else 0.0
    mean_eval_sec  = float(np.mean(fold_eval_sec))  if fold_eval_sec  else 0.0
    mean_per_class_f1 = np.mean(fold_per_class_f1, axis=0) if fold_per_class_f1 else np.zeros(num_labels)
    confusion_pairs = top_confusion_pairs(pooled_true, pooled_pred, class_names) if pooled_true else []

    oof_probs = torch.softmax(torch.from_numpy(oof_logits), dim=-1).numpy()
    return (
        {m: float(np.mean([f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")},
        {m: float(np.std( [f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")},
        mean_train_sec,
        mean_eval_sec,
        fold_models,
        mean_per_class_f1,
        confusion_pairs,
        oof_probs,
    )


def write_row(writer, cfg, mean, std, mean_train_sec, mean_eval_sec, config_idx, total):
    row = {
        "config": config_idx,
        "model": cfg["model"],
        "unfreeze_blocks": cfg["unfreeze_blocks"],
        "lr": cfg["lr"],
        "scheduler": cfg["scheduler"],
        "label_smoothing": cfg["label_smoothing"],
        "weight_decay": cfg["weight_decay"],
        "llrd_factor": cfg.get("llrd_factor", 1.0),
        "color_jitter": cfg.get("use_color_jitter", False),
        "randaugment": cfg.get("use_randaugment", False),
        "random_erasing": cfg.get("use_random_erasing", False),
        "collator": cfg.get("collator", "none"),
        "lora_r": cfg.get("lora_r", 0),
        "acc_mean": f"{mean['accuracy']:.4f}",
        "acc_std":  f"{std['accuracy']:.4f}",
        "f1_mean":  f"{mean['f1']:.4f}",
        "f1_std":   f"{std['f1']:.4f}",
        "prec_mean": f"{mean['precision']:.4f}",
        "rec_mean":  f"{mean['recall']:.4f}",
        "train_sec": f"{mean_train_sec:.1f}",
        "eval_sec":  f"{mean_eval_sec:.1f}",
    }
    writer.writerow(row)


def main():
    set_seed(SEED)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        total_gb = gpu.total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        print(f"GPU: {gpu.name} ({total_gb:.1f} GB, {GPU_MEMORY_FRACTION*100:.0f}% = {total_gb*GPU_MEMORY_FRACTION:.1f} GB)")
    else:
        print("WARNING: No GPU — running on CPU")

    full_dataset = load_dataset("imagefolder", data_files={"train": os.path.join(TRAIN_DIR, "**")})["train"]
    class_names  = full_dataset.features["label"].names
    num_labels   = len(class_names)
    label2id     = {label: str(i) for i, label in enumerate(class_names)}
    id2label     = {str(i): label for i, label in enumerate(class_names)}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(range(len(full_dataset)), full_dataset["label"]))

    model_lookup = {cfg["name"]: cfg for cfg in MODELS}

    # Load SigLIP2 baseline per-class F1 once for routing diff output
    _baseline_f1 = None
    _baseline_path = os.path.join("probs", "val_probs_siglip2_so400m_r013.npy")
    _labels_path   = os.path.join("probs", "val_labels.npy")
    if os.path.exists(_baseline_path) and os.path.exists(_labels_path):
        import numpy as _np
        from sklearn.metrics import f1_score as _f1_score
        _bvp = _np.load(_baseline_path)
        _bvl = _np.load(_labels_path)
        _baseline_f1 = _f1_score(_bvl, _bvp.argmax(1),
                                  labels=list(range(num_labels)), average=None, zero_division=0)
        print(f"Routing baseline loaded: siglip2_so400m_r013 "
              f"(solo acc={(_bvp.argmax(1)==_bvl).mean():.4f})")

    fieldnames = ["config", "model", "unfreeze_blocks", "lr", "scheduler",
                  "label_smoothing", "weight_decay", "llrd_factor",
                  "color_jitter", "randaugment", "random_erasing", "collator", "lora_r",
                  "acc_mean", "acc_std", "f1_mean", "f1_std", "prec_mean", "rec_mean",
                  "train_sec", "eval_sec"]
    per_class_fieldnames = ["config", "model"] + [f"class_{i}_f1" for i in range(num_labels)]
    per_class_file = os.path.join(os.path.dirname(RESULTS_FILE) or ".", "search_per_class.csv")
    per_class_completed = set()
    if os.path.exists(per_class_file):
        with open(per_class_file, newline="") as _f:
            for _row in csv.DictReader(_f):
                per_class_completed.add(int(_row["config"]))

    # resume support — skip configs already written
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, newline="") as f:
            for row in csv.DictReader(f):
                completed.add(int(row["config"]))
        print(f"Resuming — {len(completed)} configs already done")

    pending = CONFIGS_TO_RUN - completed if CONFIGS_TO_RUN is not None else None
    if pending is not None:
        print(f"CONFIGS_TO_RUN: {sorted(CONFIGS_TO_RUN)}  |  pending: {sorted(pending)}")

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not completed:
            writer.writeheader()

        total = len(SEARCH_CONFIGS)
        for i, cfg in enumerate(SEARCH_CONFIGS):
            if i in completed:
                print(f"[{i+1}/{total}] skipping {cfg['model']} (already done)")
                continue
            if CONFIGS_TO_RUN is not None and i not in CONFIGS_TO_RUN:
                continue  # silently skip configs outside the current run set

            print(f"\n[{i+1}/{total}] {cfg['model']}  unfreeze={cfg['unfreeze_blocks']}  lr={cfg['lr']}  "
                  f"sched={cfg['scheduler']}  ls={cfg['label_smoothing']}  wd={cfg['weight_decay']}")
            signal("config_start", config=i, model=cfg["model"], lr=cfg["lr"],
                   unfreeze=cfg["unfreeze_blocks"], total=total)

            if cfg["model"] not in model_lookup:
                print(f"  [skip] unknown model '{cfg['model']}'")
                continue

            model_cfg = model_lookup[cfg["model"]]
            try:
                mean, std, mean_train_sec, mean_eval_sec, fold_models, mean_per_class_f1, confusion_pairs, oof_probs = run_cv(
                    cfg, model_cfg, full_dataset, fold_splits,
                    class_names, num_labels, label2id, id2label,
                    cfg_index=i,
                )
            except Exception as e:
                signal("error", config=i, model=cfg["model"], msg=str(e)[:60])
                print(f"  [fail] config {i}: {e} — skipping")
                torch.cuda.empty_cache()
                continue

            write_row(writer, cfg, mean, std, mean_train_sec, mean_eval_sec, i, total)
            f.flush()

            # ── per-class F1 CSV ──────────────────────────────────────────────
            if i not in per_class_completed:
                pc_row = {"config": i, "model": cfg["model"]}
                pc_row.update({f"class_{j}_f1": f"{v:.4f}" for j, v in enumerate(mean_per_class_f1)})
                pc_exists = os.path.exists(per_class_file)
                with open(per_class_file, "a", newline="") as pf:
                    pc_writer = csv.DictWriter(pf, fieldnames=per_class_fieldnames)
                    if not pc_exists:
                        pc_writer.writeheader()
                    pc_writer.writerow(pc_row)
                per_class_completed.add(i)

            # ── routing diff vs SigLIP2 baseline ─────────────────────────────
            ROUTING_DELTA = 0.10
            KNOWN_HARD = [63, 70, 76, 78, 86, 87, 88, 68, 74]
            if _baseline_f1 is not None:
                deltas = mean_per_class_f1 - _baseline_f1
                leads  = sorted([(j, float(mean_per_class_f1[j]), float(_baseline_f1[j]), float(deltas[j]))
                                  for j in range(num_labels) if deltas[j] >= ROUTING_DELTA],
                                 key=lambda x: x[3], reverse=True)
                print(f"  Routing vs SigLIP2 (delta >= {ROUTING_DELTA}):", end="")
                if leads:
                    print(f"  {len(leads)} class(es) —")
                    for j, this_f1, base_f1, d in leads:
                        hard_tag = " [HARD]" if j in KNOWN_HARD else ""
                        print(f"    class {j:>3}: this={this_f1:.3f}  base={base_f1:.3f}  delta={d:+.3f}{hard_tag}")
                else:
                    print("  none")
                # Always show known hard classes
                print(f"  Hard-class F1 (known problem classes):")
                print(f"    {'cls':>4}  {'this':>6}  {'base':>6}  {'delta':>7}")
                for j in KNOWN_HARD:
                    d = float(deltas[j])
                    marker = " <" if d >= ROUTING_DELTA else ""
                    print(f"    {j:>4}  {mean_per_class_f1[j]:>6.3f}  {_baseline_f1[j]:>6.3f}  {d:>+7.3f}{marker}")

            # ── hard-class summary ────────────────────────────────────────────
            HARD_THRESHOLD = 0.5
            hard = [(j, class_names[j], mean_per_class_f1[j])
                    for j in range(num_labels) if mean_per_class_f1[j] < HARD_THRESHOLD]
            hard.sort(key=lambda x: x[2])
            if hard:
                print(f"  Hard classes (F1 < {HARD_THRESHOLD}): {len(hard)}")
                for j, name, val in hard[:10]:
                    print(f"    class {j:>3} ({name}): F1={val:.3f}")
                if len(hard) > 10:
                    print(f"    ... and {len(hard)-10} more")

            # ── confusion pair analysis ───────────────────────────────────
            if confusion_pairs:
                print(f"  Top confused pairs (pooled {N_FOLDS}-fold val):")
                for c_a, c_b, n_err, rate in confusion_pairs[:5]:
                    print(f"    classes {c_a}<->{c_b}: {n_err:>3} errors, rate={rate:.2f}")

            # ── generate Kaggle submission CSV + save prob arrays ─────────
            data_root = os.path.dirname(TRAIN_DIR)
            test_dir = os.path.join(data_root, "test")
            if os.path.isdir(test_dir) and fold_models:
                os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
                _ip = model_cfg["get_processor"]()
                _, val_tf = build_transforms(_ip)
                test_ids = sorted(
                    [fn for fn in os.listdir(test_dir) if fn.lower().endswith(".jpg")],
                    key=lambda x: int(x.split(".")[0]),
                )
                image_paths = [os.path.join(test_dir, fn) for fn in test_ids]
                test_preds = predict_test_images(fold_models, image_paths, val_tf)
                sub_path = os.path.join(SUBMISSIONS_DIR, f"config_{i}_{cfg['model']}.csv")
                with open(sub_path, "w", newline="") as sf:
                    writer_sub = csv.writer(sf)
                    writer_sub.writerow(["ID", "Label"])
                    for img_id, pred in zip(test_ids, test_preds):
                        writer_sub.writerow([img_id, int(class_names[int(pred)])])
                print(f"  Submission saved: {sub_path}")

                # save prob arrays for class_router.py
                os.makedirs("probs", exist_ok=True)
                _llrd_int = int(cfg.get("llrd_factor", 1.0) * 100)
                _ep = cfg.get("num_epochs", NUM_EPOCHS)
                prob_tag = f"{cfg['model']}_u{cfg['unfreeze_blocks']}_ep{_ep}_llrd{_llrd_int}_cfg{i}"
                val_probs_path  = os.path.join("probs", f"val_probs_{prob_tag}.npy")
                test_probs_path = os.path.join("probs", f"test_probs_{prob_tag}.npy")
                np.save(val_probs_path, oof_probs)
                test_probs = get_test_probs(fold_models, image_paths, val_tf)
                np.save(test_probs_path, test_probs)
                # save val labels once (used by class_router.py)
                val_labels_path = os.path.join("probs", "val_labels.npy")
                if not os.path.exists(val_labels_path):
                    np.save(val_labels_path, np.array(full_dataset["label"]))
                print(f"  Prob arrays saved: {val_probs_path}, {test_probs_path}")

                del fold_models
                torch.cuda.empty_cache()

            print(f"  -> acc={mean['accuracy']:.4f}+/-{std['accuracy']:.4f}  f1={mean['f1']:.4f}+/-{std['f1']:.4f}"
                  f"  train={mean_train_sec:.0f}s/fold  eval={mean_eval_sec:.0f}s/fold")
            signal("config_done", config=i, model=cfg["model"],
                   acc=round(mean["accuracy"], 4), f1=round(mean["f1"], 4))

    signal("search_done", total=total, results=RESULTS_FILE)
    print(f"\nDone. Results saved to {RESULTS_FILE}")

    # ── Summary of configs run this session ───────────────────────────────────
    if CONFIGS_TO_RUN is not None and os.path.exists(RESULTS_FILE):
        ran = []
        with open(RESULTS_FILE, newline="") as f_in:
            for row in csv.DictReader(f_in):
                if int(row["config"]) in CONFIGS_TO_RUN:
                    ran.append(row)
        if ran:
            ran.sort(key=lambda r: float(r["acc_mean"]), reverse=True)
            print(f"\n{'='*74}")
            print(f"  Results for this run ({len(ran)} configs), sorted by acc")
            print(f"{'='*74}")
            print(f"  {'cfg':>3}  {'model':<14} {'unfrz':>5} {'lr':>7} {'llrd':>5} {'extras':<16} {'acc':>7} {'f1':>7}")
            print(f"  {'-'*70}")
            for r in ran:
                extras = []
                if r.get("randaugment") == "True": extras.append("rand")
                if r.get("color_jitter") == "True": extras.append("cj")
                if r.get("random_erasing") == "True": extras.append("re")
                if r.get("collator", "none") != "none": extras.append(r["collator"])
                if r.get("lora_r", "0") != "0": extras.append(f"lora_r={r['lora_r']}")
                cfg_entry = SEARCH_CONFIGS[int(r["config"])]
                if cfg_entry.get("num_epochs", NUM_EPOCHS) != NUM_EPOCHS:
                    extras.append(f"ep={cfg_entry.get('num_epochs')}")
                print(f"  {r['config']:>3}  {r['model']:<14} {r['unfreeze_blocks']:>5} {float(r['lr']):>7.0e}"
                      f" {float(r['llrd_factor']):>5.2f}  {' '.join(extras) if extras else '-':<16}"
                      f" {float(r['acc_mean']):>7.4f} {float(r['f1_mean']):>7.4f}")
            print(f"{'='*74}")


if __name__ == "__main__":
    main()
