"""OOF ensemble reporter.

Loads per-backbone `oof.npz` files produced by train.py, aligns covered samples,
averages softmax probabilities, and reports individual + ensemble OOF accuracy.

Usage:
  python src/oof_ensemble.py --backbones convnext_small,convnext_base
  python src/oof_ensemble.py --ckpt-dir /path/to/checkpoints --backbones convnext_small,convnext_base
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import utils as U


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--ckpt-dir", default=None, help="default config.output.ckpt_dir")
    p.add_argument("--backbones", required=True, help="comma-separated backbone keys")
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = logits.astype(np.float64) / temperature
    x = x - np.nanmax(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def load_oof(ckpt_dir: str, backbone: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = os.path.join(ckpt_dir, backbone, "oof.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing OOF file for {backbone}: {path}")
    z = np.load(path)
    return z["logits"], z["labels"], z["covered"].astype(bool)


def accuracy(probs_or_logits: np.ndarray, labels: np.ndarray) -> float:
    return float((probs_or_logits.argmax(1) == labels).mean())


def per_class_recall(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    preds = probs.argmax(1)
    recalls = np.full(num_classes, np.nan, dtype=np.float64)
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            recalls[cls] = float((preds[mask] == cls).mean())
    return recalls


def main():
    args = parse_args()
    cfg = U.load_config(args.config)
    ckpt_dir = args.ckpt_dir or cfg["output"]["ckpt_dir"]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    num_classes = int(cfg["data"]["num_classes"])

    loaded = []
    labels_ref = None
    covered_all = None
    for backbone in backbones:
        logits, labels, covered = load_oof(ckpt_dir, backbone)
        if labels_ref is None:
            labels_ref = labels
            covered_all = covered.copy()
        else:
            if not np.array_equal(labels_ref, labels):
                raise ValueError(f"labels in {backbone}/oof.npz do not match first backbone")
            covered_all &= covered
        loaded.append((backbone, logits, covered))

    assert labels_ref is not None and covered_all is not None
    labels = labels_ref[covered_all]
    if len(labels) == 0:
        raise ValueError("no common covered OOF samples across requested backbones")

    print(f"OOF ensemble report | ckpt_dir={ckpt_dir} | covered={len(labels)}/{len(labels_ref)}")
    probs_sum = np.zeros((len(labels), num_classes), dtype=np.float64)
    for backbone, logits, _covered in loaded:
        probs = softmax_np(logits[covered_all], temperature=args.temperature)
        probs_sum += probs
        print(f"{backbone}: acc={accuracy(probs, labels):.4f}")

    ens_probs = probs_sum / len(loaded)
    ens_acc = accuracy(ens_probs, labels)
    recalls = per_class_recall(ens_probs, labels, num_classes)
    tail = np.argsort(np.nan_to_num(recalls, nan=-1.0))[:10]
    print(f"ensemble_equal: acc={ens_acc:.4f} temperature={args.temperature}")
    print("lowest_recall_classes:", ", ".join(f"{c}:{recalls[c]:.2f}" for c in tail))


if __name__ == "__main__":
    main()
