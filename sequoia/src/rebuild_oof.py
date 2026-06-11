"""Rebuild OOF logits from saved fold checkpoints.

Use this when training completed folds but crashed before writing `oof.npz`
(for example, disk full while saving the last checkpoint).

Usage:
  PYTHONPATH=src python src/rebuild_oof.py --backbone convnext_small --ckpt-dir checkpoints_local --device mps --batch-size 8
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

import data as D
import model as M
import utils as U
from train import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--ckpt-dir", default=None, help="default config.output.ckpt_dir")
    p.add_argument("--backbone", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--prefer-raw", action="store_true", help="prefer raw over EMA checkpoints")
    return p.parse_args()


def loadable(path: str) -> bool:
    try:
        torch.load(path, map_location="cpu")
        return True
    except Exception:
        return False


def find_fold_ckpts(ckpt_dir: str, backbone: str, prefer_raw: bool = False) -> dict[int, str]:
    root = os.path.join(ckpt_dir, backbone)
    by_fold: dict[int, dict[str, str]] = {}
    for name in os.listdir(root):
        m = re.match(r"fold(\d+)_(ema|raw)\.pt$", name)
        if not m:
            continue
        by_fold.setdefault(int(m.group(1)), {})[m.group(2)] = os.path.join(root, name)

    out = {}
    order = ["raw", "ema"] if prefer_raw else ["ema", "raw"]
    for fold, paths in sorted(by_fold.items()):
        for tag in order:
            path = paths.get(tag)
            if path and loadable(path):
                out[fold] = path
                break
        if fold not in out:
            raise RuntimeError(f"no loadable checkpoint for {backbone} fold {fold}")
    return out


def main():
    args = parse_args()
    cfg = U.load_config(args.config)
    ckpt_dir = args.ckpt_dir or cfg["output"]["ckpt_dir"]
    device = U.get_device(args.device)
    U.seed_everything(cfg["seed"])

    num_classes = cfg["data"]["num_classes"]
    paths, labels = D.build_samples(cfg["data"]["train_dir"], num_classes)
    fold_ckpts = find_fold_ckpts(ckpt_dir, args.backbone, prefer_raw=args.prefer_raw)
    batch_size = args.batch_size or cfg["train"]["batch_size"]

    oof_logits = np.full((len(paths), num_classes), np.nan, dtype=np.float32)
    fold_accs = {}

    for fold, _tr_idx, va_idx in D.stratified_folds(labels, cfg["cv"]["n_splits"], cfg["seed"], cfg["cv"]["shuffle"]):
        if fold not in fold_ckpts:
            continue
        ck = torch.load(fold_ckpts[fold], map_location="cpu")
        net = M.create_model(
            ck["timm_name"],
            num_classes=ck["num_classes"],
            pretrained=False,
            img_size=ck.get("img_size"),
        ).to(device)
        net.load_state_dict(ck["state_dict"])
        tf_eval = D.build_transforms(net, ck.get("img_size"), cfg["train"]["aug"], train=False)
        ds_val = D.TrainDataset(paths, labels, va_idx, tf_eval)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            num_workers=cfg["train"]["num_workers"], pin_memory=device.type == "cuda")
        acc, logits, _targets = evaluate(net, dl_val, device)
        oof_logits[va_idx] = logits
        fold_accs[fold] = acc
        print(f"fold {fold}: ckpt={os.path.basename(fold_ckpts[fold])} acc={acc:.4f}")

    covered = ~np.isnan(oof_logits).any(axis=1)
    oof_acc = float((oof_logits[covered].argmax(1) == labels[covered]).mean())
    out_dir = os.path.join(ckpt_dir, args.backbone)
    np.savez(os.path.join(out_dir, "oof.npz"), logits=oof_logits, labels=labels, covered=covered)
    print(f"rebuilt OOF: acc={oof_acc:.4f} covered={covered.sum()}/{len(labels)} fold_accs={fold_accs}")


if __name__ == "__main__":
    main()
