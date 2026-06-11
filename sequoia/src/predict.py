"""Inference: load every fold checkpoint of every ensemble backbone, run TTA
(hflip + optional multi-scale), average softmax across folds x backbones x TTA views
(optionally OOF-weighted, OOF-tuned temperature), and write a validated submission.csv
for every image in Data/test/ (1036 IDs as of Spring 2026).

Usage:
  python src/predict.py                                  # use config.ensemble, EMA ckpts
  python src/predict.py --backbones convnext_small --out submission.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data as D
import model as M
import utils as U


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--ckpt-dir", default=None, help="default config.output.ckpt_dir")
    p.add_argument("--backbones", default=None, help="comma list (default config.ensemble)")
    p.add_argument("--template", default=None, help="legacy: use sample_submission IDs (1000) instead of all test images")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--raw", action="store_true", help="use raw (non-EMA) checkpoints")
    return p.parse_args()


def find_ckpts(ckpt_dir: str, backbone: str, use_ema: bool) -> list[str]:
    """Return one checkpoint per fold, preferring EMA and falling back to raw."""
    root = os.path.join(ckpt_dir, backbone)
    by_fold: dict[int, dict[str, str]] = {}
    for path in glob.glob(os.path.join(root, "fold*_*.pt")):
        m = re.search(r"fold(\d+)_(ema|raw)\.pt$", os.path.basename(path))
        if not m:
            continue
        fold, tag = int(m.group(1)), m.group(2)
        by_fold.setdefault(fold, {})[tag] = path

    ckpts = []
    for fold in sorted(by_fold):
        choices = (["ema", "raw"] if use_ema else ["raw", "ema"])
        for tag in choices:
            if tag in by_fold[fold]:
                ckpts.append(by_fold[fold][tag])
                break
    return ckpts


def oof_accuracy(ckpt_dir: str, backbone: str) -> float | None:
    path = os.path.join(ckpt_dir, backbone, "oof.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    cov = z["covered"]
    if not cov.any():
        return None
    return float((z["logits"][cov].argmax(1) == z["labels"][cov]).mean())


@torch.no_grad()
def member_softmax(model, loader, device, tta_hflip: bool, temperature: float) -> np.ndarray:
    """Mean softmax over TTA views for one checkpoint. Returns [N, C]."""
    model.eval()
    out = []
    for imgs, _ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        views = [imgs]
        if tta_hflip:
            views.append(torch.flip(imgs, dims=[3]))
        probs = None
        for v in views:
            logits = model(v).float() / temperature
            p = F.softmax(logits, dim=1)
            probs = p if probs is None else probs + p
        out.append((probs / len(views)).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    args = parse_args()
    cfg = U.load_config(args.config)
    ckpt_dir = args.ckpt_dir or cfg["output"]["ckpt_dir"]
    test_dir = cfg["data"]["test_dir"]
    backbones = (args.backbones.split(",") if args.backbones else cfg["ensemble"])
    device = U.get_device(args.device)
    logger = U.setup_logging(cfg["output"]["log_dir"], name="predict")
    U.seed_everything(cfg["seed"])

    inf = cfg["inference"]
    temperature = inf.get("temperature", 1.0)
    if args.template:
        ids = D.load_test_ids(args.template)
        logger.warning(f"using --template ({len(ids)} IDs); Kaggle expects all images in {test_dir}")
    else:
        ids = D.load_all_test_ids(test_dir)
    num_classes = cfg["data"]["num_classes"]
    logger.info(f"ensemble={backbones} ids={len(ids)} device={device} "
                f"tta_hflip={inf['tta']['hflip']} T={temperature} weight_by_oof={inf['weight_by_oof']}")

    ensemble_probs = np.zeros((len(ids), num_classes), dtype=np.float64)
    total_weight = 0.0

    for bk in backbones:
        ckpts = find_ckpts(ckpt_dir, bk, use_ema=not args.raw)
        if not ckpts:
            logger.warning(f"no checkpoints for backbone '{bk}' in {ckpt_dir} — skipping")
            continue
        w = 1.0
        if inf["weight_by_oof"]:
            acc = oof_accuracy(ckpt_dir, bk)
            w = acc if acc is not None else 1.0
        logger.info(f"backbone {bk}: {len(ckpts)} folds, weight={w:.4f}")

        bk_probs = np.zeros((len(ids), num_classes), dtype=np.float64)
        for cpath in ckpts:
            ck = torch.load(cpath, map_location="cpu")
            net = M.create_model(ck["timm_name"], num_classes=ck["num_classes"],
                                 pretrained=False, img_size=ck.get("img_size")).to(device)
            net.load_state_dict(ck["state_dict"])
            tf_eval = D.build_transforms(net, ck.get("img_size"), cfg["train"]["aug"], train=False)
            ds = D.TestDataset(cfg["data"]["test_dir"], ids, tf_eval)
            dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["train"]["num_workers"],
                            pin_memory=device.type == "cuda")
            bk_probs += member_softmax(net, dl, device, inf["tta"]["hflip"], temperature)
        bk_probs /= len(ckpts)            # mean over folds
        ensemble_probs += w * bk_probs
        total_weight += w

    assert total_weight > 0, "no checkpoints found for any backbone — train first"
    ensemble_probs /= total_weight
    preds = ensemble_probs.argmax(1).astype(int)

    sub = pd.DataFrame({"ID": ids, "Label": preds})
    if args.template:
        U.validate_submission(sub, template_path=args.template, num_classes=num_classes)
    else:
        U.validate_submission(sub, test_dir=test_dir, num_classes=num_classes)
    sub.to_csv(args.out, index=False)
    logger.info(f"wrote {args.out} ({len(sub)} rows) — validated OK")


if __name__ == "__main__":
    main()
