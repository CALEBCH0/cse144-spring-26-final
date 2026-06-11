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
    p.add_argument("--save-probs", action="store_true",
                   help="save full softmax probabilities to <out>.probs.npz (needed for pseudo-labeling)")
    p.add_argument("--device", default=None)
    p.add_argument("--raw", action="store_true", help="use raw (non-EMA) checkpoints")
    return p.parse_args()


def find_ckpts(ckpt_dir: str, backbone: str, use_ema: bool) -> list[str]:
    tag = "ema" if use_ema else "raw"
    ck = sorted(glob.glob(os.path.join(ckpt_dir, backbone, f"fold*_{tag}.pt")))
    if not ck and use_ema:  # fall back to raw if no EMA saved
        ck = sorted(glob.glob(os.path.join(ckpt_dir, backbone, "fold*_raw.pt")))
    return ck


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


def build_scaled_loader(net, img_size, aug_cfg, test_dir, ids, scale, batch_size, num_workers, pin_memory):
    """Build a test DataLoader with crop_pct adjusted for multi-scale TTA."""
    from timm.data import resolve_data_config
    data_cfg = resolve_data_config({}, model=net)
    base_crop_pct = data_cfg.get("crop_pct", 0.875)
    crop_pct = base_crop_pct / scale
    tf = D.build_transforms(net, img_size, aug_cfg, train=False, crop_pct=crop_pct)
    ds = D.TestDataset(test_dir, ids, tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)


def main():
    args = parse_args()
    cfg = U.load_config(args.config)
    ckpt_dir = args.ckpt_dir or cfg["output"]["ckpt_dir"]
    test_dir = cfg["data"]["test_dir"]
    backbones = (args.backbones.split(",") if args.backbones else cfg["ensemble"])
    device = U.get_device(args.device)
    logger = U.setup_logging(cfg["output"]["log_dir"], name="predict")
    U.seed_everything(cfg["seed"])

    # Verify all requested backbones are registered in timm before doing any work.
    import timm as _timm
    for bk in backbones:
        if bk not in cfg["backbones"]:
            raise KeyError(f"backbone '{bk}' not found in config.backbones")
        timm_name = cfg["backbones"][bk]["timm"]
        try:
            _timm.create_model(timm_name, pretrained=False, num_classes=100)
        except Exception as e:
            raise RuntimeError(f"backbone '{bk}' (timm='{timm_name}') failed to load: {e}") from e

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

    scales = inf["tta"].get("scales", [1.0])
    explicit_weights = inf.get("backbone_weights", {})

    ensemble_probs = np.zeros((len(ids), num_classes), dtype=np.float64)
    total_weight = 0.0

    for bk in backbones:
        ckpts = find_ckpts(ckpt_dir, bk, use_ema=not args.raw)
        if not ckpts:
            logger.warning(f"no checkpoints for backbone '{bk}' in {ckpt_dir} — skipping")
            continue
        if inf["weight_by_oof"]:
            acc = oof_accuracy(ckpt_dir, bk)
            w = acc if acc is not None else 1.0
        else:
            w = explicit_weights.get(bk, 1.0)
        logger.info(f"backbone {bk}: {len(ckpts)} folds, weight={w:.4f}, scales={scales}")

        bk_probs = np.zeros((len(ids), num_classes), dtype=np.float64)
        for cpath in ckpts:
            ck = torch.load(cpath, map_location="cpu")
            net = M.create_model(ck["timm_name"], num_classes=ck["num_classes"],
                                 pretrained=False, img_size=ck.get("img_size")).to(device)
            net.load_state_dict(ck["state_dict"])
            pin = device.type == "cuda"
            scale_probs = np.zeros((len(ids), num_classes), dtype=np.float64)
            for scale in scales:
                dl = build_scaled_loader(
                    net, ck.get("img_size"), cfg["train"]["aug"],
                    cfg["data"]["test_dir"], ids, scale,
                    cfg["train"]["batch_size"], cfg["train"]["num_workers"], pin,
                )
                scale_probs += member_softmax(net, dl, device, inf["tta"]["hflip"], temperature)
            bk_probs += scale_probs / len(scales)
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

    if args.save_probs:
        probs_path = args.out + ".probs.npz"
        np.savez(probs_path, probs=ensemble_probs.astype(np.float32), ids=np.array(ids))
        logger.info(f"saved probabilities to {probs_path}")


if __name__ == "__main__":
    main()
