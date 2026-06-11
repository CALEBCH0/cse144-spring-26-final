"""Train ONE backbone across stratified k-folds.

Per fold: full fine-tune with layer-wise LR decay, AdamW, warmup+cosine, AMP (CUDA),
MixUp/CutMix + label smoothing, EMA; early-stop on EMA val accuracy. Saves best raw
and best EMA checkpoints per fold, plus leak-free out-of-fold (OOF) logits over the
whole train set and the resolved config.

Usage:
  python src/train.py --backbone convnext_small
  python src/train.py --backbone convnext_small --folds 0 --epochs 6 --limit-classes 20   # quick gate
"""
from __future__ import annotations

import argparse
import contextlib
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

import data as D
import model as M
import utils as U


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--backbone", required=True, help="key into config.backbones")
    p.add_argument("--out", default=None, help="ckpt dir (default config.output.ckpt_dir)")
    p.add_argument("--device", default=None, help="force cuda|mps|cpu")
    p.add_argument("--folds", default=None, help="comma list of fold ids (default all)")
    # quick-run overrides (for smoke / label-map gate)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--limit-classes", type=int, default=None, help="use only classes 0..N-1")
    p.add_argument("--no-mixup", action="store_true", help="disable mixup/cutmix (overfit sanity)")
    p.add_argument("--save-raw", action="store_true", help="also save raw non-EMA checkpoints")
    p.add_argument(
        "--aug-profile",
        choices=["default", "softer"],
        default="default",
        help="default config aug, or softer fine-grained aug ablation",
    )
    return p.parse_args()


def atomic_torch_save(obj, path: str) -> None:
    """Save through a temp file so interrupted writes do not corrupt the target."""
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=out_dir)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def apply_aug_profile(spec: dict, profile: str) -> None:
    """Apply named augmentation ablations in-place."""
    if profile == "default":
        return
    if profile == "softer":
        # Less destructive setting for fine-grained 100-way transfer.
        spec["train"]["aug"].update({
            "color_jitter": 0.25,
            "reprob": 0.15,
            "train_crop_min": 0.55,
        })
        spec["train"]["mixup"].update({
            "mixup_alpha": 0.1,
            "cutmix_alpha": 0.5,
        })
        return
    raise ValueError(f"unknown augmentation profile: {profile}")


def make_loss(mixup_active: bool, label_smoothing: float):
    if mixup_active:
        return SoftTargetCrossEntropy()  # label smoothing folded into Mixup soft targets
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (accuracy, logits[N,C], order_index[N]) on a val loader."""
    model.eval()
    all_logits, all_targets = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs).float().cpu()
        all_logits.append(logits)
        all_targets.append(targets)
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    acc = float((logits.argmax(1) == targets).mean())
    return acc, logits, targets


def train_fold(spec, cfg, paths, labels, tr_idx, va_idx, device, logger, ema_model_for_eval=True):
    """Train a single fold; return (best_ema_acc, best_raw_state, best_ema_state, val_logits)."""
    tcfg = spec["train"]
    bs = tcfg["batch_size"]
    epochs = tcfg["epochs"]
    num_classes = cfg["data"]["num_classes"]

    model = M.create_model(
        spec["timm"], num_classes=num_classes, pretrained=True,
        drop_rate=tcfg["drop_rate"], drop_path_rate=tcfg["drop_path_rate"],
        img_size=spec["img_size"],
    ).to(device)

    tf_train = D.build_transforms(model, spec["img_size"], tcfg["aug"], train=True)
    tf_eval = D.build_transforms(model, spec["img_size"], tcfg["aug"], train=False)
    ds_train = D.TrainDataset(paths, labels, tr_idx, tf_train)
    ds_val = D.TrainDataset(paths, labels, va_idx, tf_eval)

    if tcfg["sampler"]["weighted"]:
        w = D.class_sample_weights(labels, tr_idx)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        shuffle = False
    else:
        sampler, shuffle = None, True

    pin = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=bs, sampler=sampler, shuffle=shuffle,
                          num_workers=tcfg["num_workers"], pin_memory=pin, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False,
                        num_workers=tcfg["num_workers"], pin_memory=pin)

    # mixup
    mix_cfg = tcfg["mixup"]
    mixup_active = mix_cfg["enabled"]
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=mix_cfg["mixup_alpha"], cutmix_alpha=mix_cfg["cutmix_alpha"],
            prob=mix_cfg["prob"], switch_prob=mix_cfg["switch_prob"], mode=mix_cfg["mode"],
            label_smoothing=tcfg["label_smoothing"], num_classes=num_classes,
        )
    criterion = make_loss(mixup_active, tcfg["label_smoothing"])

    # optimizer with layer-wise LR decay; bake lr_scale into per-group lr
    groups = M.build_param_groups(model, tcfg["lr"], tcfg["weight_decay"], tcfg["layer_decay"])
    for g in groups:
        g["lr"] = tcfg["lr"] * g.get("lr_scale", 1.0)
    optimizer = torch.optim.AdamW(groups, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    scheduler = CosineLRScheduler(
        optimizer, t_initial=epochs, lr_min=tcfg["min_lr"],
        warmup_t=tcfg["warmup_epochs"], warmup_lr_init=tcfg["min_lr"], t_in_epochs=True,
    )

    ema = M.create_ema(model, tcfg["ema_decay"]) if tcfg["ema"] else None
    use_amp = U.amp_enabled(tcfg["amp"], device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc, best_raw, best_ema, best_logits = -1.0, None, None, None
    no_improve = 0
    global_step = 0

    for epoch in range(epochs):
        scheduler.step(epoch)
        model.train()
        loss_m = U.AverageMeter()
        for imgs, targets in dl_train:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                imgs, targets_sm = mixup_fn(imgs, targets)
            else:
                targets_sm = targets
            optimizer.zero_grad(set_to_none=True)
            # torch.autocast rejects device_type='mps' (even disabled) on torch 2.1 —
            # only enter the autocast context on CUDA; otherwise run in full precision.
            amp_ctx = torch.autocast(device_type="cuda", enabled=True) if use_amp else contextlib.nullcontext()
            with amp_ctx:
                logits = model(imgs)
                loss = criterion(logits, targets_sm)
            scaler.scale(loss).backward()
            if tcfg["grad_clip"]:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if ema is not None:
                ema.update(model, step=global_step)
            loss_m.update(loss.item(), imgs.size(0))

        eval_model = ema.module if (ema is not None and ema_model_for_eval) else model
        acc, logits, _ = evaluate(eval_model, dl_val, device)
        lr_now = optimizer.param_groups[-1]["lr"]
        logger.info(f"  ep{epoch:02d} loss={loss_m.avg:.3f} val_acc={acc:.4f} lr={lr_now:.2e}")

        if acc > best_acc:
            best_acc = acc
            best_raw = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ema is not None:
                best_ema = {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}
            best_logits = logits
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= tcfg["early_stop_patience"]:
                logger.info(f"  early stop at ep{epoch} (best val_acc={best_acc:.4f})")
                break

    return best_acc, best_raw, best_ema, best_logits


def main():
    args = parse_args()
    cfg = U.load_config(args.config)
    spec = U.resolve_backbone(cfg, args.backbone)
    # apply CLI overrides
    if args.epochs is not None: spec["train"]["epochs"] = args.epochs
    if args.batch_size is not None: spec["train"]["batch_size"] = args.batch_size
    if args.img_size is not None: spec["img_size"] = args.img_size
    if args.no_mixup: spec["train"]["mixup"]["enabled"] = False
    apply_aug_profile(spec, args.aug_profile)

    out_dir = os.path.join(args.out or cfg["output"]["ckpt_dir"], args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    logger = U.setup_logging(cfg["output"]["log_dir"], name=f"train_{args.backbone}")

    device = U.get_device(args.device)
    U.seed_everything(cfg["seed"])
    logger.info(f"backbone={args.backbone} timm={spec['timm']} img={spec['img_size']} "
                f"device={device} amp={U.amp_enabled(spec['train']['amp'], device)} "
                f"aug_profile={args.aug_profile}")

    num_classes = cfg["data"]["num_classes"]
    paths, labels = D.build_samples(cfg["data"]["train_dir"], num_classes)

    # optional class subset for quick label-map gate / smoke
    if args.limit_classes is not None:
        keep = labels < args.limit_classes
        paths = [p for p, k in zip(paths, keep) if k]
        labels = labels[keep]
        logger.info(f"LIMIT to classes 0..{args.limit_classes - 1}: {len(paths)} samples")

    n_splits = cfg["cv"]["n_splits"]
    want_folds = None if args.folds is None else set(int(x) for x in args.folds.split(","))

    oof_logits = np.full((len(paths), num_classes), np.nan, dtype=np.float32)
    fold_accs = {}

    for fold, tr_idx, va_idx in D.stratified_folds(labels, n_splits, cfg["seed"], cfg["cv"]["shuffle"]):
        if want_folds is not None and fold not in want_folds:
            continue
        logger.info(f"=== fold {fold} | train={len(tr_idx)} val={len(va_idx)} ===")
        acc, raw, ema_sd, logits = train_fold(spec, cfg, paths, labels, tr_idx, va_idx, device, logger)
        fold_accs[fold] = acc
        oof_logits[va_idx] = logits

        meta = {"timm_name": spec["timm"], "img_size": spec["img_size"],
                "num_classes": num_classes, "backbone_key": args.backbone,
                "fold": fold, "val_acc": acc}
        if ema_sd is not None:
            atomic_torch_save({**meta, "state_dict": ema_sd}, os.path.join(out_dir, f"fold{fold}_ema.pt"))
        if args.save_raw:
            atomic_torch_save({**meta, "state_dict": raw}, os.path.join(out_dir, f"fold{fold}_raw.pt"))
        logger.info(f"=== fold {fold} best val_acc={acc:.4f} saved ===")

    # OOF accuracy over folds actually run (full OOF when all folds run)
    done = ~np.isnan(oof_logits).any(axis=1)
    if done.any():
        oof_acc = float((oof_logits[done].argmax(1) == labels[done]).mean())
        logger.info(f"OOF accuracy over {done.sum()} samples ({len(fold_accs)} folds): {oof_acc:.4f}")
        np.savez(os.path.join(out_dir, "oof.npz"),
                 logits=oof_logits, labels=labels, covered=done)
    U.save_config(cfg, os.path.join(out_dir, "resolved_config.yaml"))
    U.save_config(spec, os.path.join(out_dir, "resolved_backbone_spec.yaml"))
    logger.info(f"fold accs: {fold_accs}")


if __name__ == "__main__":
    main()
