"""Pseudo-labeling: copy high-confidence test predictions into the training set.

Reads ensemble softmax probabilities saved by predict.py --save-probs, filters
images where max(softmax) >= threshold, and copies them into Data/train/<class>/
so that the next train.py run picks them up automatically.

Usage:
  # Step 1: generate submission + probabilities
  python src/predict.py --out submission_pl.csv --save-probs

  # Step 2: dry run to see how many images would be added
  python src/pseudolabel.py --probs submission_pl.csv.probs.npz --dry-run

  # Step 3: apply
  python src/pseudolabel.py --probs submission_pl.csv.probs.npz

  # Step 4: retrain ViT on expanded dataset
  python src/train.py --backbone vit_base_clip --aug-profile softer
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--probs", required=True, help="path to .probs.npz saved by predict.py --save-probs")
    p.add_argument("--test-dir", default="Data/test", help="test images directory")
    p.add_argument("--train-dir", default="Data/train", help="train directory to copy images into")
    p.add_argument("--threshold", type=float, default=0.9, help="min softmax confidence to include (default 0.9)")
    p.add_argument("--dry-run", action="store_true", help="print stats without copying any files")
    return p.parse_args()


def main():
    args = parse_args()

    z = np.load(args.probs, allow_pickle=True)
    probs = z["probs"]        # [N, 100] float32
    ids = z["ids"].tolist()   # list of filename strings e.g. "0.jpg"

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    mask = confidence >= args.threshold

    total = len(ids)
    selected = int(mask.sum())
    print(f"\nThreshold : {args.threshold}")
    print(f"Test images : {total}")
    print(f"Selected (>= threshold) : {selected} ({100 * selected / total:.1f}%)")
    print(f"Skipped : {total - selected}\n")

    if selected == 0:
        print("No images meet the threshold — nothing to do.")
        return

    # Per-class breakdown
    class_counts = np.bincount(predicted[mask], minlength=100)
    print("Per-class additions (top 10 classes by count):")
    top = np.argsort(class_counts)[::-1][:10]
    for c in top:
        if class_counts[c] > 0:
            print(f"  class {c:3d}: {class_counts[c]} images")

    if args.dry_run:
        print("\n[dry-run] No files copied.")
        return

    # Copy images into train/<class>/pseudo_<id>
    copied = 0
    skipped_exists = 0
    for i, (img_id, cls, conf) in enumerate(zip(ids, predicted, confidence)):
        if conf < args.threshold:
            continue
        cls_dir = os.path.join(args.train_dir, str(cls))
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(args.test_dir, img_id)
        stem = os.path.splitext(img_id)[0]
        dst = os.path.join(cls_dir, f"pseudo_{stem}.jpg")
        if os.path.exists(dst):
            skipped_exists += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"\nCopied  : {copied} images into {args.train_dir}")
    if skipped_exists:
        print(f"Skipped : {skipped_exists} already existed (re-run safe)")
    print("\nNext step:")
    print("  python src/train.py --backbone vit_base_clip --aug-profile softer")


if __name__ == "__main__":
    main()
