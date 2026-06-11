"""
Per-class model routing ensemble.

SigLIP2 is the backbone for all classes. For specific classes where a secondary
model has a large, consistent val-F1 advantage, replace the backbone's probability
column with that model's column before taking argmax.

Supports multiple secondaries — each class is routed to whichever model leads
by the largest margin above the delta threshold.

Usage:
    # Show full routing table across all secondaries (no submission)
    python class_router.py --analyze --backbone siglip2_so400m_r013 --secondaries dinov3 clip_vit_u4_ep30_llrd75_cfg92

    # Single secondary: grid-search blend weight + generate submission
    python class_router.py --backbone siglip2_so400m_r013 --secondaries dinov3 --delta 0.10

    # Multi-secondary: route each class to its best model + generate submission
    python class_router.py --backbone siglip2_so400m_r013 --secondaries dinov3 clip_vit_u4_ep30_llrd75_cfg92 --delta 0.05

    # Explicit class override (single secondary only)
    python class_router.py --backbone siglip2_so400m_r013 --secondaries dinov3 --classes 86 87 84
"""

import argparse
import csv
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

PROBS_DIR  = "probs"
TEST_IDS   = [f"{i}.jpg" for i in range(1036)]
SUBMIT_DIR = "submissions"
STEP       = 0.05


def load_val(name):
    p = os.path.join(PROBS_DIR, f"val_probs_{name}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p)


def load_test(name):
    p = os.path.join(PROBS_DIR, f"test_probs_{name}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p)


def per_class_f1(probs, labels):
    preds = probs.argmax(1)
    return f1_score(labels, preds, labels=list(range(probs.shape[1])),
                    average=None, zero_division=0)


def evaluate(probs, labels):
    preds = probs.argmax(1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, macro_f1


def blend_single(backbone_probs, secondary_probs, classes, w):
    """Blend one secondary's columns into backbone for the given classes."""
    out = backbone_probs.copy()
    for cls in classes:
        out[:, cls] = (1 - w) * backbone_probs[:, cls] + w * secondary_probs[:, cls]
    return out


def blend_multi(backbone_probs, secondary_probs_map, routing):
    """
    Route each class to its assigned model at w=1.0.

    routing : dict  class_idx -> secondary_name
    secondary_probs_map : dict  name -> probs array
    """
    out = backbone_probs.copy()
    for cls, name in routing.items():
        out[:, cls] = secondary_probs_map[name][:, cls]
    return out


def routing_table(backbone_name, secondary_names, labels, delta, verbose=True):
    """
    Compare backbone vs all secondaries per class.

    Returns:
        bb_f1        : np.ndarray (num_classes,)
        sec_f1s      : dict  name -> np.ndarray (num_classes,)
        routing      : dict  class_idx -> secondary_name  (only classes above delta)
        bb_val_probs : np.ndarray
        sec_val_probs: dict  name -> np.ndarray
    """
    bb_val = load_val(backbone_name)
    bb_f1  = per_class_f1(bb_val, labels)
    bb_acc, bb_macro = evaluate(bb_val, labels)

    sec_val_probs = {}
    sec_f1s       = {}
    for name in secondary_names:
        vp = load_val(name)
        sec_val_probs[name] = vp
        sec_f1s[name]       = per_class_f1(vp, labels)

    num_classes = bb_val.shape[1]

    # Per-class: find best secondary and its margin over backbone
    routing    = {}   # class -> best_secondary_name
    all_rows   = []   # for printing

    for cls in range(num_classes):
        best_name, best_f1, best_delta = None, -1.0, -1.0
        for name in secondary_names:
            d = float(sec_f1s[name][cls]) - float(bb_f1[cls])
            if sec_f1s[name][cls] > best_f1:
                best_f1    = float(sec_f1s[name][cls])
                best_name  = name
                best_delta = d
        if best_delta >= delta:
            routing[cls] = best_name
        all_rows.append((cls, float(bb_f1[cls]), best_name, best_f1, best_delta))

    if verbose:
        # Header
        sec_cols = "  ".join(f"{n[:12]:>12}" for n in secondary_names)
        print(f"\n{'cls':>4}  {'backbone':>8}  {sec_cols}  {'best_sec':>12}  {'margin':>7}  {'route?':>8}")
        print("-" * (4 + 2 + 8 + 2 + 14 * len(secondary_names) + 2 + 12 + 2 + 7 + 2 + 8))
        for cls, bb, bname, bf1, bdelta in all_rows:
            sec_vals = "  ".join(f"{sec_f1s[n][cls]:>12.4f}" for n in secondary_names)
            routed = f"→ {routing[cls][:8]}" if cls in routing else ""
            marker = " <--" if cls in routing else ""
            print(f"  {cls:>3}  {bb:>8.4f}  {sec_vals}  {bf1:>12.4f}  {bdelta:>+7.4f}  {routed}{marker}")

        # Summary
        print(f"\n=== Routing summary (delta >= {delta}) ===")
        print(f"  Backbone solo: acc={bb_acc:.4f}  f1={bb_macro:.4f}  ({backbone_name})")
        for name in secondary_names:
            a, f = evaluate(sec_val_probs[name], labels)
            print(f"  {name:<40} acc={a:.4f}  f1={f:.4f}")
        print(f"\n  SigLIP2 keeps : {num_classes - len(routing):>3} classes")
        for name in secondary_names:
            owned = [c for c, n in routing.items() if n == name]
            if owned:
                print(f"  {name:<40}: {len(owned):>2} classes — {owned}")
        if routing:
            print(f"\n  Routing candidates sorted by margin:")
            print(f"  {'cls':>4}  {'backbone':>8}  {'winner':>40}  {'winner_f1':>10}  {'margin':>7}")
            print("  " + "-" * 76)
            for cls, delta_val in sorted(routing.items(), key=lambda x: -( max(sec_f1s[n][x[0]] for n in secondary_names) - bb_f1[x[0]] )):
                winner = routing[cls]
                print(f"  {cls:>4}  {bb_f1[cls]:>8.4f}  {winner:<40}  {sec_f1s[winner][cls]:>10.4f}  {sec_f1s[winner][cls] - bb_f1[cls]:>+7.4f}")
        else:
            print("  No classes meet the routing threshold.")

    return bb_f1, sec_f1s, routing, bb_val, sec_val_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",    default="siglip2_so400m_r013")
    parser.add_argument("--secondaries", nargs="+", default=["dinov3"],
                        help="One or more secondary model names (space-separated)")
    parser.add_argument("--delta",       type=float, default=0.10,
                        help="Minimum per-class F1 advantage to route a class")
    parser.add_argument("--classes",     type=int, nargs="*", default=None,
                        help="Explicitly route these classes to the first secondary (overrides --delta)")
    parser.add_argument("--analyze",     action="store_true",
                        help="Print routing table only — no submission generated")
    parser.add_argument("--run-id",      default="run018", dest="run_id")
    args = parser.parse_args()

    labels = np.load(os.path.join(PROBS_DIR, "val_labels.npy"))

    print(f"Backbone : {args.backbone}")
    print(f"Secondaries: {args.secondaries}")
    print(f"Delta    : {args.delta}")

    bb_f1, sec_f1s, routing, bb_val, sec_val_probs = routing_table(
        args.backbone, args.secondaries, labels, args.delta, verbose=True
    )

    if args.analyze:
        return

    bb_acc, _ = evaluate(bb_val, labels)

    # --- Single secondary path: --classes override or grid-search blend weight ---
    if len(args.secondaries) == 1:
        sec_name = args.secondaries[0]
        sec_val  = sec_val_probs[sec_name]
        sec_f1   = sec_f1s[sec_name]

        if args.classes is not None:
            routed_classes = args.classes
            print(f"\nExplicitly routing classes: {routed_classes}")
        else:
            routed_classes = [c for c in routing if routing[c] == sec_name]

        if not routed_classes:
            print("No classes to route. Exiting.")
            return

        print(f"\nGrid search blend weight — classes {routed_classes}:")
        print(f"  {'w':>6}  {'acc':>8}  {'f1':>8}  {'delta_acc':>10}")
        print("  " + "-" * 40)

        best_w, best_acc = 0.0, bb_acc
        w = 0.0
        while w <= 1.001:
            blended = blend_single(bb_val, sec_val, routed_classes, w)
            acc, f1 = evaluate(blended, labels)
            marker = " <--" if acc > best_acc else ""
            print(f"  {w:>6.2f}  {acc:>8.4f}  {f1:>8.4f}  {acc - bb_acc:>+10.4f}{marker}")
            if acc > best_acc:
                best_acc = acc
                best_w   = w
            w = round(w + STEP, 10)

        print(f"\nBest: w={best_w:.2f}  acc={best_acc:.4f}  (backbone solo={bb_acc:.4f})")

        if best_w == 0.0:
            print("Backbone solo is optimal — no routing benefit on val.")
            return

        print(f"\nLoading test probs and generating submission (w={best_w:.2f})...")
        bb_test  = load_test(args.backbone)
        sec_test = load_test(sec_name)
        blended_test = blend_single(bb_test, sec_test, routed_classes, best_w)
        preds = blended_test.argmax(1)

        cls_tag = "_".join(str(c) for c in routed_classes)
        w_tag   = f"w{int(best_w * 100):02d}"
        out_path = os.path.join(SUBMIT_DIR,
                                f"{args.run_id}_{args.backbone}_routed{cls_tag}_{w_tag}.csv")

    # --- Multi-secondary path: each class routed to its best model at w=1.0 ---
    else:
        if not routing:
            print("No classes meet the routing threshold. Exiting.")
            return

        print(f"\nMulti-model routing — {len(routing)} classes → their best secondary (w=1.0):")
        blended_val = blend_multi(bb_val, sec_val_probs, routing)
        routed_acc, routed_f1 = evaluate(blended_val, labels)
        print(f"  Backbone solo  : acc={bb_acc:.4f}")
        print(f"  Routed result  : acc={routed_acc:.4f}  f1={routed_f1:.4f}  "
              f"({routed_acc - bb_acc:+.4f})")

        print(f"\nLoading test probs and generating submission...")
        bb_test = load_test(args.backbone)
        sec_test_probs = {n: load_test(n) for n in args.secondaries}
        blended_test = blend_multi(bb_test, sec_test_probs, routing)
        preds = blended_test.argmax(1)

        sec_tag  = "_".join(s[:8] for s in args.secondaries)
        cls_tag  = "_".join(str(c) for c in sorted(routing.keys()))
        out_path = os.path.join(SUBMIT_DIR,
                                f"{args.run_id}_{args.backbone}_multi_{sec_tag}_routed{cls_tag}.csv")

    os.makedirs(SUBMIT_DIR, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for img_id, pred in zip(TEST_IDS, preds):
            writer.writerow([img_id, int(pred)])
    print(f"Submission saved: {out_path}")


if __name__ == "__main__":
    main()
