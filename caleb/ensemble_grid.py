"""
Soft-weighted ensemble grid search.

Usage (after running pipeline.py with each model to save probs/):
    python ensemble_grid.py --a siglip2_so400m --b siglip2_so400m_512
    python ensemble_grid.py --a siglip2_so400m --b dinov3

The script:
  1. Loads val_probs_{a}.npy, val_probs_{b}.npy, val_labels.npy from probs/
  2. Grid-searches w in [0, 1] (step 0.05): ensemble = (1-w)*A + w*B
  3. Prints accuracy and F1 at each w; marks best
  4. Loads test_probs_{a}.npy, test_probs_{b}.npy
  5. Generates submission_ensemble_{a}_{b}_w{best_w}.csv
"""

import argparse
import csv
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

PROBS_DIR = "probs"
TEST_DIR = os.path.join("data", "test")
TRAIN_DIR = os.path.join("data", "train")
STEP = 0.05


def load_val(name):
    path = os.path.join(PROBS_DIR, f"val_probs_{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run pipeline.py with model '{name}' first")
    return np.load(path)  # (N_train, C)


def load_test(name):
    path = os.path.join(PROBS_DIR, f"test_probs_{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run pipeline.py with model '{name}' first")
    return np.load(path)  # (N_test, C)


def grid_search(probs_a, probs_b, labels, step=STEP):
    weights = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    best_w, best_acc, best_f1 = 0.0, 0.0, 0.0
    rows = []
    for w in weights:
        probs = (1 - w) * probs_a + w * probs_b
        preds = probs.argmax(axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        rows.append((w, acc, f1))
        if acc > best_acc:
            best_w, best_acc, best_f1 = w, acc, f1
    return rows, best_w, best_acc, best_f1


def confidence_gated_ensemble(probs_a, probs_b, w, conf_threshold):
    """Use w*B only when B's max softmax >= conf_threshold; else fall back to A."""
    b_confidence = probs_b.max(axis=1)
    gate = (b_confidence >= conf_threshold).astype(np.float32)
    w_eff = gate * w
    return (1 - w_eff[:, None]) * probs_a + w_eff[:, None] * probs_b


def gated_grid_search(probs_a, probs_b, labels):
    """2D grid search over (w, conf_threshold). Returns (results, best_w, best_conf, best_acc)."""
    from sklearn.metrics import accuracy_score
    w_values    = [round(i * 0.05, 2) for i in range(11)]        # 0.00 to 0.50
    conf_values = [round(0.80 + i * 0.05, 2) for i in range(5)]  # 0.80 to 1.00
    best_w, best_conf, best_acc = 0.0, 1.0, 0.0
    results = []
    for w in w_values:
        for conf in conf_values:
            probs = confidence_gated_ensemble(probs_a, probs_b, w, conf)
            preds = probs.argmax(axis=1)
            acc = accuracy_score(labels, preds)
            n_gated = int((probs_b.max(axis=1) >= conf).sum())
            results.append((w, conf, acc, n_gated))
            if acc > best_acc:
                best_w, best_conf, best_acc = w, conf, acc
    return results, best_w, best_conf, best_acc


def load_class_names():
    if not os.path.isdir(TRAIN_DIR):
        return None
    return sorted(d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Model A name (e.g. siglip2_so400m)")
    parser.add_argument("--b", required=True, help="Model B name (e.g. siglip2_so400m_512)")
    parser.add_argument("--step", type=float, default=STEP, help="Weight step size (default 0.05)")
    parser.add_argument("--conf", action="store_true",
                        help="Run 2D confidence-gated grid search instead of 1D")
    parser.add_argument("--run-id", type=str, default="run", dest="run_id",
                        help="Run identifier for submission naming (e.g. run012)")
    args = parser.parse_args()

    step = args.step

    print(f"Loading val probs for {args.a} and {args.b}...")
    probs_a = load_val(args.a)
    probs_b = load_val(args.b)
    labels  = np.load(os.path.join(PROBS_DIR, "val_labels.npy"))

    print(f"  {args.a}: {probs_a.shape}  (val acc solo: {(probs_a.argmax(1) == labels).mean():.4f})")
    print(f"  {args.b}: {probs_b.shape}  (val acc solo: {(probs_b.argmax(1) == labels).mean():.4f})")

    if args.conf:
        print(f"\n2D confidence-gated grid search:")
        print(f"  {'w':>5}  {'conf':>6}  {'acc':>7}  {'gated':>6}")
        print(f"  {'-'*32}")
        gated_results, best_w, best_conf, best_acc = gated_grid_search(probs_a, probs_b, labels)
        n_total = len(labels)
        for w, conf, acc, n_gated in gated_results:
            marker = " <--" if (w == best_w and conf == best_conf) else ""
            print(f"  {w:>5.2f}  {conf:>6.2f}  {acc:>7.4f}  {n_gated:>5}/{n_total}{marker}")
        print(f"\nBest: w={best_w:.2f}  conf={best_conf:.2f}  acc={best_acc:.4f}")
        print(f"  conf-gated ensemble = A + {best_w:.2f}*B when B_conf>={best_conf:.2f}")
        best_f1 = 0.0  # f1 not tracked in 2D search
    else:
        print(f"\nGrid search (1-w)*{args.a} + w*{args.b}:")
        print(f"  {'w':>5}  {'acc':>7}  {'f1':>7}")
        print(f"  {'-'*25}")
        rows, best_w, best_acc, best_f1 = grid_search(probs_a, probs_b, labels, step=step)
        for w, acc, f1 in rows:
            marker = " <--" if w == best_w else ""
            print(f"  {w:>5.2f}  {acc:>7.4f}  {f1:>7.4f}{marker}")
        print(f"\nBest: w={best_w:.2f}  acc={best_acc:.4f}  f1={best_f1:.4f}")
        print(f"  ensemble = {1-best_w:.2f}*{args.a} + {best_w:.2f}*{args.b}")
        best_conf = None

    # ── Generate submission ────────────────────────────────────────────────────
    test_ids = sorted(
        [fn for fn in os.listdir(TEST_DIR) if fn.lower().endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )
    class_names = load_class_names()

    if not test_ids or not class_names:
        print("\nNo test dir or class names found — skipping submission generation")
        return

    print(f"\nLoading test probs and generating submission...")
    test_a = load_test(args.a)
    test_b = load_test(args.b)

    if args.conf and best_conf is not None:
        test_probs = confidence_gated_ensemble(test_a, test_b, best_w, best_conf)
        w_tag   = f"w{int(round(best_w * 100))}"
        conf_tag = f"_conf_t{int(round(best_conf * 100))}"
    else:
        test_probs = (1 - best_w) * test_a + best_w * test_b
        w_tag   = f"w{int(round(best_w * 100))}"
        conf_tag = ""

    test_preds = test_probs.argmax(axis=1)

    os.makedirs("submissions", exist_ok=True)
    out_path = os.path.join("submissions", f"{args.run_id}_{args.a}_{args.b}_soft_{w_tag}{conf_tag}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for img_id, pred in zip(test_ids, test_preds):
            writer.writerow([img_id, int(class_names[int(pred)])])

    print(f"Submission saved: {out_path}")
    print(f"\nTo submit: kaggle competitions submit -c ucsc-cse-144-spring-2026-final-project -f {out_path} -m 'ensemble {args.a}+{args.b}'")


if __name__ == "__main__":
    main()
