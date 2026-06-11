"""Display and export CV results for pipeline.py."""


_W = 72  # column width for section headers


def _fmt_duration(seconds: float) -> str:
    """Format seconds as e.g. '4m32s' or '1h03m'."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def build_run_config_lines(cfg: dict) -> list[str]:
    """Build the RUN CONFIGURATION header block from a plain param dict.

    Expected keys (all optional with sensible fallbacks):
        selected_models, num_epochs, batch_size, learning_rate, n_folds, seed,
        unfreeze_blocks, use_lora, lora_r,
        use_randaugment, use_color_jitter, use_random_erasing,
        use_mixup, use_cutmix,
        use_llrd, llrd_factor, use_progressive_unfreeze, progressive_unfreeze_epoch,
        use_tta, tta_augments, use_ensemble
    """
    lora = cfg.get("use_lora", False)
    llrd = cfg.get("use_llrd", False)
    prog = cfg.get("use_progressive_unfreeze", False)
    tta  = cfg.get("use_tta", False)

    training_line = (
        f"  Training        : LLRD={llrd}" +
        (f" (factor={cfg.get('llrd_factor', 0.75)})" if llrd else "") +
        f"  ProgUnfreeze={prog}" +
        (f" (ep{cfg.get('progressive_unfreeze_epoch', 3)})" if prog else "") +
        f"  TTA={tta}" +
        (f" (n={cfg.get('tta_augments', 5)})" if tta else "") +
        f"  Ensemble={cfg.get('use_ensemble', False)}"
    )

    return [
        "=" * _W,
        "RUN CONFIGURATION",
        "=" * _W,
        f"  Models          : {', '.join(cfg.get('selected_models', []))}",
        f"  Epochs          : {cfg.get('num_epochs', '?')}  |  "
        f"Batch: {cfg.get('batch_size', '?')}  |  "
        f"Base LR: {cfg.get('learning_rate', '?')}  |  "
        f"Folds: {cfg.get('n_folds', '?')}  |  "
        f"Seed: {cfg.get('seed', '?')}",
        f"  Unfreeze blocks : {cfg.get('unfreeze_blocks', '?')}  |  LoRA: {lora}" +
        (f" (r={cfg.get('lora_r', 8)})" if lora else ""),
        f"  Augmentation    : RandAugment={cfg.get('use_randaugment', False)}"
        f"  ColorJitter={cfg.get('use_color_jitter', False)}"
        f"  RandomErasing={cfg.get('use_random_erasing', False)}",
        f"  Mix / labels    : Mixup={cfg.get('use_mixup', False)}"
        f"  CutMix={cfg.get('use_cutmix', False)}"
        f"  LabelSmoothing=0.05  WeightDecay=0.01",
        training_line,
        "=" * _W,
    ]


def print_results(results: dict, class_names: list, num_labels: int,
                  n_folds: int, run_cfg_lines: list[str]) -> None:
    """Print run config block, model comparison table, and per-class F1 table."""
    model_names = list(results.keys())

    # ── run config ────────────────────────────────────────────────────
    print()
    for line in run_cfg_lines:
        print(line)

    # ── comparison table ───────────────────────────────────────────────
    print(f"\n{'='*_W}")
    print(f"MODEL COMPARISON ({n_folds}-fold CV, mean ± std)")
    print(f"{'='*_W}")
    print(f"{'model':>14} {'train acc':>14} {'val acc':>14} {'precision':>16} {'recall':>16} {'f1':>16} {'time':>10}")
    print("-" * 102)
    for name in model_names:
        m = results[name]["mean"]
        s = results[name].get("std", {k: 0.0 for k in m})
        tr_acc  = results[name].get("train_acc_mean", float("nan"))
        tr_std  = results[name].get("train_acc_std",  0.0)
        dur = _fmt_duration(results[name].get("total_train_sec", 0))
        print(
            f"{name:>14} "
            f"{tr_acc:.4f}±{tr_std:.4f}  "
            f"{m['accuracy']:.4f}±{s['accuracy']:.4f}  "
            f"{m['precision']:.4f}±{s['precision']:.4f}  "
            f"{m['recall']:.4f}±{s['recall']:.4f}  "
            f"{m['f1']:.4f}±{s['f1']:.4f}  "
            f"{dur:>8}"
        )

    best = max(model_names, key=lambda n: results[n]["mean"]["accuracy"])
    print(f"\nBest model: {best} (mean val acc {results[best]['mean']['accuracy'] * 100:.2f}%)")

    # ── per-class F1 table ─────────────────────────────────────────────
    _print_per_class_table(results, class_names, num_labels, model_names, n_folds, print)


def _print_per_class_table(results, class_names, num_labels, model_names, n_folds, emit):
    """Per-class F1 table sorted by inter-model delta (most complementary first).

    emit — callable that accepts a string (print or file.write line).
    """
    col_w = max(10, max(len(n) for n in model_names) + 1)
    emit(f"\n{'='*_W}")
    emit(f"PER-CLASS F1 (mean across {n_folds} folds) — sorted by model spread (delta)")
    emit(f"{'='*_W}")
    for name in model_names:
        got = sum(1 for cls in class_names if results[name]["per_class"][cls]["f1"] > 0)
        emit(f"  {name}: {got}/{num_labels} classes with F1 > 0")
    emit("")

    header = f"  {'class':<8}" + "".join(f"{n:>{col_w}}" for n in model_names) + f"  {'best':<16} {'delta':>6}"
    emit(header)
    emit("  " + "-" * (len(header) - 2))

    rows = []
    for cls in class_names:
        f1s = {name: results[name]["per_class"][cls]["f1"] for name in model_names}
        best_name = max(f1s, key=f1s.get)
        vals = sorted(f1s.values(), reverse=True)
        delta = vals[0] - vals[1] if len(vals) > 1 else 0.0
        rows.append((cls, f1s, best_name, delta))

    rows.sort(key=lambda x: x[3], reverse=True)

    for cls, f1s, best_name, delta in rows:
        row = f"  {cls:<8}"
        for name in model_names:
            marker = "*" if name == best_name else " "
            row += f"{f1s[name]:>{col_w - 1}.4f}{marker}"
        row += f"  {best_name:<16} {delta:>6.4f}"
        emit(row)

    emit("")
    if len(model_names) > 1:
        dominated = sum(1 for _, _, b, _ in rows if b == model_names[0])
        emit(f"  {model_names[0]} leads on {dominated}/{num_labels} classes")
        for name in model_names[1:]:
            cnt = sum(1 for _, _, b, _ in rows if b == name)
            emit(f"  {name} leads on {cnt}/{num_labels} classes")
        high_delta = sum(1 for _, _, _, d in rows if d > 0.1)
        emit(f"  Classes with delta > 0.10 (worth class-routing): {high_delta}/{num_labels}")


def export_results(path: str, results: dict, class_names: list, num_labels: int,
                   n_folds: int, run_cfg_lines: list[str]) -> None:
    """Write run config + comparison table + per-class F1 to a text file."""
    model_names = list(results.keys())

    lines = [line + "\n" for line in run_cfg_lines]
    lines += [
        "\n",
        f"MODEL COMPARISON ({n_folds}-fold CV, mean ± std)\n",
        f"{'model':>14} {'train acc':>14} {'val acc':>14} {'precision':>16} {'recall':>16} {'f1':>16} {'time':>10}\n",
        "-" * 102 + "\n",
    ]
    for name in model_names:
        m = results[name]["mean"]
        s = results[name].get("std", {k: 0.0 for k in m})
        tr_acc = results[name].get("train_acc_mean", float("nan"))
        tr_std = results[name].get("train_acc_std",  0.0)
        dur = _fmt_duration(results[name].get("total_train_sec", 0))
        lines.append(
            f"{name:>14} "
            f"{tr_acc:.4f}±{tr_std:.4f}  "
            f"{m['accuracy']:.4f}±{s['accuracy']:.4f}  "
            f"{m['precision']:.4f}±{s['precision']:.4f}  "
            f"{m['recall']:.4f}±{s['recall']:.4f}  "
            f"{m['f1']:.4f}±{s['f1']:.4f}  "
            f"{dur:>8}\n"
        )

    best = max(model_names, key=lambda n: results[n]["mean"]["accuracy"])
    lines.append(f"\nBest model: {best} (mean val acc {results[best]['mean']['accuracy'] * 100:.2f}%)\n")

    _print_per_class_table(results, class_names, num_labels, model_names, n_folds,
                           lambda s: lines.append(s + "\n"))

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"\nResults saved to {path}")


def export_per_class_csv(path: str, results: dict, class_names: list, model_names: list) -> None:
    """Write per-class F1/precision/recall comparison to CSV, sorted by inter-model delta."""
    import csv

    rows = []
    for cls in class_names:
        f1s = {name: results[name]["per_class"][cls]["f1"] for name in model_names}
        best_name = max(f1s, key=f1s.get)
        vals = sorted(f1s.values(), reverse=True)
        delta = vals[0] - vals[1] if len(vals) > 1 else 0.0
        row = {"class": cls}
        for name in model_names:
            row[f"{name}_f1"]        = f"{results[name]['per_class'][cls]['f1']:.4f}"
            row[f"{name}_precision"] = f"{results[name]['per_class'][cls]['precision']:.4f}"
            row[f"{name}_recall"]    = f"{results[name]['per_class'][cls]['recall']:.4f}"
        row["best_model"] = best_name
        row["delta"] = f"{delta:.4f}"
        rows.append((delta, row))

    rows.sort(key=lambda x: x[0], reverse=True)

    fieldnames = ["class"]
    for name in model_names:
        fieldnames += [f"{name}_f1", f"{name}_precision", f"{name}_recall"]
    fieldnames += ["best_model", "delta"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in rows:
            writer.writerow(row)
    print(f"Per-class CSV saved to {path}")
