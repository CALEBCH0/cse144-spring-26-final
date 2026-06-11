import argparse
import csv
import os
import subprocess
import time

# Load .env for HF_TOKEN (needed for gated models like DINOv3)
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    with open(_env) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
from collections import Counter

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, set_seed

from config import (
    BATCH_SIZE,
    GPU_MEMORY_FRACTION,
    LEARNING_RATE,
    LLRD_FACTOR,
    LORA_R,
    N_FOLDS,
    NUM_EPOCHS,
    PROGRESSIVE_UNFREEZE_EPOCH,
    PSEUDO_LABEL_THRESHOLD,
    SEED,
    SELECTED_MODELS,
    TRAIN_DIR,
    TTA_AUGMENTS,
    UNFREEZE_BLOCKS,
    UPSAMPLE_MIN_COUNT,
    USE_CLASS_ROUTING,
    USE_CLASS_WEIGHTS,
    USE_COLOR_JITTER,
    USE_CUTMIX,
    USE_ENSEMBLE,
    USE_LLRD,
    USE_LORA,
    USE_MIXUP,
    USE_PROGRESSIVE_UNFREEZE,
    USE_PSEUDO_LABEL,
    PSEUDO_LABEL_ENSEMBLE_WEIGHTS,
    USE_RANDAUGMENT,
    USE_RANDOM_ERASING,
    USE_TTA,
    USE_UPSAMPLE_BALANCE,
)
from models import MODELS
from resulter import build_run_config_lines, export_per_class_csv, export_results, print_results
from transforms import build_transforms
from utils import (
    CustomTrainer,
    CutMixCollator,
    EpochAccuracyCallback,
    MixupCollator,
    MixupCutMixCollator,
    ProgressiveUnfreezeCallback,
    apply_freeze,
    apply_lora,
    ensemble_predict,
    get_test_probs,
    predict_test_images,
    predict_with_tta,
    signal,
    top_confusion_pairs,
)


SHORT_NAMES = {
    "siglip2_so400m": "sig2",
    "siglip2_so400m_512": "sig2_512",
    "dinov3": "din3",
    "dinov2_large": "din2l",
    "dinov2": "din2",
}


def _make_sub_path(run_id, model_names, round_n, pseudo_src_tag=None):
    short = "_".join(SHORT_NAMES.get(m, m) for m in model_names)
    parts = [run_id, short, f"r{round_n}"]
    if pseudo_src_tag:
        parts.append(f"pseudo_{pseudo_src_tag}")
    os.makedirs("submissions", exist_ok=True)
    return os.path.join("submissions", "_".join(parts) + ".csv")


def _softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    true = p.label_ids
    return {
        "accuracy": accuracy_score(true, preds),
        "precision": precision_score(true, preds, average="macro", zero_division=0),
        "recall": recall_score(true, preds, average="macro", zero_division=0),
        "f1": f1_score(true, preds, average="macro", zero_division=0),
    }


def _build_class_weights(dataset, num_labels):
    """Compute inverse-frequency class weights normalized to mean=1."""
    counts = Counter(dataset["label"])
    weights = torch.zeros(num_labels)
    for i in range(num_labels):
        weights[i] = 1.0 / max(counts.get(i, 1), 1)
    weights = weights / weights.mean()
    return weights


def _upsample_to_minimum(raw_fold, min_count):
    """Add copies of rare-class images until each class has min_count examples.
    Only upsamples; never removes examples from majority classes.
    Applied to the training fold only — val fold always uses real images.
    The random train transform applied at batch time provides augmentation variety."""
    labels = raw_fold["label"]
    counts = Counter(labels)
    extras = {"image": [], "label": []}
    for cls, count in counts.items():
        needed = min_count - count
        if needed > 0:
            cls_indices = [i for i, l in enumerate(labels) if l == cls]
            for j in range(needed):
                extras["image"].append(raw_fold[cls_indices[j % len(cls_indices)]]["image"])
                extras["label"].append(cls)
    if extras["image"]:
        extra_ds = Dataset.from_dict(extras, features=raw_fold.features)
        upsampled = concatenate_datasets([raw_fold, extra_ds])
        n_added = len(extras["image"])
        n_classes = sum(1 for c in counts if counts[c] < min_count)
        print(f"  Upsampled {n_classes} rare classes (+{n_added} images, {len(raw_fold)}->{len(upsampled)} total)")
        return upsampled
    return raw_fold


def run_training_loop(
    train_dir, full_dataset, class_names, num_labels, label2id, id2label,
    all_label_ids, fold_splits, selected, image_paths, test_ids,
    suffix="", store_fold_models=True, run_id="run", pseudo_src_tag=None,
):
    """Train all selected models with K-fold CV. Returns (results, fold_models_by_name, val_probs_by_fold, val_tf_by_name)."""
    results = {}
    all_fold_models_by_name = {}
    all_val_probs_by_fold = {}
    all_val_tf_by_name = {}
    _fmt = lambda s: f"{s/60:.1f}m" if s >= 60 else f"{s:.0f}s"

    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = _build_class_weights(full_dataset, num_labels)
        print(f"Class weights enabled — min={class_weights.min():.3f}  max={class_weights.max():.3f}")

    for model_cfg in selected:
        name = model_cfg["name"]
        print(f"\n{'='*60}")
        print(f"Training: {name} ({model_cfg['model_id']})  [{N_FOLDS}-fold CV]{' [pseudo]' if suffix else ''}")
        print(f"{'='*60}")
        signal("model_start", model=name, folds=N_FOLDS)

        image_processor = model_cfg["get_processor"]()
        train_tf, val_tf = build_transforms(
            image_processor,
            use_color_jitter=USE_COLOR_JITTER,
            use_randaugment=USE_RANDAUGMENT,
            use_random_erasing=USE_RANDOM_ERASING,
        )

        if USE_MIXUP and USE_CUTMIX:
            data_collator = MixupCutMixCollator(num_labels, mixup_alpha=0.2, cutmix_alpha=1.0)
        elif USE_CUTMIX:
            data_collator = CutMixCollator(num_labels, alpha=1.0)
        elif USE_MIXUP:
            data_collator = MixupCollator(num_labels, alpha=0.2)
        else:
            data_collator = collate_fn

        fold_metrics = []
        fold_train_acc = []
        fold_per_class_f1 = []
        fold_per_class_precision = []
        fold_per_class_recall = []
        fold_models = []
        fold_test_probs_list = []  # R2 only: per-fold test probs (avoids accumulating heavy models in RAM)
        fold_train_sec = []
        fold_eval_sec = []
        pooled_true = []
        pooled_pred = []

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            print(f"\n  -- Fold {fold_idx + 1}/{N_FOLDS} --")

            train_fold_raw = full_dataset.select(train_idx)
            if USE_UPSAMPLE_BALANCE:
                train_fold_raw = _upsample_to_minimum(train_fold_raw, UPSAMPLE_MIN_COUNT)
            train_fold = train_fold_raw.with_transform(train_tf)
            val_fold   = full_dataset.select(val_idx).with_transform(val_tf)

            model = model_cfg["get_model"](num_labels, label2id, id2label)

            model_unfreeze = model_cfg.get("unfreeze_blocks", UNFREEZE_BLOCKS)
            model_epochs   = model_cfg.get("num_epochs", NUM_EPOCHS)
            model_llrd     = model_cfg.get("llrd_factor", LLRD_FACTOR)

            if USE_LORA:
                model = apply_lora(model, name, r=LORA_R)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  LoRA r={LORA_R} — {trainable:,} trainable params")
            else:
                trainable = apply_freeze(model, name, model_unfreeze)
                print(f"  Frozen backbone — {trainable:,} trainable params (unfreeze_blocks={model_unfreeze})")

            train_eval_fold = full_dataset.select(train_idx).with_transform(val_tf)

            callbacks = []
            callbacks.append(EpochAccuracyCallback(train_eval_fold, val_fold, collate_fn))
            if USE_PROGRESSIVE_UNFREEZE and not USE_LORA:
                callbacks.append(ProgressiveUnfreezeCallback(name, PROGRESSIVE_UNFREEZE_EPOCH))

            training_args = TrainingArguments(
                output_dir=os.path.join(model_cfg["output_dir"] + suffix, f"fold_{fold_idx}"),
                do_train=True,
                do_eval=True,
                num_train_epochs=model_epochs,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                learning_rate=model_cfg.get("learning_rate", LEARNING_RATE),
                weight_decay=0.01,
                label_smoothing_factor=0.05,
                eval_strategy="no",
                save_strategy="no",
                load_best_model_at_end=False,
                seed=SEED,
                remove_unused_columns=False,
                logging_steps=10,
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=0,
                dataloader_pin_memory=torch.cuda.is_available(),
            )

            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_fold,
                eval_dataset=val_fold,
                compute_metrics=compute_metrics,
                processing_class=image_processor,
                data_collator=data_collator,
                callbacks=callbacks,
                llrd_factor=model_llrd if (USE_LLRD and not USE_LORA) else 1.0,
                model_name=name,
                use_mixup=USE_MIXUP or USE_CUTMIX,
                class_weights=class_weights,
            )

            t0 = time.perf_counter()
            trainer.train()
            train_sec = time.perf_counter() - t0

            trainer.data_collator = collate_fn
            t0 = time.perf_counter()
            preds_out = trainer.predict(val_fold)
            eval_sec = time.perf_counter() - t0

            preds = np.argmax(preds_out.predictions, axis=1)
            true = preds_out.label_ids
            pooled_true.extend(true.tolist())
            pooled_pred.extend(preds.tolist())

            all_val_probs_by_fold.setdefault(name, []).append(
                (val_idx, _softmax_np(preds_out.predictions))
            )

            fold_metrics.append({
                "accuracy":  accuracy_score(true, preds),
                "precision": precision_score(true, preds, average="macro", zero_division=0),
                "recall":    recall_score(true, preds, average="macro", zero_division=0),
                "f1":        f1_score(true, preds, average="macro", zero_division=0),
            })
            fold_per_class_f1.append(f1_score(true, preds, labels=all_label_ids, average=None, zero_division=0))
            fold_per_class_precision.append(precision_score(true, preds, labels=all_label_ids, average=None, zero_division=0))
            fold_per_class_recall.append(recall_score(true, preds, labels=all_label_ids, average=None, zero_division=0))

            train_eval_fold = full_dataset.select(train_idx).with_transform(val_tf)
            train_preds_out = trainer.predict(train_eval_fold)
            train_acc = accuracy_score(train_preds_out.label_ids,
                                       np.argmax(train_preds_out.predictions, axis=1))
            fold_train_acc.append(train_acc)
            fold_train_sec.append(train_sec)
            fold_eval_sec.append(eval_sec)

            print(f"  Fold {fold_idx + 1}: train_acc={train_acc:.4f}  val_acc={fold_metrics[-1]['accuracy']:.4f}"
                  f"  f1={fold_metrics[-1]['f1']:.4f}  train={_fmt(train_sec)}  eval={_fmt(eval_sec)}")
            signal("fold_done", model=name, fold=fold_idx + 1, n_folds=N_FOLDS,
                   train_acc=round(train_acc, 4),
                   val_acc=round(fold_metrics[-1]["accuracy"], 4),
                   f1=round(fold_metrics[-1]["f1"], 4),
                   train_sec=round(train_sec))

            if USE_TTA:
                tta_probs = predict_with_tta(trainer, full_dataset.select(val_idx), train_tf, n_augments=TTA_AUGMENTS)
                tta_preds = np.argmax(tta_probs, axis=1)
                tta_acc = accuracy_score(true, tta_preds)
                tta_f1 = f1_score(true, tta_preds, average="macro", zero_division=0)
                print(f"  Fold {fold_idx + 1} TTA:  acc={tta_acc:.4f}  f1={tta_f1:.4f}  (delta acc {tta_acc - fold_metrics[-1]['accuracy']:+.4f})")

            if image_paths:
                # Run per-fold test inference immediately and free model to avoid CPU RAM OOM.
                # Accumulating fold models (~1.6 GB each) exhausts Windows virtual address space by fold 3.
                _fold_tp = get_test_probs([model], image_paths, val_tf)
                fold_test_probs_list.append(_fold_tp)
                del model
                import gc; gc.collect()
                torch.cuda.empty_cache()
            elif USE_ENSEMBLE or USE_CLASS_ROUTING:
                model.cpu()
                fold_models.append(model)
            else:
                del model
                import gc; gc.collect()
                torch.cuda.empty_cache()

            # Free trainer, fold datasets, and flush before next fold
            del trainer, train_fold, train_fold_raw, val_fold
            torch.cuda.empty_cache()
            import gc; gc.collect()

        mean = {m: float(np.mean([f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")}
        std  = {m: float(np.std( [f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")}

        mean_per_class_f1        = np.mean(fold_per_class_f1, axis=0)
        mean_per_class_precision = np.mean(fold_per_class_precision, axis=0)
        mean_per_class_recall    = np.mean(fold_per_class_recall, axis=0)
        confusion_pairs = top_confusion_pairs(pooled_true, pooled_pred, class_names)

        total_train = sum(fold_train_sec)
        total_eval  = sum(fold_eval_sec)

        results[name] = {
            "folds": fold_metrics,
            "mean": mean,
            "std": std,
            "train_acc_mean": float(np.mean(fold_train_acc)),
            "train_acc_std":  float(np.std(fold_train_acc)),
            "per_class": {
                class_names[i]: {
                    "f1":        float(mean_per_class_f1[i]),
                    "precision": float(mean_per_class_precision[i]),
                    "recall":    float(mean_per_class_recall[i]),
                }
                for i in range(num_labels)
            },
            "total_train_sec": total_train,
            "total_eval_sec":  total_eval,
            "confusion_pairs": confusion_pairs,
        }
        if store_fold_models and USE_CLASS_ROUTING:
            all_fold_models_by_name[name] = fold_models[:]
            all_val_tf_by_name[name] = val_tf

        signal("model_done", model=name, acc=round(mean["accuracy"], 4),
               f1=round(mean["f1"], 4), total_min=round(total_train / 60, 1))
        print(f"\n{name} CV results ({N_FOLDS} folds):")
        print(f"  {'acc':>6}: {mean['accuracy']:.4f} +/-{std['accuracy']:.4f}")
        print(f"  {'prec':>6}: {mean['precision']:.4f} +/-{std['precision']:.4f}")
        print(f"  {'rec':>6}: {mean['recall']:.4f} +/-{std['recall']:.4f}")
        print(f"  {'f1':>6}: {mean['f1']:.4f} +/-{std['f1']:.4f}")
        print(f"  time : train {total_train/60:.1f}m total ({np.mean(fold_train_sec):.0f}s/fold)"
              f"  |  eval {total_eval:.0f}s total ({np.mean(fold_eval_sec):.0f}s/fold)")

        if confusion_pairs:
            print(f"  Top confused pairs (pooled {N_FOLDS}-fold val):")
            for c_a, c_b, n_err, rate in confusion_pairs[:5]:
                print(f"    classes {c_a}<->{c_b}: {n_err:>3} errors, rate={rate:.2f}")

        if USE_ENSEMBLE and len(fold_models) > 1:
            print(f"\n  Running ensemble ({len(fold_models)} fold models) on full dataset...")
            ens_logits = ensemble_predict(fold_models, full_dataset.with_transform(val_tf), collate_fn)
            ens_preds = np.argmax(ens_logits, axis=1)
            ens_labels = np.array(full_dataset["label"])
            ens_acc = accuracy_score(ens_labels, ens_preds)
            ens_f1 = f1_score(ens_labels, ens_preds, average="macro", zero_division=0)
            print(f"  Ensemble (full train set): acc={ens_acc:.4f}  f1={ens_f1:.4f}")

        # Only save val probs for R1 (real data); R2 pseudo-dataset is a different distribution
        if suffix == "":
            os.makedirs("probs", exist_ok=True)
            n_total = len(full_dataset)
            val_prob_matrix = np.zeros((n_total, num_labels), dtype=np.float32)
            for v_idx, v_probs in all_val_probs_by_fold.get(name, []):
                val_prob_matrix[v_idx] = v_probs
            np.save(f"probs/val_probs_{name}.npy", val_prob_matrix)
            if not os.path.exists("probs/val_labels.npy"):
                np.save("probs/val_labels.npy", np.array(full_dataset["label"]))
            print(f"  Val probs saved: probs/val_probs_{name}.npy")

        # ── Kaggle test inference ──────────────────────────────────────────────
        submission_models = fold_models if fold_models else []

        if image_paths:
            round_n = 2 if suffix else 1
            sub_path = _make_sub_path(run_id, [name], round_n, pseudo_src_tag)
            if fold_test_probs_list:
                # R2: fold models were freed per-fold; average the per-fold test probs accumulated above
                test_probs = np.mean(fold_test_probs_list, axis=0).astype(np.float32)
                print(f"\n  Test probs averaged from {len(fold_test_probs_list)} fold(s) (R2 per-fold inference).")
            else:
                print(f"\n  Predicting {len(test_ids)} test images ({len(submission_models)} fold model(s))...")
                test_probs = get_test_probs(submission_models, image_paths, val_tf)
            np.save(f"probs/test_probs_{name}.npy", test_probs)
            print(f"  Test probs saved: probs/test_probs_{name}.npy")
            test_preds = test_probs.argmax(axis=1)

            with open(sub_path, "w", newline="") as sf:
                writer = csv.writer(sf)
                writer.writerow(["ID", "Label"])
                for img_id, pred in zip(test_ids, test_preds):
                    writer.writerow([img_id, int(class_names[int(pred)])])
            print(f"  Submission saved: {sub_path}")

        if not (store_fold_models and USE_CLASS_ROUTING):
            # Free fold models — not needed unless class routing will use them after all models done
            for _m in fold_models:
                del _m
            fold_models.clear()
            import gc as _gc; _gc.collect()
            torch.cuda.empty_cache()

    return results, all_fold_models_by_name, all_val_probs_by_fold, all_val_tf_by_name


def _log_zscore(probs: np.ndarray) -> np.ndarray:
    """Log-prob z-score normalisation across samples per class column.

    Converts softmax probs → log space (≈ logits), then centres and scales
    each class column to mean=0 std=1.  This removes inter-model calibration
    differences so column-wise selection is on a comparable scale.
    """
    log_p = np.log(probs + 1e-9)                          # (N, C)
    mu    = log_p.mean(axis=0, keepdims=True)              # (1, C)
    sigma = log_p.std(axis=0, keepdims=True) + 1e-9        # (1, C)
    return (log_p - mu) / sigma                            # (N, C)


def _run_class_routing(results, all_fold_models_by_name, all_val_probs_by_fold, all_val_tf_by_name,
                       full_dataset, class_names, num_labels, all_label_ids, fold_splits,
                       image_paths, test_ids, args, suffix=""):
    """Hard class routing: for each class use the model with highest per-class val F1.
    Log-prob z-score normalisation removes inter-model calibration differences before
    column-wise selection, so a peaked SigLIP2 column cannot dominate a flatter
    DINOv2-Large column that is relatively more confident on that class.
    """
    model_names_only = [n for n in results if n != "routed_ensemble"]
    # Tiebreak by overall accuracy so SigLIP2 wins when all models are equal.
    model_priority = sorted(model_names_only,
                            key=lambda n: results[n]["mean"]["accuracy"], reverse=True)

    print(f"\n{'='*60}")
    print(f"Hard-routed ensemble ({len(model_names_only)} models){' [pseudo]' if suffix else ''}")
    print(f"{'='*60}")

    # Build class → best-model mapping from per-class val F1.
    best_model_for_class = []
    for i, cls in enumerate(class_names):
        best_name = model_priority[0]
        best_f1   = -1.0
        for name in model_priority:   # iterate highest→lowest so ties go to best model
            f1 = results[name]["per_class"][cls]["f1"]
            if f1 > best_f1:
                best_f1   = f1
                best_name = name
        best_model_for_class.append(best_name)

    from collections import Counter as _Counter
    routing_counts = _Counter(best_model_for_class)
    for name in model_priority:
        print(f"  {name}: owns {routing_counts.get(name, 0)}/100 classes")

    # Val CV evaluation — z-score within each fold independently (different model
    # instance per fold; pooling would mix distributions from different weights).
    ens_fold_metrics, ens_fold_per_class_f1 = [], []
    ens_fold_per_class_precision, ens_fold_per_class_recall = [], []

    for fold_idx, (_, val_idx) in enumerate(fold_splits):
        # Normalise each model's val probs within this fold.
        val_norm = {}
        for name in model_names_only:
            _, probs = all_val_probs_by_fold[name][fold_idx]   # (N_val, C)
            val_norm[name] = _log_zscore(probs)

        # Column-wise selection.
        n_val     = len(val_idx)
        hard_probs = np.zeros((n_val, num_labels), dtype=np.float32)
        for cls_idx in range(num_labels):
            name = best_model_for_class[cls_idx]
            hard_probs[:, cls_idx] = val_norm[name][:, cls_idx]

        preds = hard_probs.argmax(axis=1)
        true  = np.array(full_dataset.select(val_idx)["label"])

        ens_fold_metrics.append({
            "accuracy":  accuracy_score(true, preds),
            "precision": precision_score(true, preds, average="macro", zero_division=0),
            "recall":    recall_score(true, preds, average="macro", zero_division=0),
            "f1":        f1_score(true, preds, average="macro", zero_division=0),
        })
        ens_fold_per_class_f1.append(f1_score(true, preds, labels=all_label_ids, average=None, zero_division=0))
        ens_fold_per_class_precision.append(precision_score(true, preds, labels=all_label_ids, average=None, zero_division=0))
        ens_fold_per_class_recall.append(recall_score(true, preds, labels=all_label_ids, average=None, zero_division=0))

    ens_mean = {m: float(np.mean([f[m] for f in ens_fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")}
    ens_std  = {m: float(np.std( [f[m] for f in ens_fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")}

    results["routed_ensemble"] = {
        "folds":           ens_fold_metrics,
        "mean":            ens_mean,
        "std":             ens_std,
        "train_acc_mean":  float("nan"),
        "train_acc_std":   0.0,
        "per_class": {
            class_names[i]: {
                "f1":        float(np.mean(ens_fold_per_class_f1, axis=0)[i]),
                "precision": float(np.mean(ens_fold_per_class_precision, axis=0)[i]),
                "recall":    float(np.mean(ens_fold_per_class_recall, axis=0)[i]),
            }
            for i in range(num_labels)
        },
        "total_train_sec": 0.0,
        "total_eval_sec":  0.0,
    }
    print(f"  acc={ens_mean['accuracy']:.4f}+/-{ens_std['accuracy']:.4f}"
          f"  f1={ens_mean['f1']:.4f}+/-{ens_std['f1']:.4f}")

    if image_paths:
        # Test inference — z-score across all test samples (ensemble-averaged per model).
        test_norm = {}
        for name in model_names_only:
            raw = get_test_probs(all_fold_models_by_name[name], image_paths, all_val_tf_by_name[name])
            test_norm[name] = _log_zscore(raw)

        final_probs = np.zeros((len(image_paths), num_labels), dtype=np.float32)
        for cls_idx in range(num_labels):
            name = best_model_for_class[cls_idx]
            final_probs[:, cls_idx] = test_norm[name][:, cls_idx]

        routed_test = final_probs.argmax(axis=1)
        sub_suffix  = suffix.lstrip("_") + "_" if suffix else ""
        routed_path = f"submission_{sub_suffix}routed.csv"
        with open(routed_path, "w", newline="") as sf:
            writer = csv.writer(sf)
            writer.writerow(["ID", "Label"])
            for img_id, pred in zip(test_ids, routed_test):
                writer.writerow([img_id, int(class_names[int(pred)])])
        print(f"  Hard-routed submission saved: {routed_path}")

        if args.submit:
            print(f"  Submitting {routed_path} to Kaggle...")
            result = subprocess.run(
                ["kaggle", "competitions", "submit",
                 "-c", "ucsc-cse-144-spring-2026-final-project",
                 "-f", routed_path, "-m", args.message],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"  {result.stdout.strip()}")
            else:
                print(f"  Kaggle submission failed: {result.stderr.strip()}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=str, metavar="FILE", help="Save results to a text file")
    parser.add_argument("--submit", action="store_true", help="Submit generated submission CSV to Kaggle after training")
    parser.add_argument("--message", type=str, default="pipeline run", metavar="MSG", help="Kaggle submission message (used with --submit)")
    parser.add_argument("--run-id", type=str, default="run", dest="run_id", metavar="ID",
                        help="Run identifier for submission naming (e.g. run012)")
    parser.add_argument("--r2-only", action="store_true", dest="r2_only",
                        help="Skip R1 training; use existing probs/test_probs_*.npy for pseudo-labeling")
    args = parser.parse_args()

    set_seed(SEED)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        total_gb = gpu.total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        print(f"GPU: {gpu.name} ({total_gb:.1f} GB, using {GPU_MEMORY_FRACTION*100:.0f}% = {total_gb * GPU_MEMORY_FRACTION:.1f} GB)")
    else:
        print("WARNING: No CUDA GPU detected — training will run on CPU")

    full_dataset = load_dataset("imagefolder", data_files={"train": os.path.join(TRAIN_DIR, "**")})["train"]

    class_names = full_dataset.features["label"].names
    num_labels = len(class_names)
    label2id = {label: str(i) for i, label in enumerate(class_names)}
    id2label = {str(i): label for i, label in enumerate(class_names)}
    all_label_ids = list(range(num_labels))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(range(len(full_dataset)), full_dataset["label"]))

    selected = [cfg for cfg in MODELS if cfg["name"] in SELECTED_MODELS]

    data_root = os.path.dirname(TRAIN_DIR)
    test_dir  = os.path.join(data_root, "test")
    test_ids, image_paths = [], []
    if os.path.isdir(test_dir):
        test_ids = sorted(
            [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")],
            key=lambda x: int(x.split(".")[0]),
        )
        image_paths = [os.path.join(test_dir, img_id) for img_id in test_ids]

    # ── Round 1: train on real data ────────────────────────────────────────────
    if args.r2_only:
        # Skip R1 — load R1 accuracy from saved test_probs to build the results stub
        print("--r2-only: skipping R1 training, using existing probs/test_probs_*.npy")
        results = {}
        for cfg in selected:
            _tp = os.path.join("probs", f"test_probs_{cfg['name']}.npy")
            if not os.path.exists(_tp):
                raise FileNotFoundError(f"--r2-only requires {_tp} from a completed R1 run")
            # Accuracy stub — only used to pick the best pseudo-label source model
            results[cfg["name"]] = {"mean": {"accuracy": 1.0}}
        fold_models_by_name, val_probs_by_fold, val_tf_by_name = {}, {}, {}
    else:
        results, fold_models_by_name, val_probs_by_fold, val_tf_by_name = run_training_loop(
            TRAIN_DIR, full_dataset, class_names, num_labels, label2id, id2label,
            all_label_ids, fold_splits, selected, image_paths, test_ids,
            run_id=args.run_id,
        )

    if USE_CLASS_ROUTING and len(results) > 1:
        results = _run_class_routing(
            results, fold_models_by_name, val_probs_by_fold, val_tf_by_name,
            full_dataset, class_names, num_labels, all_label_ids, fold_splits,
            image_paths, test_ids, args,
        )

    # ── Round 2: pseudo-labeling ───────────────────────────────────────────────
    if USE_PSEUDO_LABEL and image_paths:
        print(f"\n{'='*60}")
        print(f"Pseudo-labeling (threshold={PSEUDO_LABEL_THRESHOLD})")
        print(f"{'='*60}")

        if PSEUDO_LABEL_ENSEMBLE_WEIGHTS:
            # Load pre-saved test probs and blend with given weights.
            # Higher quality than any single model — use when probs/ are already on disk.
            test_probs = None
            for _name, _w in PSEUDO_LABEL_ENSEMBLE_WEIGHTS.items():
                _path = os.path.join("probs", f"test_probs_{_name}.npy")
                if not os.path.exists(_path):
                    raise FileNotFoundError(
                        f"Ensemble pseudo-label source missing: {_path}\n"
                        f"Run pipeline.py with SELECTED_MODELS=['{_name}'] first to save probs."
                    )
                _p = np.load(_path).astype(np.float32)
                test_probs = _p * _w if test_probs is None else test_probs + _p * _w
            print(f"  Pseudo-label source: ensemble {PSEUDO_LABEL_ENSEMBLE_WEIGHTS}")
            # Free round-1 models — not needed for pseudo-label generation
            del fold_models_by_name
        else:
            # Self-distillation: load pre-saved test probs from R1 (saved by run_training_loop).
            # Fold models were freed after R1 when USE_CLASS_ROUTING=False — use the npy instead.
            _pseudo_src = max(
                (n for n in SELECTED_MODELS if n in results),
                key=lambda n: results[n]["mean"]["accuracy"],
            )
            _tp_path = os.path.join("probs", f"test_probs_{_pseudo_src}.npy")
            if os.path.exists(_tp_path):
                test_probs = np.load(_tp_path).astype(np.float32)
                print(f"  Pseudo-label source: {_pseudo_src} (loaded {_tp_path}, acc={results[_pseudo_src]['mean']['accuracy']:.4f})")
            else:
                # Fallback when fold models are still in memory (USE_CLASS_ROUTING=True path)
                test_probs = get_test_probs(
                    fold_models_by_name[_pseudo_src], image_paths, val_tf_by_name[_pseudo_src]
                )
            del fold_models_by_name
        import torch as _torch; _torch.cuda.empty_cache()

        max_probs     = test_probs.max(axis=1)
        pseudo_labels = test_probs.argmax(axis=1)
        confident_mask = max_probs >= PSEUDO_LABEL_THRESHOLD
        n_pseudo = int(confident_mask.sum())
        print(f"  Accepting {n_pseudo}/{len(test_ids)} pseudo-labels (conf>={PSEUDO_LABEL_THRESHOLD})")

        if n_pseudo > 0:
            # Build pseudo dataset from file paths to avoid loading all PIL images into RAM at once
            from datasets import Dataset, concatenate_datasets
            from PIL import Image as PILImage

            pseudo_paths, pseudo_label_list = [], []
            for i, fpath in enumerate(image_paths):
                if confident_mask[i]:
                    pseudo_paths.append(fpath)
                    pseudo_label_list.append(int(pseudo_labels[i]))

            # Use file paths — HuggingFace Image feature decodes lazily at access time
            combined_features = full_dataset.features
            pseudo_only = Dataset.from_dict(
                {"image": pseudo_paths, "label": pseudo_label_list},
                features=combined_features,
            )
            # Concatenate lazily — orig images stay on disk until needed per-batch
            pseudo_dataset = concatenate_datasets([full_dataset, pseudo_only])

            print(f"  Combined dataset: {len(full_dataset)} real + {n_pseudo} pseudo = {len(pseudo_dataset)} total")

            # Reuse same fold splits (val indices are real-data only; pseudo images are beyond len(full_dataset))
            # Build pseudo_src_tag for submission naming
            if PSEUDO_LABEL_ENSEMBLE_WEIGHTS:
                pseudo_src_tag = "ens_" + "".join(
                    f"{SHORT_NAMES.get(m, m)}{int(round(w * 100))}"
                    for m, w in PSEUDO_LABEL_ENSEMBLE_WEIGHTS.items()
                )
            else:
                pseudo_src_tag = "self"

            results_pseudo, fold_models_pseudo, val_probs_pseudo, val_tf_pseudo = run_training_loop(
                TRAIN_DIR, pseudo_dataset, class_names, num_labels, label2id, id2label,
                all_label_ids, fold_splits, selected, image_paths, test_ids,
                suffix="_pseudo", store_fold_models=False,
                run_id=args.run_id, pseudo_src_tag=pseudo_src_tag,
            )

            if USE_CLASS_ROUTING and len(results_pseudo) > 1:
                # Pass empty image_paths: CV metrics only, skip test inference (fold models freed)
                results_pseudo = _run_class_routing(
                    results_pseudo, fold_models_pseudo, val_probs_pseudo, val_tf_pseudo,
                    pseudo_dataset, class_names, num_labels, all_label_ids, fold_splits,
                    [], test_ids, args, suffix="_pseudo",
                )

            print(f"\n  Pseudo-label comparison (val acc):")
            for mname in SELECTED_MODELS:
                r1 = results.get(mname, {}).get("mean", {}).get("accuracy", float("nan"))
                r2 = results_pseudo.get(mname, {}).get("mean", {}).get("accuracy", float("nan"))
                print(f"    {mname}: {r1:.4f} -> {r2:.4f}  ({r2 - r1:+.4f})")
        else:
            print("  No confident pseudo-labels above threshold; skipping round 2.")

    # ── Summary and export ─────────────────────────────────────────────────────
    run_cfg = {
        "selected_models": SELECTED_MODELS,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "unfreeze_blocks": UNFREEZE_BLOCKS,
        "use_lora": USE_LORA,
        "lora_r": LORA_R,
        "use_randaugment": USE_RANDAUGMENT,
        "use_color_jitter": USE_COLOR_JITTER,
        "use_random_erasing": USE_RANDOM_ERASING,
        "use_mixup": USE_MIXUP,
        "use_cutmix": USE_CUTMIX,
        "use_llrd": USE_LLRD,
        "llrd_factor": LLRD_FACTOR,
        "use_progressive_unfreeze": USE_PROGRESSIVE_UNFREEZE,
        "progressive_unfreeze_epoch": PROGRESSIVE_UNFREEZE_EPOCH,
        "use_tta": USE_TTA,
        "tta_augments": TTA_AUGMENTS,
        "use_ensemble": USE_ENSEMBLE,
        "use_pseudo_label": USE_PSEUDO_LABEL,
        "pseudo_label_threshold": PSEUDO_LABEL_THRESHOLD,
        "use_class_weights": USE_CLASS_WEIGHTS,
    }
    run_cfg_lines = build_run_config_lines(run_cfg)
    print_results(results, class_names, num_labels, N_FOLDS, run_cfg_lines)

    best_model = max(results, key=lambda n: results[n]["mean"]["accuracy"]) if results else "none"
    best_acc = results[best_model]["mean"]["accuracy"] if results else 0.0
    signal("pipeline_done", best=best_model, acc=round(best_acc, 4), models=len(results))

    if args.export:
        export_results(args.export, results, class_names, num_labels, N_FOLDS, run_cfg_lines)
        if len(results) > 1:
            base, ext = os.path.splitext(args.export)
            export_per_class_csv(
                base + ".per_class.csv",
                results, class_names, list(results.keys()),
            )


if __name__ == "__main__":
    main()
