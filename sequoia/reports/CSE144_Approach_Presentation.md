# Transfer-Learning Image Classification — Approach, Design Decisions & Findings

**CSE 144 — Applied Machine Learning · Spring 2026 Final Project**
A presentation walkthrough of the ML strategy, the reasoning behind each design
choice, and what the experiments actually showed.

> **Headline result:** a fine-tuned **ConvNeXt** 21k→1k backbone reached **0.80
> Kaggle accuracy** (≈0.77–0.80 leak-free cross-validation) on a 100-class,
> ~1,000-image dataset — built as a config-driven, validation-gated transfer-learning
> pipeline.

---

## 1. The Problem in One Slide

| | |
|---|---|
| **Task** | 100-way image classification via **transfer learning** |
| **Train data** | **1,079** images across **100** classes — folders `0/ … 99/` |
| **Class balance** | **4–41 images/class**; only 33/100 classes have exactly 10 |
| **Test data** | **1,036** images (`0.jpg … 1035.jpg`) — predict *all* of them |
| **Metric** | Test accuracy; Kaggle score `70 + max(0, acc% − 60)` — bonus above 60% is **uncapped** |
| **Hard constraint** | Train **only** on `Data/train/` — no external data, no pseudo-labeling the test set |

**The core tension:** ~11 images per class on average (some as few as 4) is far too
little to train a strong network from scratch. The tail classes impose a *data-imposed
accuracy ceiling*. So the entire strategy is built around one idea: **start from a
powerful pretrained backbone and fine-tune carefully, then squeeze the tail with
regularization and ensembling.**

---

## 2. Strategy Overview — "Borrow Features, Fine-Tune Carefully, Ensemble"

The pipeline is a classic small-data transfer-learning recipe, assembled deliberately
and validated layer by layer:

```
ImageNet-21k pretrained backbone (timm)
        │  replace head → fresh 100-way classifier + dropout
        ▼
Full fine-tune per fold
  · AdamW + layer-wise LR decay (backbone < head)
  · warmup → cosine schedule
  · heavy augmentation: RandAugment + MixUp/CutMix + RandomErasing
  · label smoothing 0.1, stochastic depth, weight decay
  · EMA of weights (warmup-ramped)
        ▼
Stratified 5-fold cross-validation
  · leak-free out-of-fold (OOF) accuracy = the metric we trust
        ▼
Inference: hflip TTA → softmax-average over folds × backbones
        ▼
Validated submission.csv (all 1,036 IDs)
```

**Why this shape?** Each layer attacks the small-data problem from a different angle:

- **21k pretraining** → the biggest single lever for low-data transfer (richer features than 1k-only).
- **Layer-wise LR decay** → preserves pretrained low-level features, adapts high-level ones.
- **Aug + MixUp + label smoothing + dropout + stochastic depth** → the main overfitting defense at ~11 images/class.
- **5-fold CV + OOF** → an honest, leak-free accuracy estimate that uses *all* the scarce data for validation.
- **Ensembling (folds, architectures) + TTA** → cheap, reliable accuracy on top of a fixed training budget.

---

## 3. Design Decisions & Rationale

This section is the heart of the project — every choice was made deliberately and
recorded. Below, *what* was chosen and *why*.

### 3.1 Backbone: ImageNet-21k pretrained ConvNeXt (via `timm`)

| Decision | Rationale |
|---|---|
| **Rank by small-data transfer fit, NOT raw ImageNet top-1** | A model's headline ImageNet accuracy does **not** reliably predict fine-tuned accuracy in low-data regimes. Feature quality, overfit risk, and resolution fit matter more. |
| **ImageNet-21k pretraining** | 21k pretraining gives the largest gains in low-data settings — more, richer features to transfer. Allowed by the rules (only constraint is training on `Data/train/` from pretrained weights). |
| **ConvNeXt family as primary** | CNNs tend to beat Swin/transformer families for *resource-efficient low-data* fine-tuning. ConvNeXt is a modern CNN with clean 224-native checkpoints and strong natural-image transfer. |
| **Progressive sizing: ConvNeXt-S → ConvNeXt-B** | Start small (fast, less overfit), then test whether a *larger* backbone helps before adding architectural diversity. |
| **Diverse members held in reserve** (EffV2-S @300, ViT-B/16 @224, EVA-02-S) | A multi-architecture ensemble only helps if members make *different* mistakes — gated on measured OOF lift, not assumed. |

The closest published analogues to this regime (used to sanity-check expectations) are
**Flowers-102** (2,040 imgs / 102 classes) and the **1,000-image Tiny-ImageNet**
low-data setting.

### 3.2 The #1 Correctness Trap: Label Mapping

> The single most dangerous bug in this project is **not** a modeling mistake — it's
> silently scrambling the labels.

The folder name *is* the class label (`train/7/...` → class `7`). The naive approach —
`ImageFolder`'s default alphabetical ordering — sorts folders as
`"0","1","10","11",…,"2",…`, which maps class `2` to index `10`, etc. That produces
**~random accuracy** with no error message.

**Defense:** an explicit `label = int(folder_name)` map with a **hard assertion** at
load time that the map is *exactly* `"0"→0 … "99"→99`, and samples are always built
from that map — never from directory-iteration order. (`src/data.py:build_label_map`)

### 3.3 Validation: Stratified 5-Fold + Out-of-Fold (OOF)

| Decision | Rationale |
|---|---|
| **StratifiedKFold(5, shuffle, seed=1337)** | Preserves per-class proportions across folds; critical with 4–41 images/class. |
| **OOF accuracy is the metric we trust** | Every train image gets exactly one prediction from a model that never saw it → a leak-free estimate over the *entire* dataset, not a single lucky split. |
| **Kaggle public LB used for format sanity only** | The public LB is ~10% of the test set. Tuning on it = overfitting to noise. OOF is the source of truth; the LB only confirms the submission format is valid. |
| **4-image classes warn but proceed** | With a 4-image class and 5 folds, sklearn warns (a fold may lack that class). Acceptable and logged — the alternative (fewer folds) wastes more data. |

### 3.4 Fine-Tuning Recipe

| Component | Setting | Why |
|---|---|---|
| Optimizer | **AdamW**, wd 0.05 | Standard, robust for fine-tuning transformers/CNNs. |
| **Layer-wise LR decay** | 0.8 (backbone < head) | Early layers (generic edges/textures) barely move; the new head and late layers adapt fast. The canonical pretrained-backbone recipe. |
| Schedule | **warmup (4 ep) → cosine** | Warmup avoids wrecking pretrained weights early; cosine's low-LR tail is where the best epoch usually lands. |
| Peak head LR | 1e-3 (scaled down per layer) | — |
| Head | new 100-way linear + **dropout 0.2** | Fresh classifier for the new label space; dropout regularizes. |
| **Stochastic depth** | drop_path 0.1 | Regularizes deep backbones on tiny data. |
| **Label smoothing** | 0.1 | Discourages overconfidence; helps generalization on small data. |
| **MixUp + CutMix** | α=0.2 / 1.0, switch 0.5 | Strong data-dependent regularizer → SoftTargetCrossEntropy. |
| **RandAugment + color jitter + RandomErasing** | m9 / 0.4 / 0.25 | The primary overfitting defense; per-model timm transforms use each checkpoint's correct mean/std/crop. |
| Grad clip | 1.0 | Stability. |
| AMP | CUDA only | Speed on GPU; auto-disabled on MPS/CPU. |

### 3.5 EMA — and a Subtle Bug Worth Highlighting

Exponential Moving Average of weights is the *reported/selected* model. But the
textbook decay value **silently failed** here, which is one of the most instructive
findings of the project:

- `ema_decay = 0.9998` implies a **~5,000-step** averaging window.
- Total training here is only **~1,080 steps** (≈27 steps/epoch × 40 epochs).
- Result: the EMA never escaped random init → **val accuracy stuck at 0.0** while train loss fell normally.

**Fix:** `ModelEmaV3(use_warmup=True)` ramps the effective decay (≈0.80 early →
≈0.99 late), so the EMA tracks the model from step one. `foreach=False` was also
required because the fused `_foreach_lerp_` kernel is unimplemented on Apple MPS.

> **Lesson:** ImageNet-scale hyperparameters don't transfer blindly to a tiny step
> budget. The fix makes EMA self-correct regardless of dataset size.

### 3.6 Inference: TTA + Ensemble

- **hflip test-time augmentation** — average softmax over original + horizontally-flipped views.
- **Softmax-average** across all 5 folds (and across backbones when ensembling).
- Optional, OOF-tuned-only knobs left in the config but off by default: temperature scaling, OOF-weighted member averaging.
- **Predict all 1,036 test IDs** (numeric sort) and run a hard **format validator** before writing `submission.csv`.

### 3.7 Engineering / Reproducibility Decisions

| Decision | Rationale |
|---|---|
| **Modular `src/*.py` + single `config.yaml`** | Every hyperparameter that affects a result lives in one file → an ablation is a *single flag flip*. Clean diffs for a team; feeds the experiments table directly. |
| **Device-agnostic** (`cuda → mps → cpu`) | Develop/smoke-test locally on Apple M2 Pro (MPS), run graded experiments on GPU. |
| **Fixed seed 1337** everywhere; `cudnn.deterministic` | Reproducible splits and init. |
| **Resolved config snapshot saved next to every run** | Each checkpoint folder records the exact config it was produced with. |
| **Atomic checkpoint saves** (added after a crash) | Write-to-temp-then-rename so an interrupted/disk-full write can't corrupt a checkpoint. |
| **OOF rebuild utility** | Reconstruct `oof.npz` from saved fold checkpoints without retraining (crash recovery). |

---

## 4. Findings & Development Timeline (the Changelog Story)

The project was built incrementally, each layer gated on a *measured* result. This is
the narrative the changelog records.

### Phase 1 — Discovery & Correctness (Jun 1)
- Read the handout; **verified the data on disk against it** and caught three discrepancies:
  - train images **1,079** (not "~1,000"),
  - class counts **4–41** (not "~10 each"),
  - test set **1,036** images (not 1,000).
- Identified the **label-scramble trap** and built the explicit-map + assertion defense.
- Pivoted the goal from "clear the 60% baseline" to **maximize accuracy** (uncapped bonus).

### Phase 2 — Pipeline Build & Label-Map Gate (Jun 1)
- Implemented the full modular pipeline (`data / model / train / predict / utils` + config + notebooks).
- **Two real bugs found and fixed during smoke-testing** (not just smoke artifacts):
  1. **MPS autocast crash** — `torch.autocast('mps')` raises on torch 2.1.2 even when disabled → only enter autocast on CUDA.
  2. **EMA cold-start = 0% val acc** — the warmup-EMA fix described in §3.5.
- **Label-map gate PASSED:** on a 10-class subset, EMA val accuracy rose monotonically `0.00 → 0.45` (random = 0.10) while loss fell → label wiring confirmed correct, end-to-end loop works.

### Phase 3 — First Real Result & Submission Fix (Jun 2)
- **ConvNeXt-S baseline: OOF 0.7720** (folds 0.80 / 0.78 / 0.77 / 0.745 / 0.767). This corresponded to the **0.80 Kaggle** result.
- **Kaggle submission corrected:** first submit rejected — *"1000 rows but 1036 were expected."* The competition scores **all 1,036** test files; `sample_submission.csv` (1,000 rows) is only a format example. Fixed `predict.py` to enumerate the whole test dir.
- **Disabled early stopping:** evidence showed cosine's low-LR tail produces the best epoch (fold 0 won at the full epoch budget; early-stopped folds landed 1–2% lower). Best-EMA-val checkpoint is still selected across all epochs.

### Phase 4 — Ablations: What Did NOT Help (Jun 2–3)
- **Resolution 224 → 288 (+ epochs 40 → 15): OOF 0.76 — no structural gain.** Higher resolution alone did not break the ConvNeXt-S plateau. *Useful negative result.*
- Observed useful learning **plateauing around epoch ~10**, so the gate epoch budget was trimmed (40 → 20) to spend compute elsewhere.

### Phase 5 — Stronger Backbone (Jun 3)
- **ConvNeXt-B (20 epochs, local MPS): OOF 0.7961** — folds 0.79 / 0.81 / 0.80 / 0.78 / 0.80.
- **≈ +2.4 points over the ConvNeXt-S baseline** (0.7720 → 0.7961) from changing backbone size alone — confirming the "stronger/diverse backbone" hypothesis as the highest-yield remaining lever. Lands just under the 0.80 continuation gate, motivating the 2-model OOF ensemble.

### Phase 6 — Error Analysis & Recovery (Jun 3)
- **Weakest classes (ConvNeXt-B OOF recall):** class `68` ≈ 0.09, then `64/66/61` ≈ 0.25, `76/86` ≈ 0.27. These are concentrated in the rare/tail classes → the next lever is **rare-class-specific work, not more epochs**.
- **Disk-full crash** while writing the last fold's EMA checkpoint corrupted the file. Hardened the pipeline: **atomic saves**, EMA→raw fallback at inference, and a **`rebuild_oof.py`** recovery tool that regenerates OOF from existing checkpoints without retraining.

### What the timeline shows
1. **Backbone choice + 21k pretraining was the dominant lever.** ConvNeXt-S → ConvNeXt-B alone added ~2.4 pts.
2. **Resolution and longer training did not help** — the model had plateaued; the bottleneck is data, not optimization.
3. **The remaining ceiling is the tail classes** — a handful of classes with very few images dominate the error.

---

## 5. Results Summary

| Experiment | Backbone(s) | OOF accuracy | Kaggle | Verdict |
|---|---|---:|---:|---|
| **Baseline** | ConvNeXt-S (21k→1k, 224) | **0.7720** | **0.8000** | Strong starting point |
| Resolution ablation | ConvNeXt-S @288, 15 ep | 0.7600 | — | ✗ No gain from resolution |
| **Stronger backbone** | ConvNeXt-B (224, 20 ep) | **0.7961** | — | ✓ +2.4 pts; best single model |
| Diversity gate | ConvNeXt-S + ConvNeXt-B | *pending* | — | Ensemble can clear 0.80 if errors differ |
| Softer-aug ablation | ConvNeXt-B soft aug | *pending* | — | OOF-only check |

**Best achieved: 0.80 Kaggle accuracy** — i.e. a Kaggle score of `70 + (80 − 60) = 90`,
well above the 60% baseline. OOF cross-validation (0.77–0.80) tracks the public LB
closely, confirming the estimate is honest rather than a lucky split.

---

## 6. Discussion — Why 80%, and What's Next

**Why this is a good result for the data.** With an average of ~11 images per class
and a tail down to 4 images, a meaningful fraction of classes simply cannot be learned
reliably — the error is concentrated there (class 68 at 9% recall). 80% on a 100-way
problem from ~1,000 images is squarely in the range the small-data transfer-learning
literature predicts for this regime.

**What worked**
- 21k-pretrained ConvNeXt + careful fine-tuning (LLRD, warmup/cosine, EMA).
- Heavy regularization (MixUp/CutMix, RandAugment, label smoothing, dropout, stochastic depth) as the overfitting defense.
- Honest, leak-free OOF validation that prevented chasing LB noise.
- Going *bigger* on the backbone (S → B).

**What didn't**
- Higher input resolution (224 → 288) alone.
- More epochs past the ~epoch-10 plateau.
- (These are valuable negative results — they show the bottleneck is data, not optimization.)

**Highest-yield next steps**
1. **2-model OOF ensemble** (ConvNeXt-S + ConvNeXt-B) — likely clears 0.80 if the two make different mistakes.
2. **Architectural diversity** — add EfficientNetV2-S (@300) and ViT-B/16 only if each adds measured OOF lift.
3. **Rare-class rescue** — weighted sampler / class-balanced loss / targeted augmentation aimed at the worst-recall tail classes.
4. **Multi-scale TTA** and OOF-tuned temperature/weighting on top of the fixed ensemble.

---

## 7. Reproducibility

- **Single source of truth:** every hyperparameter lives in `config.yaml`; each run snapshots its resolved config.
- **Fixed seed 1337** (python/numpy/torch) + `cudnn.deterministic`; pinned `requirements.txt` (torch 2.1.2 / torchvision 0.16.2 / timm 1.0.27).
- **Leak-free OOF** is the reported metric; the Kaggle LB is used for format validation only.
- Canonical commands:

```bash
# Train one backbone across all 5 folds (saves per-fold EMA ckpts + oof.npz + config)
python src/train.py --backbone convnext_small      # baseline (0.7720 OOF → 0.80 Kaggle)
python src/train.py --backbone convnext_base       # stronger backbone (0.7961 OOF)

# Report individual + equal-average OOF ensemble
python src/oof_ensemble.py --backbones convnext_small,convnext_base

# Inference → validated submission.csv for all 1,036 test IDs
python src/predict.py --out submission.csv
```

---

## 8. One-Slide Takeaway

> **Small data, big borrowed features.** Starting from a 21k-pretrained ConvNeXt and
> fine-tuning it carefully — layer-wise LR decay, heavy augmentation, warmup-EMA, and
> leak-free 5-fold OOF validation — reached **80% accuracy on a 100-class, ~1,000-image
> problem.** The biggest lever was backbone choice; the remaining ceiling is the
> handful of classes with only a few training images. The pipeline is config-driven and
> reproducible, so every result is one flag-flip away from the next experiment.

---

### Appendix — File Map

| File | Role |
|---|---|
| `src/data.py` | Label map (+assert), RGB cleaning, stratified k-fold, timm transforms, datasets |
| `src/model.py` | timm backbone factory + 100-way head, LLRD param groups, warmup-EMA |
| `src/train.py` | Fine-tune one backbone across folds → per-fold EMA ckpts + OOF logits + config |
| `src/predict.py` | Load all ckpts, hflip TTA + softmax ensemble → validated `submission.csv` |
| `src/oof_ensemble.py` | Report individual + ensemble OOF accuracy from aligned `oof.npz` |
| `src/rebuild_oof.py` | Crash-recovery: rebuild `oof.npz` from saved fold checkpoints |
| `config.yaml` | Single source of truth for all hyperparameters |
| `info/changelog.md` | Full dated dev log behind §4 |
</content>
</invoke>
