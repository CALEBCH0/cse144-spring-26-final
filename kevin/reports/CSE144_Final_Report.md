# CSE 144 Final Project — Kevin Chan's Approach

Transfer Learning Challenge  
Contributor: Kevin Chan

---

## 1. Introduction

This report documents Kevin's contributions to the UCSC CSE 144 Spring 2026
Kaggle transfer-learning challenge. The task is 100-class image classification
on a small dataset (~10 images per class) sampled from diverse real-world
benchmarks (food, flowers, vehicles). The core approach is fine-tuning strong
pretrained backbones and ensembling their predictions.

**Final Kaggle score: 87%**

---

## 2. Dataset

The training set contains 1,079 images across 100 class folders named `0` through
`99`. The folder name is the class label, enforced with an explicit
`label = int(folder_name)` mapping and a hard assert at load time.
The test directory contains 1,036 images. Visual inspection of the training images
confirms the dataset is sampled from multiple known benchmarks, which aligns well
with ImageNet-21k pretrained backbones.

---

## 3. Implementation

### 3.1 Model

All backbones are loaded from `timm` with pretrained ImageNet-21k → ImageNet-1k
weights. A fresh 100-way classification head (with dropout) replaces the original
head. Layer-wise LR decay (LLRD) gives progressively smaller learning rates to
earlier backbone layers, preserving pretrained features while still allowing
full-model fine-tuning.

### 3.2 Training Pipeline

- **Optimizer:** AdamW with cosine LR schedule and linear warmup
- **Regularization:** MixUp + CutMix, label smoothing, dropout, stochastic depth, weight decay, EMA
- **Validation:** Stratified 5-fold cross-validation; OOF accuracy is the primary metric
- **Inference:** Multi-scale TTA (3 scales × hflip = 6 passes), softmax ensemble across all fold checkpoints

### 3.3 Hardware

- GPU: NVIDIA RTX 3080 (10GB VRAM)
- CPU: AMD Ryzen 7 5800X (8 cores / 16 threads)
- OS: Windows 11

---

## 4. Training Log

---

### Run 1 — Baseline (ConvNeXt-Small + ConvNeXt-Base)

**Date:** June 3, 2026

**Commands:**
```bash
python src/train.py --backbone convnext_small
python src/train.py --backbone convnext_base
python src/predict.py --out submission.csv
```

**Hyperparameters:**

| Setting | ConvNeXt-Small | ConvNeXt-Base |
|---|---|---|
| Backbone | `convnext_small.fb_in22k_ft_in1k` | `convnext_base.fb_in22k_ft_in1k` |
| Epochs | 20 | 20 |
| Batch size | 32 | 16 |
| Peak LR | 1e-3 | 1e-3 |
| Head warmup epochs | 0 | 0 |
| Layer decay | 0.8 | 0.8 |
| MixUp / CutMix alpha | 0.2 / 1.0 | 0.2 / 1.0 |
| Aug profile | default | default |
| TTA scales | [1.0] | [1.0] |
| num_workers | 2 | 2 |

**Results:**

| Backbone | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | OOF |
|---|---|---|---|---|---|---|
| ConvNeXt-Small | 0.7870 | 0.7685 | 0.7778 | 0.7546 | 0.7488 | 0.7674 |
| ConvNeXt-Base  | 0.7778 | 0.8241 | 0.7917 | 0.7731 | 0.7907 | 0.7915 |

**Kaggle score (ensemble):** **0.81**

**Reasoning:**
Starting hyperparameters designed by teammate Sequoia. ConvNeXt-Small and
ConvNeXt-Base are both pretrained on ImageNet-21k, well-matched to this dataset's
visual categories. 20 epochs and default augmentation establish a baseline before
tuning.

---

### Run 2 — 40 Epochs + Softer Augmentation + Hardware Tuning

**Date:** June 3–4, 2026

**Commands:**
```bash
python src/train.py --backbone convnext_small --aug-profile softer
python src/train.py --backbone convnext_base --aug-profile softer
python src/predict.py --out submission.csv
```

**Changes from Run 1:**

| Setting | Run 1 | Run 2 |
|---|---|---|
| Epochs | 20 | 40 |
| Head warmup epochs | 0 | 2 |
| MixUp alpha | 0.2 | 0.1 |
| CutMix alpha | 1.0 | 0.5 |
| Color jitter | 0.4 | 0.25 |
| Random erase prob | 0.25 | 0.15 |
| Train crop min scale | 0.4 | 0.55 |
| TTA scales | [1.0] | [1.0, 1.14] |
| num_workers | 2 | 6 |

**Results:**

| Backbone | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | OOF |
|---|---|---|---|---|---|---|
| ConvNeXt-Small | 0.8056 | 0.8009 | 0.7731 | 0.7454 | 0.7767 | 0.7804 |
| ConvNeXt-Base  | 0.8056 | 0.8287 | 0.7824 | 0.8009 | 0.8093 | 0.8054 |

**Reasoning:**
Five improvements based on hardware analysis and dataset inspection: (1) epochs
doubled to 40 for a full cosine LR cycle; (2) softer augmentation reduces noise
on the 10-images-per-class dataset; (3) head warmup of 2 epochs stabilizes the
randomly initialized head before the backbone unfreezes; (4) multi-scale TTA added
to config; (5) num_workers tuned from 2 to 6 for the Ryzen 7 5800X.

---

### Run 3 — ViT-CLIP Third Backbone + Multi-Scale TTA Fix *(Best Result)*

**Date:** June 4, 2026

**Commands:**
```bash
python src/train.py --backbone vit_base_clip --aug-profile softer
python src/predict.py --out submission_vit_ensemble.csv
```

**New backbone:**

| Setting | Value |
|---|---|
| Backbone | `vit_base_patch16_clip_224.laion2b_ft_in12k_in1k` |
| Epochs | 40 |
| Batch size | 16 |
| Peak LR | 3e-4 |
| Layer decay | 0.65 |
| Head warmup epochs | 5 |

**Inference changes:**

| Setting | Before | Run 3 |
|---|---|---|
| TTA scales | [1.0, 1.14] (in config, never used) | [0.875, 1.0, 1.14] (implemented) |
| TTA passes per image | 2 | 6 (3 scales × hflip on/off) |
| Backbone weights | 1.0 each | ConvNeXt-S=1.0, ConvNeXt-B=1.0, ViT=3.0 |
| Ensemble members | 2 | 3 |

**Results:**

| Backbone | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | OOF |
|---|---|---|---|---|---|---|
| ViT-CLIP | 0.8241 | 0.8102 | 0.8009 | 0.8241 | 0.8279 | **0.8174** |

**Kaggle score (3-model ensemble): 0.87**

**Reasoning:**
Three coordinated improvements: (1) ViT-CLIP is architecturally different from
ConvNeXt — self-attention vs convolutions — with CLIP pretraining on LAION-2B
giving it richer semantic features. Uncorrelated errors between ViT and ConvNeXt
make the ensemble stronger than either family alone. (2) The multi-scale TTA
bug was fixed — scales existed in config.yaml but were never read by predict.py,
meaning all prior submissions were single-scale. The fix is a free gain on
existing checkpoints. (3) ViT is weighted 3× in the ensemble reflecting its
stronger pretrained features.

---

### Run 4 — Pseudo-Labeling Experiment

**Date:** June 4–5, 2026

**Process:**
153 test images with softmax confidence ≥ 0.9 added to training set.
ViT retrained on expanded 1,232-sample dataset.

**Results:**

| Backbone | OOF | Kaggle Score |
|---|---|---|
| ViT-CLIP (pseudo-labeled) | 0.8344 | 0.8636 |

**Outcome:** Score dropped from 87% to 86.36% despite higher OOF. Three factors:
noisy pseudo-labels at 0.9 threshold, mismatched ensemble (only ViT retrained),
and circular OOF evaluation inflating the reported improvement.

---

## 5. Results Summary

| Run | Backbones | Epochs | Aug | ConvNeXt-S OOF | ConvNeXt-B OOF | ViT OOF | Kaggle |
|---|---|---|---|---|---|---|---|
| 1 | Small + Base | 20 | default | 0.7674 | 0.7915 | — | 0.81 |
| 2 | Small + Base | 40 | softer | 0.7804 | 0.8054 | — | — |
| 3 | Small + Base + ViT | 40 | softer | — | — | 0.8174 | **0.87** |
| 4 | ViT + pseudo-labels | 40 | softer | — | — | 0.8344 | 0.8636 |

**Best result: 87% Kaggle accuracy (Run 3)**

---

## 6. Discussion

The largest single accuracy gain came from adding the ViT-CLIP backbone in Run 3
(81% → 87%), confirming that architectural diversity in the ensemble is more
valuable than hyperparameter tuning alone. The multi-scale TTA fix contributed an
additional free +1% on top of existing checkpoints.

Pseudo-labeling at 0.9 confidence hurt rather than helped, primarily due to label
noise and an inconsistent ensemble. A higher threshold (≥ 0.95) or retraining all
three backbones jointly on the expanded dataset would likely reduce this regression.

---

## 7. Reproducibility

Seed: 1337 (fixed across all runs via `seed_everything`). All hyperparameters in
`config.yaml`. Each run saves checkpoints, OOF logits, and a resolved config
snapshot under `checkpoints/<backbone>/`.

**Final inference command:**
```bash
python src/predict.py --out submission_vit_ensemble.csv
```

---

## 8. References

- PyTorch and TorchVision pretrained model documentation
- timm pretrained model library and model cards
- UCSC CSE 144 Spring 2026 final project instructions
