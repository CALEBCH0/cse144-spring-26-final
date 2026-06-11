# CSE 144 Spring 2026 — Final Project Report

**Transfer Learning Challenge: 100-Class Image Classification**

**Team Members:** Sequoia Boubion-McKay, Kevin Chan, Caleb Cho

**Course:** CSE 144 — Applied Machine Learning, UC Santa Cruz, Spring 2026

---

## 1. Introduction

### 1.1 Problem Setting

This project addresses a 100-class image classification task on an extremely small labeled dataset: 1,079 training images (~10.8 images per class, ranging from 4 to 41) and 1,036 unlabeled test images. Images span diverse real-world categories including food, flowers, and vehicles — consistent with a multi-benchmark sample. The goal is to maximize accuracy on the Kaggle public leaderboard for the UCSC CSE 144 Spring 2026 Final Project competition.

The dataset's scale — roughly one order of magnitude below what is typically used for fine-tuning modern vision models — makes generalization the central challenge. Every design decision must trade off accuracy against overfitting on approximately 10 examples per class.

### 1.2 Why Transfer Learning

With ~10 images per class, training any modern vision model from scratch is infeasible. The network would memorize the training set within a few epochs and fail to generalize. Transfer learning sidesteps this by initializing from backbones pretrained on hundreds of millions to billions of images, where lower layers have already learned universal low-level features (edges, textures) and higher layers encode rich semantic representations. Fine-tuning only the top transformer blocks while keeping the rest frozen dramatically reduces effective free parameters, allowing the model to adapt its final representations to the 100-class vocabulary without destroying pretrained features.

### 1.3 Summary of Approaches and Results

Each team member independently developed a transfer-learning pipeline, exploring different backbones, training frameworks, and optimization strategies. The three approaches produced complementary insights across the full spectrum from ConvNeXt baselines to large-scale image-text pretrained models.

| Member | Approach | Best Kaggle Score |
|---|---|---|
| Sequoia Boubion-McKay | ConvNeXt-S/B baseline (timm, device-agnostic) | **0.8000** |
| Kevin Chan | ConvNeXt + ViT-CLIP 3-model ensemble | **0.8700** |
| Caleb Cho | 121-config backbone search → SigLIP2-SO400M | **0.95454** |

**Team best result: 0.95454 (5th place on the public leaderboard).** Source: SigLIP2-SO400M-384 with self-distillation pseudo-labeling (Caleb Cho, `caleb/`).

---

## 2. Dataset

### 2.1 Structure and Scale

The dataset is organized as an image-folder hierarchy:

```
Data/
  train/     # 1,079 images across 100 subdirectories named 0–99
  test/      # 1,036 unlabeled images named 0.jpg–1035.jpg
  sample_submission.csv
```

The training set contains 1,079 images across 100 classes. The class distribution is imbalanced: the smallest class has 4 images, the largest has 41, and the mean is 10.79 images per class.

| Images per class | Number of classes |
|---:|---:|
| 4 | 1 |
| 5–7 | 10 |
| 8–9 | 9 |
| 10–11 | 61 |
| 12–16 | 13 |
| 20 | 2 |
| 31–41 | 2 |

### 2.2 Critical Label-Mapping Issue

A key correctness risk in this project: PyTorch `ImageFolder` sorts folder names alphabetically as strings, producing the order `0, 1, 10, 11, ..., 19, 2, 20, ...` instead of the numeric order `0, 1, 2, ..., 99`. This scrambles all labels silently — the model would appear to train normally while making near-random predictions.

All three team members independently discovered and fixed this by building an explicit mapping `label = int(folder_name)` with a hard assertion that the resulting label set is exactly `{0, 1, ..., 99}`. This was the highest-priority correctness fix in the project.

### 2.3 Submission Row Count

The `sample_submission.csv` contains 1,000 rows, but the test directory contains 1,036 images. A 1,000-row submission is rejected by Kaggle. All three members' inference scripts enumerate test files numerically and validate that the output contains all 1,036 rows before writing.

### 2.4 Preprocessing

All approaches use standard ImageNet normalization. Training augmentation varies by approach — ConvNeXt-based pipelines benefit from standard augmentations (MixUp, CutMix, RandAugment), while stronger pretrained models (DINOv2, SigLIP2) require minimal or no augmentation beyond basic flipping and cropping.

---

## 3. Approaches

### 3.1 Sequoia's Approach — ConvNeXt Baseline (timm, Device-Agnostic)

**Framework:** `timm` model factory, 5-fold StratifiedKFold cross-validation, AdamW + cosine LR schedule with linear warmup, layer-wise learning-rate decay (LLRD = 0.8), EMA checkpointing.

**Hardware:** Apple M2 Pro (MPS) for local development; Colab CUDA for the submitted run.

#### 3.1.1 Model

Two ConvNeXt backbones, both with ImageNet-21k → ImageNet-1k pretraining:

| Alias | `timm` ID | Input | Role |
|---|---|---|---|
| ConvNeXt-S | `convnext_small.fb_in22k_ft_in1k` | 224px | Baseline / submitted model |
| ConvNeXt-B | `convnext_base.fb_in22k_ft_in1k` | 224px | Strongest single OOF model |

The original classifier is replaced by a 100-way head with dropout 0.2 and stochastic depth (drop_path=0.1). The full model is fine-tuned using LLRD so earlier pretrained layers receive proportionally smaller gradient updates.

#### 3.1.2 Training

| Setting | Value |
|---|---|
| Seed | 1337 |
| CV | 5-fold StratifiedKFold |
| Optimizer | AdamW |
| Peak LR (head) | 1e-3 |
| LLRD | 0.8 |
| Epochs | 20 |
| Warmup | 4 epochs |
| Label smoothing | 0.1 |
| Augmentation | RandomResizedCrop (scale≥0.4), HFlip, RandAugment, ColorJitter, RandomErasing, MixUp (α=0.2), CutMix (α=1.0) |
| EMA | 0.9998 decay with warmup |

#### 3.1.3 Results

| Experiment | OOF Accuracy | Kaggle |
|---|---|---|
| ConvNeXt-S baseline | 0.7785 | **0.8000** |
| ConvNeXt-B | 0.7961 | — |
| ConvNeXt-S + ConvNeXt-B ensemble (equal weight) | 0.7924 | — |
| Resolution ablation (288px, 15ep) | 0.7600 | — |

**Key finding:** The equal-weight ConvNeXt-S + ConvNeXt-B ensemble scored 0.7924 — below ConvNeXt-B alone (0.7961). The two models were not architecturally diverse enough for a simple average ensemble to help; the weaker model diluted the stronger model's predictions. Increasing resolution to 288px did not improve OOF accuracy.

---

### 3.2 Kevin's Approach — ConvNeXt + ViT-CLIP Ensemble

**Framework:** Same `timm` pipeline (shared with Sequoia). AdamW, 5-fold StratifiedKFold, multi-scale TTA, weighted ensemble.

**Hardware:** NVIDIA RTX 3080 (10 GB VRAM), Windows 11.

#### 3.2.1 Experimental Runs

**Run 1 — Baseline (ConvNeXt-S + ConvNeXt-B, 20 epochs, default aug)**

| Backbone | OOF |
|---|---|
| ConvNeXt-Small | 0.7674 |
| ConvNeXt-B | 0.7915 |
| **Kaggle (ensemble)** | **0.81** |

**Run 2 — Extended Training + Softer Augmentation (40 epochs)**

Doubled epochs, reduced MixUp/CutMix alpha, added 2-epoch head warmup, tuned num_workers to 6 (from 2) for the 8-core Ryzen 7.

| Backbone | OOF |
|---|---|
| ConvNeXt-Small | 0.7804 |
| ConvNeXt-B | 0.8054 |

**Run 3 — ViT-CLIP Third Backbone + TTA Fix (Best Result)**

Added `vit_base_patch16_clip_224.laion2b_ft_in12k_in1k` as a third backbone. Simultaneously fixed a TTA bug — multi-scale TTA scales existed in `config.yaml` but were never read by `predict.py`, meaning all prior submissions were single-scale. The fix is a free gain on existing checkpoints.

| Backbone | OOF |
|---|---|
| ViT-CLIP | **0.8174** |
| **Kaggle (3-model ensemble, ViT weighted 3×)** | **0.87** |

**Run 4 — Pseudo-Labeling Experiment**

153 test images with softmax confidence ≥ 0.9 added to training set; ViT-CLIP retrained on expanded 1,232-sample dataset.

| Backbone | OOF | Kaggle |
|---|---|---|
| ViT-CLIP (pseudo-labeled) | 0.8344 | 0.8636 |

Score dropped from 87% to 86.36% despite higher OOF. Three factors: noisy pseudo-labels at 0.9 threshold, only ViT retrained (mismatched ensemble), and circular OOF evaluation inflating the reported gain.

#### 3.2.2 Key Findings

- **Architectural diversity is the biggest lever.** Adding ViT-CLIP to two ConvNeXt models gained +6% (81% → 87%). ViT-CLIP's self-attention mechanism and CLIP pretraining on LAION-2B produce uncorrelated errors relative to ConvNeXt's convolutional hierarchy.
- **Multi-scale TTA bug fix gave a free +1%** on existing checkpoints.
- **Pseudo-labeling at threshold=0.9 hurt** due to label noise. A higher threshold (≥0.95) or retraining all backbones jointly would be needed.

---

### 3.3 Caleb's Approach — Exhaustive Backbone Search (121 Configs, SigLIP2)

**Framework:** HuggingFace Transformers (`AutoModel`, `Trainer`), 5-fold stratified CV, Windows Python (native CUDA, no WSL filesystem overhead).

**Hardware:** NVIDIA RTX 5070 Ti (16 GB VRAM), PyTorch 2.12.0+cu128, mixed precision (fp16).

#### 3.3.1 Backbone Progression

Eight backbone families were evaluated across 121 hyperparameter configurations. The progression reveals a clear hierarchy dominated by pretraining quality:

| Backbone | Best CV Acc | Config |
|---|---|---|
| ResNet-50 | 6.79% | Baseline (10ep) |
| ResNet-101 | 14.81% | Baseline (10ep) |
| ConvNeXt-Base | 76.27% | unfreeze=4, lr=3e-4 |
| ConvNeXt-Large | 77.85% | unfreeze=4, progressive unfreeze |
| ViT-B/16 | 74.42% | unfreeze=4, LLRD=0.85, +ColorJitter |
| DINOv2-Base | 85.54% | unfreeze=2, lr=1e-4, LLRD=0.85, 10ep |
| DINOv2-Large | 87.39% | unfreeze=2, lr=1e-4, LLRD=0.75, 50ep |
| DINOv2-Large + upsample | 89.71% | pipeline run, upsample min=20 |
| CLIP-ViT-Large/14 | 88.23% | unfreeze=4, LLRD=0.75, 30ep |
| DINOv3 ViT-L/16 | 93.79% | unfreeze=4, lr=1e-4, 50ep |
| SigLIP2-SO400M-384 | **95.27%** | unfreeze=2, lr=1e-4, LLRD=0.8, 20ep |

#### 3.3.2 Key Hyperparameter Findings

**Unfreeze depth is the dominant lever for ViT-family models:**

| DINOv2 unfreeze_blocks | CV Accuracy |
|---|---|
| 0 (head only) | 41.1% |
| 2 (optimal) | **84.2%** |
| 4 | 82.6% |
| 12 (full fine-tune) | 77.4% |

Unfreezing just 2 top encoder layers of DINOv2 achieves 84.2% — a 43-point jump over a frozen linear probe. More unfreezing catastrophically overwrites pretrained features. The same pattern holds for SigLIP2: unfreeze=2 → 95.27%; unfreeze=4 → 94.81%; unfreeze=6 → 94.44%.

**LLRD is critical for ViT families, neutral for ConvNeXt.** Without LLRD, a frozen ViT-B/16 reaches only 45%; with unfreeze=4 and LLRD=0.85, it reaches 72.6% (+27pp). For SigLIP2, LLRD=0.8 is optimal.

**Augmentation hurts strongly pretrained models.** MixUp (−2%), CutMix (−6%), and RandAugment (−0.18% for DINOv2, neutral for SigLIP2) all degrade models with DINO/SigLIP pretraining. These models' features are already highly invariant; augmentation destroys fine-grained discriminative cues. For ConvNeXt, RandAugment alone gives +0.93%; ColorJitter gives +1.85% on ViT-B.

**Higher resolution consistently hurts.** DINOv2-Large at 518px: −9.64% vs 224px. SigLIP2-512px: −0.55% vs 384px. The ~1,079 training images cannot support the 5× larger token sequence; positional embedding interpolation noise dominates.

**Class weights are harmful.** Inverse-frequency weighting hurts all models (SigLIP-Base −0.75%, DINOv2-Large −2.13%). Upsampling rare classes to a minimum of 20 per fold helps DINOv2-Large significantly (+2.32pp).

#### 3.3.3 SigLIP2 Best Configuration

SigLIP2-SO400M-384 (`google/siglip2-so400m-patch16-384`) is the final best backbone: 400M parameters, 27 encoder layers, native 384px pretraining on billions of image-text pairs.

| Setting | Value |
|---|---|
| Unfreeze blocks | 2 of 27 |
| Learning rate | 1e-4 |
| LLRD | 0.8 |
| Epochs | 20 |
| Batch size | 16 |
| Augmentation | Disabled |
| Pseudo-labeling | Self-distillation, threshold=0.97 |

**Round 1 CV accuracy: 95.18%**  
**Round 2 CV accuracy (self-distillation): 95.27%** (+0.09%)  
**Kaggle public score: 95.454% (5th place)**

Self-distillation works because SigLIP2 generates its own pseudo-labels from its highest-confidence test predictions. At threshold=0.97, 801 of 1,036 test images are accepted; these high-quality labels inject accurate signal for round 2. Blended ensemble pseudo-labels (mixing in weaker DINOv2 predictions) produced zero gain — the weaker model's scores diluted SigLIP2's softmax distributions below the confidence threshold.

#### 3.3.4 Ensemble Results

All global ensemble strategies were evaluated and ultimately failed to improve on the test set:

| Strategy | CV Acc | Kaggle | Gap |
|---|---|---|---|
| SigLIP2 R2 solo (run_004c) | 95.27% | **95.454%** | −0.18% |
| SigLIP2-384 + SigLIP2-512 (w=0.20) | 95.00% | 94.545% | −0.455% |
| SigLIP2 + DINOv3 (w=0.65) | 95.00% | 91.818% | −3.18% |
| Different-seed ensemble (SEED=42 + SEED=0, w=0.30) | 95.18% | 95.454% | −0.27% |

The 1,079-sample validation set cannot reliably identify which test-set errors are independent across models. Any CV ensemble gain of ≤0.28pp reflects val-set overfitting rather than genuine complementarity. All ensemble paths are exhausted.

**Per-class hard routing** (`class_router.py`) is the most promising remaining direction: SigLIP2 handles 87–88 classes while secondary models replace probability columns for specific hard classes where they show large, consistent per-class F1 advantages (e.g., DINOv3 on class 87: +16.2pp; CLIP on class 52: +16.0pp; CLIP on class 86: +15.4pp). This mechanism is different from global weighting and less susceptible to val-overfitting. Post-deadline analysis showed a +1.21pp CV gain routing 12 classes.

---

## 4. Consolidated Results

### 4.1 Key Results Across All Approaches

| Member | Backbone(s) | OOF / CV Acc | Kaggle | Notes |
|---|---|---|---|---|
| Sequoia | ConvNeXt-S | 0.7785 | **0.8000** | First submitted result |
| Sequoia | ConvNeXt-B | 0.7961 | — | Best single ConvNeXt OOF |
| Kevin | ConvNeXt-S + B (20ep) | 0.7674 / 0.7915 | 0.81 | Baseline ensemble |
| Kevin | ConvNeXt-S + B (40ep, softer aug) | 0.7804 / 0.8054 | — | Run 2 |
| Kevin | ConvNeXt-S + B + ViT-CLIP | — / 0.8174 | **0.8700** | Run 3; TTA fix |
| Caleb | DINOv2-Base | 0.8554 | — | unfreeze=2, 10ep |
| Caleb | DINOv2-Large | 0.8739 | — | unfreeze=2, 50ep |
| Caleb | DINOv2-Large + upsample | 0.8971 | — | min 20/class/fold |
| Caleb | CLIP-ViT-Large/14 | 0.8823 | — | unfreeze=4, 30ep |
| Caleb | DINOv3 ViT-L/16 | 0.9379 | — | unfreeze=4, 50ep |
| Caleb | SigLIP2-SO400M-384 R1 | 0.9518 | — | pipeline run_004c |
| Caleb | SigLIP2-SO400M-384 R2 | **0.9527** | **0.95454** | **5th place — final result** |

### 4.2 Hard Class Analysis

The following confusion pairs are consistent across all models and all team members' approaches:

| Pair | Primary confusion rate | Present in |
|---|---|---|
| Class 63 ↔ 70 | 0.42 (SigLIP2 OOF) | All models |
| Class 86 ↔ 88 | 0.32 | All models |
| Class 76 ↔ 78 | 0.30 | All models |
| Class 86 ↔ 87 | 0.26 | All models |
| Class 68 ↔ 74 | 0.16 | Most models |

Class 66 scores F1=0.00 across every model tested (ResNet-50, ConvNeXt, DINOv2, DINOv3, CLIP, SigLIP2) — making it the hardest single class and unroutable via any secondary model. Class 86 appears in two confusion pairs, acting as a hub of visual ambiguity.

The persistence of these hard classes across all eight architectures suggests image-level visual ambiguity (near-identical appearance between categories) rather than any model-specific failure mode.

---

## 5. Discussion

### 5.1 What Worked Best and Why

**Pretraining quality is the dominant factor.** The accuracy hierarchy is directly explained by pretraining breadth and depth:

- **SigLIP2** (image-text contrastive on billions of pairs, 384px native, 400M params) → 95.27% CV
- **DINOv3** (self-supervised distillation from ViT-7B teacher, 1.689B images) → 93.79% CV
- **CLIP-ViT-L** (image-text contrastive on 400M pairs) → 88.23% CV
- **DINOv2-Large** (self-supervised DINO on 142M images) → 87.39% CV
- **ConvNeXt-B/L** (supervised ImageNet-21K) → 77.9% CV

No amount of hyperparameter tuning on a weaker backbone closed the gap to a stronger one. The jump from ConvNeXt-Large (77.9%) to DINOv2-Base (85.5%) by unfreezing only 2 layers demonstrates that the value is in the pretrained representations, not the fine-tuning procedure.

**Shallow fine-tuning is critical.** On ~1,000 training images, only the top 1–2 transformer blocks need to adapt. Unfreezing more layers overwrites pretrained features faster than the small training set can replace them. Full fine-tuning of DINOv2-Large collapses from 87.4% to 77.4%.

**Self-distillation pseudo-labeling works when the model is strong.** SigLIP2 gains +0.09% CV and +0.64% Kaggle test from its own high-confidence predictions. The key is that only the strongest model generates pseudo-labels — mixing in weaker model predictions produces zero gain.

### 5.2 What Didn't Work

**Ensembles overfit the 1,079-sample validation set.** Every ensemble strategy showed a positive CV gain (0.10–0.28pp) that failed to generalize to the 1,036-image test set. The validation set has only ~10 samples per class — not enough to distinguish whether two models' errors are genuinely independent on the test set.

**Higher resolution hurts.** More pixels means more patches, a larger token sequence, and positional embedding interpolation noise from the pretrained resolution. The small training set cannot compensate. Both DINOv2-Large at 518px (−9.64%) and SigLIP2 at 512px (−0.55%) confirm this.

**Aggressive augmentation hurts strong pretrained models.** MixUp (−2%), CutMix (−6%), and full augmentation stacks degrade DINOv2 and SigLIP2. These models' DINO/SigLIP features are already highly invariant to appearance variation; augmentation destroys the fine-grained discriminative cues they rely on.

**Class-weighted loss hurts all models.** Inverse-frequency weighting distorts gradient balance more than it helps rare classes at ~10 images/class.

### 5.3 Limitations

- Class 66 is fundamentally unresolvable with available training data — all 8 architectures achieve F1=0 on this class.
- The 1,079-sample validation set is too small to select ensemble weights that generalize to the test set.
- Pseudo-label gain (+0.09% CV) is within run-to-run noise; it is not reliable across seeds.

---

## 6. Reproducibility

### 6.1 Sequoia's Pipeline (`sequoia/`)

**Environment:** Python 3.11.5, PyTorch 2.1.2, torchvision 0.16.2, timm 1.0.27  
**Seed:** 1337

```bash
pip install -r requirements.txt

# Train ConvNeXt-B across all five folds
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train.py \
  --backbone convnext_base --device mps --batch-size 8 --out checkpoints_local

# Report OOF ensemble accuracy
python src/oof_ensemble.py --ckpt-dir checkpoints_local \
  --backbones convnext_small,convnext_base

# Generate 1,036-row submission
python src/predict.py --ckpt-dir checkpoints_local \
  --backbones convnext_base --out outputs/submissions/submission.csv
```

### 6.2 Kevin's Pipeline (`kevin/`)

**Environment:** Same as Sequoia (shared timm pipeline). RTX 3080 for CUDA runs.  
**Seed:** 1337

```bash
pip install -r requirements.txt

# Train each backbone
python src/train.py --backbone convnext_small
python src/train.py --backbone convnext_base
python src/train.py --backbone vit_base_clip --aug-profile softer

# Generate 3-model ensemble submission
python src/predict.py --out submission_vit_ensemble.csv
```

### 6.3 Caleb's Pipeline (`caleb/`)

**Environment:** PyTorch 2.12.0+cu128, Python 3.12, RTX 5070 Ti, Windows Python (native CUDA)  
**Seed:** 42

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Full pipeline (train SigLIP2 across 5 folds + pseudo-label R2 + generate submission)
python pipeline.py

# Hyperparameter search (121 configs)
python search.py

# Kaggle submission
python pipeline.py --submit --message "SigLIP2 R2 self-distillation"
```

---

## 7. Team Contributions

**Caleb Cho** built the hyperparameter search infrastructure (`search.py`, 121 configs), evaluated 8 backbone families across 121 configurations, discovered and optimized SigLIP2-SO400M-384 as the dominant backbone, implemented self-distillation pseudo-labeling, developed the per-class routing ensemble (`class_router.py`), and produced the final Kaggle submission at 95.454% (5th place). He also implemented best-epoch in-memory recovery, the log-staleness watchdog for crash detection, and automatic run monitoring with push notifications.

**Kevin Chan** extended the shared `timm` pipeline with ViT-CLIP as a third backbone, identified and fixed the multi-scale TTA bug that was silently single-scaling all prior submissions, ran 4 structured experimental runs from baseline to ensemble, and investigated pseudo-labeling. His best result (87% Kaggle) validated that architectural diversity — not just more hyperparameter tuning — is the key lever for ensemble gain.

**Sequoia Boubion-McKay** designed the initial production pipeline architecture: the `timm`-based 5-fold cross-validation framework, EMA checkpointing, atomic saves, ConvNeXt baselines, and the submission validator that identified the 1,036-row vs 1,000-row discrepancy. Her pipeline was shared with and extended by Kevin. She completed all documentation, the submission checklist, and the project report.

As a group, the team compared results across approaches and shipped Caleb's SigLIP2 solution as the final submission.

---

## 8. References

- UCSC CSE 144 Spring 2026 Final Project handout.
- PyTorch documentation, `torch.utils.data`. https://docs.pytorch.org/docs/2.1/data.html
- PyTorch documentation, "Reproducibility". https://docs.pytorch.org/docs/2.1/notes/randomness.html
- TorchVision documentation, "Models and pre-trained weights". https://docs.pytorch.org/vision/stable/models.html
- `timm` documentation, "Models" reference. https://huggingface.co/docs/timm/reference/models
- `timm` documentation, "Data" reference (`create_transform`, `resolve_data_config`). https://huggingface.co/docs/timm/reference/data
- HuggingFace Transformers, `AutoModel`, `Trainer` documentation. https://huggingface.co/docs/transformers
- Google SigLIP2 model card: `google/siglip2-so400m-patch16-384`. https://huggingface.co/google/siglip2-so400m-patch16-384
- Meta DINOv2 model card: `facebook/dinov2-large`. https://huggingface.co/facebook/dinov2-large
- Meta DINOv3 model card: `facebook/dinov3-vitl16-pretrain-lvd1689m`. https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- OpenAI CLIP model card: `openai/clip-vit-large-patch14`. https://huggingface.co/openai/clip-vit-large-patch14
- Zhuang Liu et al. "A ConvNet for the 2020s." arXiv:2201.03545. https://arxiv.org/abs/2201.03545
- Ilya Loshchilov and Frank Hutter. "Decoupled Weight Decay Regularization." arXiv:1711.05101. https://arxiv.org/abs/1711.05101
- Maxime Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision." arXiv:2304.07193. https://arxiv.org/abs/2304.07193
- Alec Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." arXiv:2103.00020. https://arxiv.org/abs/2103.00020
