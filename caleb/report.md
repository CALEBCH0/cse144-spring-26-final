# CSE 144 Final Report

## 1 Introduction

1. **Problem goal and setting.** We tackle a 100-class image classification task on an extremely small labeled dataset: 1079 training images (~10.8 images/class on average, ranging from 4 to 41) and 1036 unlabeled test images. The goal is to maximise accuracy on the Kaggle public leaderboard (competition: UCSC CSE 144 Spring 2026 Final Project). The dataset's scale — roughly one order of magnitude below what is typically used for fine-tuning large vision models — makes generalisation the central challenge: every design decision must trade off accuracy against overfitting.

2. **Why transfer learning is appropriate.** With ~10 images per class, training any modern vision model from scratch is infeasible — the network would memorise the training set within a few epochs and fail to generalise. Transfer learning sidesteps this by initialising from a backbone pretrained on hundreds of millions to billions of images, where the lower layers have already learned universal low-level features (edges, textures) and higher layers encode rich semantic representations. Fine-tuning only the top 2–6 transformer blocks while freezing the rest dramatically reduces the effective number of free parameters, allowing the model to adapt its final representations to the 100-class vocabulary without destroying the pretrained features. This is the only practical approach at this dataset scale — as confirmed empirically: a fully fine-tuned DINOv2-Large achieves 77.4% while partial unfreezing (2 blocks) reaches 87.4%.

3. **Brief summary of approach and main result.** Eight pretrained backbones were evaluated across 121 hyperparameter configurations using 5-fold stratified cross-validation. The dominant backbone is SigLIP2-SO400M-384 (`google/siglip2-so400m-patch16-384`), a 400M-parameter ViT-So400M pretrained on billions of image-text pairs at native 384px resolution. With 2 unfrozen transformer layers, lr=1e-4, LLRD=0.8, and 20 training epochs, it achieves **95.27% CV accuracy**. Self-distillation pseudo-labeling (threshold=0.97 on test predictions) adds +0.09% in a second training round. Final Kaggle public leaderboard score: **95.454%** (5th place). Post-submission, a per-class routing ensemble (`class_router.py`) was developed — SigLIP2 serves as the backbone for 87–88 classes while secondary models (DINOv3, CLIP-ViT) replace specific probability columns for hard classes where they show large, consistent per-class F1 advantages.

---

## 2 Dataset

1. **Number of classes and dataset sizes.** 100 classes; 1079 total labeled images split into ~917 train / ~162 validation (85/15 split, seed 42). Test set: 1,036 images (unlabeled).
2. **Directory structure and label mapping.** ImageFolder format under `data/train/`; labels 0–99 assigned alphabetically by folder name via HuggingFace `datasets` `imagefolder` loader.
3. **Preprocessing and augmentation.**
   - Train: `RandomResizedCrop`, `RandomHorizontalFlip`, `ToTensor`, ImageNet normalization
   - Validation: `Resize`, `CenterCrop`, `ToTensor`, ImageNet normalization

---

## 3 Implementation

### 3.1 Model

Eight pretrained backbones were evaluated across the experiment:

| Model | HuggingFace ID | Pretrained On |
|---|---|---|
| ResNet-50 | `microsoft/resnet-50` | ImageNet-1K (1,000 classes) |
| ResNet-101 | `microsoft/resnet-101` | ImageNet-1K (1,000 classes) |
| ConvNeXt-Base | `facebook/convnext-base-224-22k-1k` | ImageNet-22K → fine-tuned on 1K |
| ViT-B/16 | `google/vit-base-patch16-224` | ImageNet-21K → fine-tuned on 1K |
| DINOv2-Base | `facebook/dinov2-base` | Self-supervised DINO (unlabeled ImageNet); 12 encoder layers |
| DINOv2-Large | `facebook/dinov2-large` | Self-supervised DINO (unlabeled ImageNet); 24 encoder layers |
| DINOv3 ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINO distillation from ViT-7B teacher; LVD-1689M (1.689B images); 24 encoder layers |
| CLIP-ViT-Large/14 | `openai/clip-vit-large-patch14` | Image-text contrastive (CLIP) on 400M pairs; ViT-L/14; 24 encoder layers |
| SigLIP2-SO400M-384 | `google/siglip2-so400m-patch16-384` | Image-text contrastive (SigLIP2) on billions of pairs; ViT-So400M; 27 encoder layers; native 384px |

1. **Pretrained backbone.** ResNet-50/101 chosen as standard baselines. ConvNeXt-Base chosen as a modern CNN with 22K pretraining. DINOv2 was the best early model; the search progressively shifted toward image-text pretrained models (SigLIP2, CLIP-ViT) and the stronger DINO distillation model (DINOv3). SigLIP2-SO400M-384 is the final best backbone: 400M parameters, native 384px pretraining on billions of image-text pairs, with only 2 of 27 layers unfrozen to achieve 95.27% CV accuracy.
2. **Architecture changes.** The final classification head was replaced: original 1000-class linear layer reinitialized to 100 classes (`ignore_mismatched_sizes=True`). All other weights initialized from pretrained checkpoint.
3. **Fine-tuning strategy.** Partial fine-tuning with selective layer unfreezing. By default, the top 4 backbone blocks/stages are unfrozen alongside the classifier head. Five additional techniques are applied:

   | Technique | Description | Implementation |
   |---|---|---|
   | Layer-wise LR Decay (LLRD) | Earlier layers receive a lower LR multiplied by `decay_factor` per layer; classifier gets full `base_lr` | `get_llrd_optimizer()` in `utils.py` |
   | Mixup | Pairs of images blended with Beta(0.2, 0.2) lambda; cross-entropy on soft one-hot labels | `MixupCollator` + `CustomTrainer` in `utils.py` |
   | Progressive Unfreezing | Backbone frozen for first 3 epochs (head stabilizes), then fully unfrozen | `ProgressiveUnfreezeCallback` in `utils.py` |
   | Test-Time Augmentation (TTA) | Inference run 5× with random train augmentations; softmax probs averaged | `predict_with_tta()` in `utils.py` |
   | Ensemble | Logits from all 5 fold models averaged; final prediction from argmax | `ensemble_predict()` in `utils.py` |

   **Techniques considered but not implemented:**
   - Stochastic Weight Averaging (SWA) / Exponential Moving Average (EMA)
   - Focal loss for hard-class emphasis
   - Knowledge distillation (large → small model)
   - Larger backbone variants (SwinV2-Base, EfficientNetV2-M, DeiT III)

### 3.2 Training

1. **Loss function and optimizer.** Cross-entropy loss with label smoothing (0.05). AdamW optimizer with LLRD: classifier head uses `base_lr`, each encoder layer below is multiplied by `decay_factor=0.75`.
2. **Hyperparameters (CV run).**

| Parameter | ConvNeXt-Base | ViT-B/16 | DINOv2-Base | DINOv2-Large | SigLIP2-SO400M |
|---|---|---|---|---|---|
| Learning rate (base) | **3e-4** | **1e-4** | **1e-4** | **1e-4** | **1e-4** |
| Batch size | 32 | 32 | 32 | 32 | 16 |
| Epochs | 10 | 10 | **10** | **30** | **20** |
| Weight decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| LR scheduler | linear | cosine | linear | linear | linear |
| Unfreeze blocks | 4 | **4** | **2** | **2** | **2** |
| LLRD decay factor | — (disabled) | **0.85** | **0.85** | **0.75** | **0.80** |
| Mixup alpha | — (disabled) | — | — | — | — |
| RandAugment | — | — | — | — | — |
| Progressive unfreeze | — | — | **OFF** | **OFF** | **OFF** |
| TTA augments | 5 | 5 | 5 | 5 | 5 |

Bold values reflect the best configuration found via hyperparameter search. DINOv2-base peaks at 10 epochs; DINOv2-Large benefits from 30 epochs; SigLIP2 benefits from 20 epochs. LLRD is beneficial for ViT (0.85), DINOv2-base (0.85), DINOv2-Large (0.75), and SigLIP2 (0.80) but neutral for ConvNeXt. Mixup, CutMix, and RandAugment disabled for all DINO/SigLIP2 models — aug degrades already-robust pretrained features. Progressive unfreeze disabled for DINOv2 and SigLIP2 — see section 5.5.9.

3. **Hardware/software.** NVIDIA RTX 5070 Ti (16 GB VRAM); PyTorch 2.12.0+cu128; mixed precision (fp16); `dataloader_num_workers=0` (Windows WSL2 multiprocessing constraint). Scripts are run via Windows Python (`.venv-win/Scripts/python.exe`) from WSL2 to avoid the `/mnt/c/` filesystem bridge overhead for native CUDA throughput.

---

## 4 Experiments

1. **Baseline setup.** All three models trained with standard resize/flip augmentation, 5-fold stratified CV, AdamW, label smoothing 0.05. Results reported in sections 5.1–5.4.
2. **Hyperparameter tuning method.** Grid search via `search.py`: each config trained with 5-fold stratified CV (batch=32). Results written incrementally to `search_results.csv` with resume support. 121 total configs covering unfreeze depth, LR, scheduler, label smoothing, weight decay, LLRD, augmentation stack, collator (Mixup/CutMix), LoRA rank, epoch count, model sizes (Base/Large/518px/512px), class-balanced loss, upsample balance, and new architectures (SigLIP2-SO400M, CLIP-ViT-Large/14, DINOv3). Each config also generates a Kaggle-submittable CSV in `search_results/config_{i}_{model}.csv`. See section 5.5 for findings.
3. **Ablation axes covered:**
   - Freeze depth: head-only → partial unfreeze (1–8 blocks) → full fine-tune
   - LR: 1e-5 to 3e-4 per model family
   - Scheduler: linear vs cosine
   - LLRD: decay factors 0.65, 0.75, 0.80, 0.85
   - Augmentation: ColorJitter, RandAugment, RandomErasing (isolated and combined)
   - Collator: none, Mixup, CutMix, MixupCutMix
   - PEFT: LoRA rank 4, 8, 16 on DINOv2 and ViT
   - Epochs: 10, 15, 20, 30, 50 (varied by model family)
   - Model scale: Base vs Large (DINOv2); resolution (384px vs 512px for SigLIP2; 224px vs 518px for DINOv2-Large)
   - Architecture family: CNN (ResNet, ConvNeXt) → ViT supervised (ViT-B) → DINO self-supervised (DINOv2-Base/Large, DINOv3) → image-text contrastive (CLIP-ViT, SigLIP2)
   - Class balance: class weights vs upsample balance (min 20 per class)
   - Pseudo-labeling: self-distillation vs blended ensemble source (threshold 0.97)
   - Per-class routing: SigLIP2 backbone + secondary columns for hard classes (DINOv3, CLIP-ViT)

---

## 5 Results

### 5.1 Baseline Run (Epoch 10)

| Model | Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| ResNet-50 | Train | 0.1287 | 0.1085 | 0.0710 | 0.0594 |
| ResNet-50 | Validation | 0.0679 | 0.0134 | 0.0337 | 0.0171 |
| ResNet-101 | Train | 0.2236 | 0.1953 | 0.1499 | 0.1219 |
| ResNet-101 | Validation | 0.1481 | 0.0774 | 0.1118 | 0.0812 |
| ConvNeXt-Base | Train | 0.8942 | 0.9141 | 0.8840 | 0.8878 |
| ConvNeXt-Base | Validation | **0.6790** | 0.6491 | 0.6467 | 0.6239 |

All metrics are macro-averaged across 100 classes.

### 5.2 Extended Run — ConvNeXt-Base vs ConvNeXt-Large (batch=128, lr=5e-5, 10 epochs, GPU)

| Model | Split | Accuracy | Precision | Recall | F1 | Train-Val Gap |
|---|---|---|---|---|---|---|
| ConvNeXt-Base | Train | 0.6925 | 0.6778 | 0.6402 | 0.6247 | |
| ConvNeXt-Base | Validation | 0.4568 | 0.3441 | 0.3819 | 0.3327 | 23 pts |
| ConvNeXt-Large | Train | 0.8353 | 0.8414 | 0.8057 | 0.8019 | |
| ConvNeXt-Large | Validation | **0.5617** | 0.4866 | 0.5244 | 0.4814 | 27 pts |

ConvNeXt-Large improves validation accuracy by ~9 points over Base but overfits more (27 vs 23 point train-val gap), consistent with larger capacity memorizing the small training set.

### 5.3 5-Fold Stratified CV — DINOv2 / ConvNeXt-Tiny / ConvNeXt-Base / ViT (batch=32, label_smoothing=0.05, 10 epochs)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| DINOv2-Base (frozen) | 0.4133 ± 0.0334 | 0.3189 ± 0.0415 | 0.3645 ± 0.0425 | 0.3196 ± 0.0398 |
| ConvNeXt-Tiny | 0.2410 ± 0.0229 | 0.1449 ± 0.0338 | 0.1860 ± 0.0191 | 0.1397 ± 0.0219 |
| ConvNeXt-Base | **0.7071 ± 0.0274** | 0.6849 ± 0.0298 | 0.6980 ± 0.0250 | 0.6702 ± 0.0278 |
| ViT-B/16 (timm) | **0.7081 ± 0.0231** | 0.7063 ± 0.0383 | 0.7010 ± 0.0267 | 0.6795 ± 0.0295 |

All metrics are macro-averaged across 100 classes. Mean ± std across 5 folds.

**Key observations:**
- ConvNeXt-Base and ViT-B/16 are statistically tied (~70.7–70.8% accuracy, overlapping std). Both benefit from ImageNet-22K/21K pretraining.
- DINOv2 with a fully frozen backbone achieves only 41.3% — strong for a linear probe but limited by not updating any backbone features. Partial unfreezing expected to improve this significantly.
- ConvNeXt-Tiny underperforms at 24.1%, likely because `facebook/convnext-tiny-224` was only pretrained on ImageNet-1K (vs 22K for Base), giving it weaker feature representations for transfer.
- Low std values (~2–3%) confirm the 5-fold CV estimates are stable.

### 5.4 5-Fold CV with All Techniques — DINOv2 / ConvNeXt-Base / ConvNeXt-Large / ViT (batch=32, 10 epochs, LLRD + Mixup + Progressive Unfreeze + TTA + Ensemble)

**Techniques active:** LLRD (decay=0.75), Mixup (α=0.2), Progressive Unfreeze (epoch 3), TTA (5 passes), Ensemble (5-fold average). Top 4 backbone blocks unfrozen initially.

| Model | Accuracy | Precision | Recall | F1 | Classes F1>0 |
|---|---|---|---|---|---|
| DINOv2-Base | 0.4133 ± 0.0334 | 0.3189 ± 0.0415 | 0.3645 ± 0.0425 | 0.3196 ± 0.0398 | 84/100 |
| ConvNeXt-Base | 0.7071 ± 0.0274 | 0.6849 ± 0.0298 | 0.6980 ± 0.0250 | 0.6702 ± 0.0278 | 98/100 |
| ConvNeXt-Large | **0.7785 ± 0.0302** | **0.7667 ± 0.0415** | **0.7756 ± 0.0284** | **0.7499 ± 0.0340** | 99/100 |
| ViT-B/16 | 0.6942 ± 0.0247 | 0.6800 ± 0.0312 | 0.6877 ± 0.0300 | 0.6634 ± 0.0283 | 98/100 |

All metrics macro-averaged across 100 classes. Mean ± std across 5 folds.

**Key observations:**
- ConvNeXt-Large is the clear winner at **77.85%** — a +7.1 point gain over Base (70.71%) and +13.5 points over the earlier single-split ConvNeXt-Large baseline (section 5.2). The selective unfreezing + progressive unfreeze strategy substantially reduces the overfitting seen in the full fine-tune baseline.
- DINOv2 unchanged at 41.3% — the 4-block partial unfreeze with progressive unfreeze did not noticeably improve over the fully frozen linear probe result. DINOv2 likely requires more epochs or higher LR once unfrozen.
- ViT-B/16 dropped slightly vs section 5.3 (0.6942 vs 0.7081), suggesting Mixup and/or LLRD may slightly hurt ViT at this dataset size. Further ablation needed.
- Class 66 scores F1=0.0000 across all four models — a genuinely hard class. Classes 64, 68, 71, 76, 86, 98 also score below 0.15 for all models.
- ConvNeXt-Large achieves F1=1.0 on 39/100 classes, vs 33/100 for Base — the larger pretrained capacity provides better feature coverage.

**Hard classes (F1 < 0.30 across all models):**

| Class | DINOv2 | ConvNeXt-B | ConvNeXt-L | ViT |
|---|---|---|---|---|
| 66 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 64 | 0.0000 | 0.2000 | 0.1333 | 0.0500 |
| 76 | 0.0000 | 0.1333 | 0.2667 | 0.0000 |
| 86 | 0.0000 | 0.0000 | 0.4267 | 0.3800 |
| 71 | 0.1467 | 0.2800 | 0.2286 | 0.1143 |

### 5.5 Hyperparameter Search — Key Findings (`search_results.csv`, 78/78 configs complete)

#### 5.5.1 DINOv2 unfreeze depth (biggest lever)

| unfreeze_blocks | acc | f1 |
|---|---|---|
| 0 (head only) | 0.4105 ± 0.0364 | 0.3175 |
| **2** | **0.8424 ± 0.0172** | **0.8194** |
| 4 | 0.8257 ± 0.0183 | 0.8043 |
| 12 (full FT, lr=1e-5) | 0.7739 ± 0.0155 | 0.7401 |

**Best overall config in the entire search.** Unfreezing just 2 top encoder layers achieves 84.24% — exceeding ConvNeXt-Large (77.85%) by +6.4 points. Unfreezing more layers _hurts_: 4 blocks −1.7 pts, full fine-tune −6.8 pts. Classic catastrophic forgetting on ~1,000 images; only the top two DINOv2 layers need adapting.

#### 5.5.2 ConvNeXt-Base LR sweep (unfreeze=4 stages)

| lr | acc |
|---|---|
| 1e-5 | 0.3160 |
| 3e-5 | 0.6497 |
| 5e-5 | 0.7146 |
| 1e-4 | 0.7544 |
| **3e-4** | **0.7627** |

ConvNeXt requires 10× higher LR than DINOv2. The frozen backbone + lr=1e-3 (head only) also performs surprisingly well at 73.87%, confirming that for CNN backbones, a high-LR head-only strategy can compete with partial fine-tuning.

#### 5.5.3 LLRD — no meaningful benefit

All LLRD decay factors (0.65 / 0.75 / 0.85) on ConvNeXt cluster within ±0.05% of the no-LLRD baseline (71.73%). LLRD is disabled in the final pipeline.

#### 5.5.4 Augmentation ablation (ConvNeXt, unfreeze=4, cosine)

| Augmentation | acc | Δ vs baseline |
|---|---|---|
| None (baseline) | 0.7173 | — |
| **+RandAugment only** | **0.7266** | **+0.93%** |
| +ColorJitter + RA + RandomErasing (full) | 0.7174 | +0.01% |
| +ColorJitter only | 0.6997 | −1.76% |

RandAugment alone gives the best gain. ColorJitter alone hurts — likely erasing subtle color discriminative cues the ConvNeXt backbone relies on. The full stack cancels out RandAugment's benefit when combined with ColorJitter.

#### 5.5.5 ViT-B/16 LR sweep (frozen backbone, unfreeze=0)

| Config | LR | Scheduler | Weight Decay | Acc |
|---|---|---|---|---|
| 12 | 1e-5 | linear | 0.01 | 0.0232 |
| 13 | 3e-5 | linear | 0.01 | 0.1010 |
| 14 | 5e-5 | linear | 0.01 | 0.2290 |
| **15** | **1e-4** | **linear** | **0.01** | **0.4495** |
| 18 | 5e-5 | cosine | 0.01 | 0.2392 |
| **19** | **1e-4** | **cosine** | **0.01** | **0.4523** |
| 24 | 5e-5 | linear | 0.05 | 0.2290 |

With a frozen backbone (head-only training), ViT-B/16 tops out at **45.2%** — dramatically below DINOv2 (84.2%) under comparable conditions. The LR sensitivity is steep: lr=1e-5 collapses to 2.3% while lr=1e-4 reaches 45%. Cosine scheduler matches linear at the same LR (45.23% vs 44.95%). Higher weight decay (0.05) offers no improvement over the default (0.01). Backbone unfreezing (configs 28–30) closes this gap substantially — see section 5.5.7.

#### 5.5.6 LoRA (frozen backbone)

| Model | rank r | acc | vs head-only |
|---|---|---|---|
| DINOv2 LoRA | 4 | 0.6562 | +24.6 pts |
| DINOv2 LoRA | 8 | 0.6692 | +25.9 pts |
| DINOv2 LoRA | **16** | **0.6719** | **+26.1 pts** |
| ViT LoRA | 4 | 0.5496 | +4.7 pts vs cfg 15 |
| ViT LoRA | 8 | 0.5515 | +4.9 pts |
| ViT LoRA | 16 | 0.5505 | +4.8 pts |

LoRA dramatically beats frozen head-only DINOv2 (41% → 67%) but cannot match partial unfreezing (84%). LoRA rank makes almost no difference — r=4 and r=16 are within 0.16%. For ViT, LoRA at ~55% modestly outperforms the best frozen-backbone baseline (45%), but still leaves ViT well below DINOv2. LoRA is useful for parameter-efficient deployment but not accuracy maximization at this dataset scale.

#### 5.5.7 ViT-B/16 with partial unfreezing + LLRD (configs 28–30)

| Config | unfreeze | LLRD factor | Acc | F1 | train/fold |
|---|---|---|---|---|---|
| 28 | 4 | 0.65 | 0.7146 | 0.6875 | 45s |
| 29 | 4 | 0.75 | 0.7183 | 0.6903 | 45s |
| **30** | **4** | **0.85** | **0.7257** | **0.6974** | **47s** |

Unfreezing 4 ViT encoder blocks with LLRD pushes accuracy from 45% (frozen) to **72.6%** — a +27 point jump. The gentlest decay (0.85) works best, suggesting ViT's pretrained features need only light adjustment. At 72.6%, ViT is competitive with ConvNeXt-Base (76.3%) and within reach of ConvNeXt-Large (77.9%), though still 12 points below DINOv2 (84.2%).

**Note:** unlike ConvNeXt, LLRD is beneficial for ViT — without it the frozen ViT reaches only 45%. The difference is that LLRD enables backbone layers to update while the frozen run trains the head alone.

#### 5.5.8 Collator ablation (ConvNeXt, full aug stack + Mixup / CutMix / MixupCutMix)

These configs use ConvNeXt-Base with the full augmentation stack (CJ + RA + RE) and add a label-mixing collator on top. All three **hurt** accuracy relative to no collator (config 34: 71.74%):

| Config | Collator | Acc | F1 | Δ vs no-collator |
|---|---|---|---|---|
| 34 | none | 0.7174 | 0.6798 | — |
| 35 | Mixup (α=0.2) | 0.6979 | 0.6555 | **−1.95%** |
| 36 | CutMix (α=1.0) | 0.6544 | 0.5981 | **−6.30%** |
| 37 | MixupCutMix | 0.6608 | 0.6097 | **−5.66%** |

All three collators consistently hurt. With ~10 images/class and 10 training epochs, soft labels from label blending add noise faster than they regularize. CutMix's large rectangular patches are particularly disruptive at this dataset scale. **Both Mixup and CutMix are disabled in the final pipeline.**

*(These configs previously failed due to a bug where `MixupCollator` applied label blending to validation images during prediction — HuggingFace Trainer iterates the DataLoader outside `torch.no_grad()`, so the grad-enabled check did not fire. Fix: swap to a plain collator before `.predict()`.)*

#### 5.5.9 Progressive unfreeze incompatibility with DINOv2

Running `pipeline.py` with `USE_PROGRESSIVE_UNFREEZE=True` and `UNFREEZE_BLOCKS=2` produced only **63.02%** — far below the 84.24% search result. The cause: `ProgressiveUnfreezeCallback` unfreezes the **entire model** at epoch 3 (not just the top-N blocks), effectively switching to full fine-tuning at `lr=5e-5` for the remaining 7 epochs. Since the search showed full fine-tuning (unfreeze=12, `lr=1e-5`) only reaches 77.4% — and at `lr=5e-5` it collapses further — the progressive unfreeze was actively destroying the learned representations.

**Fix:** `USE_PROGRESSIVE_UNFREEZE = False` for DINOv2. The 84.24% result used a fixed unfreeze=2 for all 10 epochs with no progressive strategy.

Progressive unfreeze is appropriate for ConvNeXt (where it stabilizes the head before full fine-tuning) but is incompatible with the DINOv2 strategy of keeping the backbone mostly frozen throughout training.

#### 5.5.10 DINOv2-Base LR sweep at unfreeze=2 (configs 44–47)

| lr | acc | notes |
|---|---|---|
| 1e-5 | 0.7489 | too conservative; features don't adapt |
| 3e-5 | 0.8313 | |
| **5e-5** | **0.8424** | original best (config 1) |
| 1e-4 | 0.8359 | slight drop vs 5e-5 at 10ep alone |
| 3e-4 | 0.8007 | too aggressive; catastrophic forgetting |

5e-5 is the optimal standalone LR. However, lr=1e-4 combined with LLRD (section 5.5.15) ultimately exceeds 5e-5 — LLRD shields the lower layers from the higher LR.

#### 5.5.11 DINOv2-Base LLRD sweep (configs 48–50)

| LLRD factor | acc | delta vs no LLRD (0.8424) |
|---|---|---|
| 0.65 | 0.8415 | −0.09% |
| 0.75 | 0.8425 | +0.01% |
| **0.85** | **0.8434** | **+0.10%** |

Unlike ConvNeXt (where LLRD was neutral), DINOv2-base shows a small but consistent benefit. Gentle decay (0.85) preserves more of the top-layer gradient while still protecting lower representations. The payoff is modest at 5e-5; it becomes significant when combined with a higher base LR (see section 5.5.15).

#### 5.5.12 DINOv2-Base augmentation ablation (configs 51–52)

| augmentation | acc | delta vs baseline (0.8424) |
|---|---|---|
| +RandAugment only | 0.8406 | −0.18% |
| +CJ + RA + RE (full) | 0.8322 | −1.02% |

Both hurt. DINOv2's DINO-pretrained features are already highly invariant; additional augmentation degrades fine-grained discriminative signals without providing useful generalisation variance at ~10 images/class. Compare to ConvNeXt where RA alone gave +0.93%.

#### 5.5.13 ViT-B/16 augmentation ablation (configs 53–55)

Best ViT base: unfreeze=4, LLRD=0.85, lr=5e-5, cosine (0.7257, config 30).

| augmentation | acc | delta |
|---|---|---|
| baseline | 0.7257 | — |
| +RandAugment only | 0.7331 | +0.74% |
| **+ColorJitter only** | **0.7442** | **+1.85%** |
| +CJ + RA + RE (full) | 0.7257 | 0.00% |

ColorJitter alone is the best ViT augmentation — a different pattern from ConvNeXt (where CJ hurt −1.76%). ViT's patch attention mechanism benefits from color variation; ConvNeXt's deep feature hierarchy is more sensitive to color shift. The full aug stack cancels out gains, consistent with the ConvNeXt finding.

**Updated best ViT:** 74.42% (config 54: unfreeze=4, LLRD=0.85, +ColorJitter).

#### 5.5.14 DINOv2-Large — introduction and initial results (configs 56–63)

DINOv2-Large (`facebook/dinov2-large`) uses 24 encoder layers vs 12 for Base. A layer-count bug in `apply_freeze` (hardcoded to 12) caused configs 57–58 to unfreeze layers 10–11 (middle of the 24-layer network) instead of 22–23 (top two). Configs 57–58 are invalid (acc ~0.52). Fixed by adding `_count_transformer_layers()` which scans parameter names dynamically.

Post-fix results (10-epoch baseline):

| config | unfreeze | lr | epochs | aug | acc |
|---|---|---|---|---|---|
| 56 | 0 (linear probe) | 5e-5 | 10 | — | 0.4986 |
| 59 | 2 | 1e-5 | 10 | — | 0.7803 |
| 60 | 2 | 1e-4 | 10 | — | 0.8517 |
| 61 | 2 | 5e-5 | 20 | — | 0.8545 |
| **62** | **2** | **5e-5** | **30** | — | **0.8665** |
| 63 | 2 | 5e-5 | 10 | RandAugment | 0.8573 |

**Key finding:** unlike DINOv2-base (which peaks at 10 epochs), Large benefits substantially from extended training. 30ep → 0.8665 vs 10ep (config 61 20ep → 0.8545 → config 60 10ep → 0.8517). Each additional epoch block adds ~1 point. DINOv2-Large has more capacity to absorb the small dataset over more passes without the catastrophic forgetting seen with full fine-tuning.

#### 5.5.15 DINOv2-Base best combos — lr=1e-4 + LLRD + depth (configs 64–69)

Combining the best LR (1e-4) with the best LLRD factor (0.85) and probing deeper unfreeze + longer training:

| config | unfreeze | lr | LLRD | epochs | acc |
|---|---|---|---|---|---|
| **64** | 2 | **1e-4** | **0.85** | 10 | **0.8554** |
| 65 | 2 | 1e-4 | 0.75 | 10 | 0.8517 |
| 66 | 4 | 1e-4 | 0.85 | 10 | 0.8146 |
| 67 | 4 | 5e-5 | 0.85 | 10 | 0.8387 |
| 68 | 6 | 5e-5 | 0.85 | 10 | 0.8146 |
| 69 | 2 | 1e-4 | 0.85 | **20** | 0.8526 |

**New DINOv2-base best: 85.54%** — combining lr=1e-4 with LLRD=0.85 exceeds the prior best (5e-5 no LLRD) by +1.3 points. The LLRD allows a higher head/top-layer LR while keeping lower layers at 1e-4 × 0.85^n. Unfreeze=4 and =6 still hurt significantly, confirming that 2 top blocks is the correct freeze depth regardless of LLRD. Extending to 20ep regresses (0.8526 < 0.8554) — DINOv2-base overfits past 10 epochs on this dataset.

#### 5.5.16 DINOv2-Large overnight — best configs (configs 70–77)


Combining all best findings (lr=1e-4, 30ep, LLRD, selective augmentation) for Large:

| config | lr | LLRD | epochs | aug | acc |
|---|---|---|---|---|---|
| 70 | 1e-4 | 0.85 | 30 | — | 0.8665 |
| 71 | 1e-4 | 0.85 | 20 | — | 0.8564 |
| **72** | **1e-4** | **0.75** | **30** | — | **0.8684** |
| 73 | 1e-4 | 0.85 | 30 | RandAugment | 0.8638 |
| 74 | 1e-4 | none | 30 | — | 0.8619 |
| 75 | 5e-5 | 0.85 | 30 | — | 0.8563 |
| 76 (base) | 1e-4 | 0.85 | 20 | — | 0.8499 |
| 77 (base) | 1e-4 | 0.85 | 30 | — | 0.8406 |

**New overall best: 86.84%** (config 72: DINOv2-Large, unfreeze=2, lr=1e-4, LLRD=0.75, 30ep).

Key takeaways:
- **LLRD matters for Large at lr=1e-4:** LLRD=0.75 → 0.8684 vs no LLRD → 0.8619 (+0.65 pts); LLRD=0.85 → 0.8665. Gentler decay (0.75) performs best, allowing the top layers more gradient while protecting lower representations from a high LR.
- **30 epochs is necessary:** 20ep → 0.8564, 30ep → 0.8665 (same LLRD=0.85, lr=1e-4). Large consistently absorbs more training without overfitting.
- **RandAugment still slightly hurts:** 0.8638 vs 0.8665 at same settings — consistent with the DINOv2-base augmentation finding.
- **DINOv2-base with 20/30ep regresses:** 76 (20ep) → 0.8499, 77 (30ep) → 0.8406 vs best base 10ep → 0.8554. Base overfits past 10 epochs; Large does not.
- **lr=5e-5 vs 1e-4 for Large:** at 30ep, 5e-5 + LLRD=0.85 → 0.8563, 1e-4 + LLRD=0.75 → 0.8684. Higher LR requires LLRD to protect earlier layers but yields +1.2 points.

#### 5.5.17 LLRD lower bound for DINOv2-Large (config 78)

| config | LLRD | epochs | acc | delta vs best (0.8684) |
|---|---|---|---|---|
| 72 | 0.75 | 30 | 0.8684 | — (best) |
| 70 | 0.85 | 30 | 0.8665 | −0.19% |
| **78** | **0.65** | **30** | **0.8610** | **−0.74%** |

LLRD=0.65 hurts — aggressively cutting the LR on lower layers starves the upper-layer gradients. The optimal range is 0.75–0.85; 0.75 is the sweet spot for Large at lr=1e-4.

#### 5.5.18 Extended epochs — 40ep and 50ep for DINOv2-Large (configs 79–80)

| config | epochs | acc | delta vs 30ep best |
|---|---|---|---|
| 72 | 30 | 0.8684 | — |
| 79 | 40 | 0.8647 | −0.37% |
| **80** | **50** | **0.8739** | **+0.55%** |

**New overall best: 87.39%** (config 80: DINOv2-Large, 50ep, LLRD=0.75, unfreeze=2, lr=1e-4).

The training curve is non-monotonic: 40ep dips below 30ep (−0.37%) before 50ep surpasses both (+0.55% over 30ep). This likely reflects fold-level variance and the optimizer traversing a loss plateau at 40ep; 50ep escapes it. The key finding is that DINOv2-Large's capacity is still not saturated at 30 epochs on this dataset, and 50ep delivers a meaningful gain without signs of catastrophic forgetting.

#### 5.5.19 Cosine scheduler, deeper unfreeze, and higher LR (configs 81–84)

| config | model | unfreeze | lr | LLRD | epochs | scheduler | acc | note |
|---|---|---|---|---|---|---|---|---|
| 72 (ref) | dinov2_large | 2 | 1e-4 | 0.75 | 30 | linear | 0.8684 | prior best |
| 81 | dinov2_large | 2 | 1e-4 | 0.75 | 30 | **cosine** | 0.8610 | −0.74% |
| **82** | **dinov2_large** | **4** | **1e-4** | **0.65** | **30** | linear | **0.8711** | **+0.27%** |
| 83 | dinov2_large | 2 | **2e-4** | 0.65 | 30 | linear | 0.8563 | −1.21% |
| 84 | dinov2 (base) | 2 | 1e-4 | 0.85 | 10 | cosine | 0.8415 | base cosine baseline |

Key takeaways:
- **Cosine scheduler hurts Large at 30ep** (0.8610 vs 0.8684 linear). Linear warmup-decay is consistently better for DINOv2-Large; cosine likely decays the LR too early, under-utilising later epochs.
- **unfreeze=4 with LLRD=0.65 reaches 87.11%** — competitive with config 72, and within std overlap. Deeper unfreezing can recover when paired with an aggressively lower LLRD to protect the extra exposed layers. However, config 80 (50ep, unfreeze=2) at 87.39% is still superior: fewer exposed layers means less risk of overwriting pretrained representations.
- **lr=2e-4 hurts** even with aggressive LLRD=0.65. The upper layers receive 2e-4 full LR on only ~860 training samples — too much gradient noise per step even with decay protection.
- **DINOv2-base with cosine** (config 84) scores 84.15%, well below the best base config 64 (85.54% linear). Consistent with the finding that cosine degrades DINOv2 for both sizes.

#### 5.5.20 518px resolution and RandAugment on DINOv2-Large (configs 85–87)

| config | model | res | epochs | aug | acc | Δ vs 72 (30ep ref) |
|---|---|---|---|---|---|---|
| 72 (ref) | dinov2_large | 224px | 30 | none | 0.8684 | — |
| 85 | dinov2_large_518 | **518px** | 30 | none | 0.7720 | **−9.64%** |
| 86 | dinov2_large | 224px | 30 | **RandAugment** | 0.8693 | +0.09% |
| 87 | dinov2_large_518 | **518px** | 30 | **RandAugment** | 0.7627 | −10.57% |

Key takeaways:
- **518px is definitively harmful: −9.64 pts vs 224px at identical settings.** DINOv2-Large was pretrained at 518px, but the fine-tuning regime with ~860 train images cannot support the 5× larger token sequence (1369 patches at 518px vs 256 at 224px). The expanded context introduces positional interpolation noise and drastically increases per-sample memory, forcing batch_size=16 (vs 32), which further destabilises gradient estimates at this dataset scale.
- **The "more information at higher resolution" hypothesis is refuted.** Config 85 used 30 full epochs — sufficient for the 224px Large to reach 86.84% — yet 518px still yields only 77.2%. The bottleneck is not training duration but the mismatch between the positional embedding interpolation and the tiny fine-tuning set.
- **RandAugment is neutral on DINOv2-Large at 224px** (86.93% vs 86.84%, +0.09%). This contrasts with DINOv2-Base where RA gave +0.93%. Large's higher capacity already generalises well from DINOv2 pretraining; augmentation offers no additional regularisation benefit. This is consistent with the earlier finding that Large can absorb 50 epochs without overfitting.
- **RandAugment makes 518px worse** (76.27% vs 77.20%, −0.93%). Spatial augmentations compound the positional interpolation artefacts at high resolution. Both 518px configs are firmly ruled out.
- **Best config remains config 80** (50ep, 224px, no aug): 87.39%.

#### 5.5.21 SigLIP-Base baselines (configs 88–90)

| config | model | epochs | class_weights | acc | note |
|---|---|---|---|---|---|
| 88 | siglip-base | 10 | no | 0.8332 | baseline |
| 89 | siglip-base | 10 | yes | 0.8257 | −0.75% |
| **90** | **siglip-base** | **20** | **no** | **0.8461** | **+1.29%** |

SigLIP-Base (google/siglip-base-patch16-224, unfreeze=2, lr=1e-4) at 10 epochs reaches 83.3% — slightly below DINOv2-Base (85.54%) and well below DINOv2-Large (87.39%). Class weights hurt (same pattern as ConvNeXt and DINOv2-Large). Doubling epochs to 20 improves to 84.6%, suggesting SigLIP benefits from longer training; the gap to DINOv2-Large persists (~3 pts). SigLIP-Base is not competitive with DINOv2-Large for this task; the much larger SigLIP2-SO400M variant (configs 96+) addresses this.

#### 5.5.22 CLIP-ViT-Large/14 optimised search (configs 91–92, run_006)

CLIP-ViT-Large/14 (`openai/clip-vit-large-patch14`, ViT-L/16 with image-text contrastive pretraining on 400M image-text pairs) with two optimised configs informed by DINOv3's sweep — both use LLRD=0.75, 30ep, linear scheduler, lr=1e-4, which were the sweet-spot settings for the same ViT-L architecture.

| Config | Unfreeze | LLRD | Epochs | Acc | Std | F1 |
|---|---|---|---|---|---|---|
| 91 | 2 | 0.75 | 30 | 0.8656 | 0.0172 | 0.8485 |
| **92** | **4** | **0.75** | **30** | **0.8823** | **0.0217** | **0.8691** |

**Best config: 92 (unfreeze=4, LLRD=0.75, 30ep) → 88.23%**

**Key findings:**
- **unfreeze=4 significantly beats unfreeze=2** (+1.67pp). This is unlike SigLIP2 (peaks at unfreeze=2) but mirrors DINOv3 (also peaks at unfreeze=4). Both DINOv3 and CLIP share the ViT-L architecture — the pattern suggests image-text contrastive pretraining (like DINO distillation) produces representations that tolerate deeper fine-tuning than discriminative-only pretraining.
- **CLIP-ViT 50ep (cfg 93) regressed to 87.40%** — contrary to DINOv3's trajectory where 50ep added +1.4pp. For CLIP-ViT, extended training overshoots with only ~860 training images at 30ep having already converged the top 4 unfrozen layers. The ceiling for this setting is at 30ep; 50ep likely induces overfitting.
- **CLIP-ViT best (88.23%) < DINOv3 best (93.79%) by −5.6pts.** Despite identical architecture (ViT-L), DINOv3's self-supervised distillation from a ViT-7B teacher on 1.689B images produces richer vision features than CLIP's image-text contrastive objective — the visual features are less adapted to fine-grained recognition.
- **Per-class routing potential:** CLIP-ViT leads SigLIP2 on specific classes (see section 5.5.38). For classes 52 and 86, CLIP-92 outperforms both SigLIP2 and DINOv3, making it a useful secondary for targeted class routing even at lower overall accuracy.

#### 5.5.23 DINOv2-Large + class weights and augmentation (configs 94–95)

| config | model | epochs | class_weights | RandAug | acc | Δ vs 80 |
|---|---|---|---|---|---|---|
| 80 (ref) | dinov2_large | 50 | no | no | 0.8739 | — |
| 94 | dinov2_large | 50 | **yes** | no | 0.8526 | **−2.13%** |
| 95 | dinov2_large | 50 | **yes** | **yes** | 0.8665 | **−0.74%** |

Class weights hurt DINOv2-Large by 2.13 pts, confirming the pattern seen in SigLIP-Base and ConvNeXt. Adding RandAugment partially offsets the damage (−0.74% net), but the combination still underperforms the clean baseline. **Class weights are disabled in the pipeline for all models.** The finding is consistent: inverse-frequency weighting distorts gradient balance in ways that outweigh any benefit from over-representing rare classes at this dataset scale.

#### 5.5.24 SigLIP2-SO400M-384 — initial results (config 96, preliminary)

Config 96 (conservative baseline: unfreeze=2, lr=1e-4, LLRD=0.8, 10ep, linear) running on Kaggle T4. Preliminary results (4/5 folds complete):

| fold | acc | peak epoch |
|---|---|---|
| 1 | 0.9167 | ep7 (0.9306) |
| 2 | **0.9676** | ep8 |
| 3 | 0.9028 | ep5 (0.9167) |
| 4 | 0.9352 | ep6 (0.9352) |
| 5 | _pending_ | ep8: 0.9581 |

Preliminary 4-fold mean: **~93.1%** — a +5.7 pt jump over DINOv2-Large (87.39%). SigLIP2-SO400M is a 400M-param ViT-So400M pretrained on image-text pairs at 384px. The gap is decisive: DINOv2-Large, SigLIP-Base, and CLIP-ViT are all outclassed.

**Key observation:** fold 1 peaked at ep7 (93.06%) but the final epoch (ep10) used the ep10 weights (91.67%) — 1.39% left on the table. Same regression visible in fold 3 (ep5=91.67%, ep10=90.28%). This motivated the best-epoch recovery fix below.

#### 5.5.25 Best-epoch in-memory recovery

All previous configs used `load_best_model_at_end=False` (eval_strategy="no"), meaning the last epoch's weights were always used for inference and ensemble, regardless of whether an earlier epoch was better. Across DINOv2-Large and SigLIP2 we consistently observe 1–3% accuracy left on the table when the model peaks before the final epoch.

**Fix:** `EpochAccuracyCallback` in `utils.py` now tracks `_best_val_acc` and `_best_state` (CPU copy of best state dict) across epochs. On `on_train_end`, if the final epoch is not the best, the best weights are restored in-memory. No disk writes, no double eval, no change to `eval_strategy`.

The epoch log now marks new bests with `✓best`. This fix is applied to all future runs (search.py and pipeline.py). Expected gain: +1–2% for SigLIP2 at 10–15ep.

#### 5.5.26 SigLIP2-SO400M unfreeze depth and epoch sweep (configs 103–107, local RTX 5070 Ti)

Five configs exploring unfreeze depth (2–8 blocks of 27), epochs (15–20), and LR (1e-4 vs 5e-5) with LLRD=0.8 or 0.75. All use best-epoch recovery.

| config | unfreeze | lr | LLRD | epochs | acc | acc_std | f1 |
|--------|----------|----|------|--------|-----|---------|-----|
| 103 | 4 | 1e-4 | 0.8 | 15 | 0.9481 | 0.0080 | 0.9358 |
| **104** | **2** | **1e-4** | **0.8** | **20** | **0.9500** | **0.0169** | **0.9396** |
| 105 | 6 | 1e-4 | 0.8 | 20 | 0.9435 | 0.0107 | 0.9315 |
| 106 | 8 | 1e-4 | 0.75 | 15 | 0.9444 | 0.0077 | 0.9330 |
| 107 | 6 | 5e-5 | 0.8 | 15 | 0.9444 | 0.0192 | 0.9325 |

**Key takeaways:**
- **Best config: 104 (unfreeze=2, 20ep) at 95.00%.** The shallowest unfreeze wins again — the same pattern as DINOv2 (unfreeze=2 optimal). SigLIP2's pretrained representations are highly transferable; adapting only the top 2 layers is sufficient.
- **Deeper unfreeze consistently hurts:** unfreeze=4 → 94.81%, unfreeze=6 → 94.35%/94.44%, unfreeze=8 → 94.44%. The range is compressed (94.35–95.00%), suggesting that at this dataset scale, unfreeze depth matters less than for DINOv2, but the shallow setting still wins.
- **Longer epochs help:** config 103 (unfreeze=4, 15ep) vs config 104 (unfreeze=2, 20ep) — the 20ep run gains from both factors simultaneously. Config 105 (unfreeze=6, 20ep) scores 94.35%, showing that 20ep alone cannot compensate for over-unfreezing.
- **LLRD=0.75 (config 106) offers no advantage** over LLRD=0.8 and introduces higher variance risk with 8 exposed layers. LLRD=0.8 is the right setting for SigLIP2.
- **Lower LR (5e-5, config 107) matches 1e-4 in mean (94.44%) but triples the variance** (±0.0192 vs ±0.0077). 1e-4 with LLRD=0.8 is more stable.
- **Stability vs. mean trade-off:** config 106 has the tightest std (±0.0077) but lower mean. Config 104 has the highest mean but also the highest std (±0.0169 — one fold hit 97.22%, one hit 92.13%). For a final submission, config 104's settings (unfreeze=2, 20ep, LLRD=0.8) are used.
- **`models/__init__.py` updated:** siglip2_so400m → unfreeze=2, num_epochs=20, llrd_factor=0.8.

#### 5.5.27 Multi-model ensemble + pseudo-labeling (overnight pipeline: DINOv2-Base + DINOv2-Large + SigLIP2-SO400M)

Full pipeline run combining all three best-config models with class-routed ensemble and pseudo-label round 2.

**Round 1 — 5-fold CV:**

| Model | Val Acc | Std | F1 | Train time |
|---|---|---|---|---|
| DINOv2-Base | 0.8499 | ±0.0191 | 0.8287 | 6.9 min |
| DINOv2-Large | 0.8758 | ±0.0268 | 0.8553 | 55.3 min |
| SigLIP2-SO400M | **0.9481** | ±0.0122 | 0.9399 | 59.0 min |
| Class-routed ensemble | 0.9370 | ±0.0086 | 0.9267 | — |

**Round 2 — pseudo-labeling (threshold=0.97, blended ensemble source):**

| Model | R1 acc | R2 acc | Delta |
|---|---|---|---|
| DINOv2-Base | 0.8499 | 0.8471 | −0.28% |
| DINOv2-Large | 0.8758 | 0.8786 | +0.28% |
| SigLIP2-SO400M | 0.9481 | 0.9481 | **0.00%** |

**Key finding 1 — blended pseudo-labels are useless for SigLIP2:**
Using an average of all three model groups (DINOv2 ×5 + DINOv2-Large ×5 + SigLIP2 ×5) to generate pseudo-labels produced zero improvement for SigLIP2 and −0.28% for DINOv2. Root cause: DINOv2's weaker predictions (85%) dilute SigLIP2's confident softmax scores, causing fewer test images to pass the 0.97 threshold and injecting label noise into round 2 training.

**Key finding 2 — class-routed ensemble hurts when mixing models of very different strength:**
The routed ensemble (93.70%) is **−1.11% below SigLIP2 alone (94.81%)**. Class routing distributes weight by per-class validation F1, which means DINOv2 "wins" on the 36 easy classes where all models score F1=1.0 — adding its predictions to SigLIP2's on those classes creates noise without benefit. The ensemble only genuinely outperforms all individuals on 9/100 classes.

**Class leadership breakdown:**
- SigLIP2 leads: **45/100** classes (all hard classes, most medium)
- DINOv2 leads: 36/100 (mostly easy, F1=1.0 for all models — ties counted)
- DINOv2-Large leads: 10/100
- Routed ensemble outperforms all: 9/100 (classes 88, 64, 87, 61, 62, 54, 85, 74, 84)
- Classes with spread >0.10 where routing helps most: 63 only had 9/100

**Fix applied for next run (`pipeline.py`):** Pseudo-labels are now generated exclusively from the highest R1 accuracy model (auto-selected via `max(results, key=acc)`), which is SigLIP2. DINOv2 and DINOv2-Large receive SigLIP2-quality pseudo-labels — true knowledge distillation. SigLIP2 gets self-distillation from its own highest-confidence predictions only.

**Fix applied for next run (`config.py`):** `USE_UPSAMPLE_BALANCE=True` — rare classes augmented to minimum 20 images per training fold.

**Submission files from this run:**
- `submission_siglip2_so400m.csv` — SigLIP2 R1 (best CV: 94.81%)
- `submission_pseudo_siglip2_so400m.csv` — SigLIP2 R2 (same: 94.81%)
- `submission_routed.csv` — class-routed R1 ensemble (93.70% CV — do NOT submit)

#### 5.5.28 SigLIP2-SO400M solo with self-distillation pseudo-labeling (run_004c)

SigLIP2 trained solo (no DINOv2 mixing) to test whether self-distillation pseudo-labels outperform blended ensemble pseudo-labels. Key change from 5.5.27: pseudo-labels generated exclusively from SigLIP2's own highest-confidence predictions rather than an ensemble average.

| Round | Val Acc | Notes |
|---|---|---|
| R1 | 0.9518 | SigLIP2 solo (unfreeze=2, 20ep) |
| **R2** | **0.9527** | **+0.09% from self-distillation** |

**Kaggle: 95.454%** (5th place, tied with 3rd and 4th).

Self-distillation works where blended ensemble pseudo-labels failed (zero gain in 5.5.27). The mechanism: DINOv2's weaker predictions previously diluted SigLIP2's softmax scores below the 0.97 confidence threshold. With SigLIP2 generating its own pseudo-labels, more test images pass the threshold and the injected signal is high-quality. The +0.09% CV gain understates the test benefit (95.454% vs prior 94.81% baseline on Kaggle = +0.644% test improvement).

#### 5.5.29 Hard class routing + upsample balance (run_005a — DINOv2-Large + SigLIP2)

Testing whether hard class routing with log-prob z-score normalisation improves over SigLIP2 solo when both models use `USE_UPSAMPLE_BALANCE=True` (min 20 images per class per training fold via upsampling). R2 pseudo-labeling included but crashed (SigLIP2 R2 fold 4 OOM — same location as run_004).

**R1 results (5-fold CV):**

| Model | Val Acc | Notes |
|---|---|---|
| DINOv2-Large | 0.8971 | New pipeline best; +2.32 pts over search cfg 80 (87.39%) due to upsample balance |
| SigLIP2-SO400M | 0.9518 | Consistent with run_004c R1 |
| **Hard routing** | **0.9509** | **−0.09% vs SigLIP2 solo** |

**R2 crash:** SigLIP2 R2 fold 4 OOM — cumulative RAM pressure from the larger pseudo-label dataset (1997 vs 1079 images) by the 4th fold. Fold models freed; no R2 submission from this run.

Key findings:
- **Upsample balance helps DINOv2-Large significantly** (87.39% → 89.71%, +2.32 pts). More balanced per-class training improves coverage for rare classes. Effect likely to help SigLIP2 less due to its higher starting accuracy.
- **Hard routing is essentially neutral** (−0.09% vs SigLIP2 solo). With only 2 models and SigLIP2 strongly dominant, routing cannot improve the mean. The 9 hard classes routed to DINOv2-Large may improve on the actual test set (unverifiable from CV).
- **SigLIP2 solo remains the best CV strategy.** run_004c's R2 submission (95.454% Kaggle) is the current best.

#### 5.5.30 SigLIP2-SO400M-512 solo validation (run_005)

Testing the 512px resolution variant (`google/siglip2-so400m-patch16-512`) with competitor-inspired settings (unfreeze=6, 15ep, cosine, batch=8). Note: these were suboptimal hyperparameters — unfreeze=6 is already known to hurt 384px SigLIP2 by −0.65%.

| Config | Resolution | Unfreeze | Epochs | Val Acc | Δ vs 384px best (cfg 104) |
|---|---|---|---|---|---|
| 104 (ref) | 384px | 2 | 20 | 0.9500 | — |
| run_005 | **512px** | 6 | 15 | 0.9453 | **−0.47%** |

Higher resolution is worse at these settings. Part of the regression is attributable to suboptimal hyperparameters (unfreeze=6 vs optimal unfreeze=2). Config 112 (512px, unfreeze=2, 20ep) confirmed this in run_008: **94.72%** — an improvement over suboptimal settings (+0.19%) but still −0.55% below the 384px best (95.27%). See section 5.5.32 for the full 512px search.

#### 5.5.31 DINOv3 ViT-L/16 — model added, search staged (configs 113–119)

DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) integrated as a new backbone. Model is access-gated on HuggingFace — `.env` HF_TOKEN authentication loader added to both `pipeline.py` and `search.py`.

**Architecture:**
- 300M parameters (same as DINOv2-Large, ViT-L/16)
- Pretrained on LVD-1689M (1.689B images — 12× DINOv2-Large's 142M dataset)
- Distilled from a ViT-7B teacher; no classification head (pretrain-only checkpoint)
- Layer naming: `model.layer.{i}.` (vs DINOv2's `encoder.layer.{i}.`); `utils.py` patched accordingly
- Uses `AutoModel` with linear head on `pooler_output`
- Validated: 24 transformer layers detected; 25.3M/303.2M trainable params at unfreeze=2

**Results (configs 113–119, all complete):**

| Config | Unfreeze | LR | Epochs | Acc | Std | F1 |
|---|---|---|---|---|---|---|
| 113 | 2 | 1e-4 | 20 | 0.9221 | 0.0130 | 0.9134 |
| 114 | 2 | 5e-5 | 20 | 0.8888 | 0.0206 | 0.8734 |
| 115 | 4 | 1e-4 | 20 | 0.9259 | 0.0066 | 0.9149 |
| 116 | 6 | 1e-4 | 20 | 0.9286 | 0.0081 | 0.9201 |
| 117 | 2 | 1e-4 | 50 | 0.9360 | 0.0119 | 0.9277 |
| **118** | **4** | **1e-4** | **50** | **0.9379** | **0.0119** | **0.9303** |
| 119 | 4 | 5e-5 | 50 | 0.9333 | 0.0109 | 0.9259 |

**Best config: 118 (unfreeze=4, lr=1e-4, 50ep) → 93.79%**

**Key findings:**
- **DINOv3 tolerates deeper unfreezing better than any prior model.** At 20ep, accuracy increases monotonically with unfreeze depth: unfreeze=2 → 92.21%, unfreeze=4 → 92.59%, unfreeze=6 → 92.86%. SigLIP2 peaked at unfreeze=2 and degraded with depth; DINOv2-Large also peaked at unfreeze=2. DINOv3's richer pretraining (1.689B images, ViT-7B distillation) makes the weights more resilient to gradient updates across more layers.
- **50ep adds +1.4–1.6% over 20ep**, consistent with DINOv2-Large's non-monotonic epoch trend. Config 117 (unfreeze=2, 50ep): +1.39% over cfg 113 (unfreeze=2, 20ep).
- **lr=5e-5 is harmful at 20ep (−3.3%)** but mostly recovers at 50ep (−0.46% vs lr=1e-4). More epochs compensate for the slower convergence.
- **DINOv3 best (93.79%) < SigLIP2 best (95.27%) by 1.48 pts.** DINOv3 is a strong pure-vision backbone but cannot close the gap to SigLIP2's image-text pretraining on this 100-class task.
- **Pipeline value:** DINOv3 at 93.79% offers complementarity to SigLIP2 (both ViT-L architecture, different pretraining). Hard routing with DINOv3 as the secondary model could improve on the 9 hard classes where DINOv2-Large currently leads.

#### 5.5.32 SigLIP2-SO400M-512 hyperparameter search (run_008, configs 110–112)

Systematic search over unfreeze depth and epoch count for the 512px variant, all with cosine scheduler, batch=8, lr=1e-4, LLRD=0.8.

| Config | Unfreeze | Epochs | Acc | Std | F1 |
|---|---|---|---|---|---|
| 110 | 6 | 20 | 0.9407 | 0.0106 | 0.9295 |
| 111 | 6 | 30 | 0.9370 | 0.0112 | 0.9237 |
| **112** | **2** | **20** | **0.9472** | **0.0139** | **0.9352** |

**Best config: 112 (unfreeze=2, 20ep) → 94.72%**

**Key findings:**
- **unfreeze=2 is optimal at 512px**, identical to the 384px finding. Deeper unfreezing (unfreeze=6, cfg 110) loses −0.65%. The larger token grid (1024 vs 576 patches) does not require additional backbone adaptation — shallow fine-tuning of the top 2 layers is sufficient.
- **Longer training at unfreeze=6 regresses.** Config 111 (30ep, unfreeze=6) = 93.70%, −0.37% vs cfg 110 (20ep, unfreeze=6). More epochs amplify overfitting when the backbone is over-unfrozen at higher resolution.
- **512px best (94.72%) < 384px best (95.27%) by −0.55%.** Native resolution is confirmed optimal for SigLIP2. The positional embedding interpolation from 384px pretraining to 512px inference degrades patch-level alignment, and ~1079 training images cannot compensate.
- **Ensemble potential:** Despite the regression, 94.72% at 512px is a viable second model for a soft-weighted ensemble with the 384px SigLIP2 (competitor achieved 97% from a similar 96.4%+95.4% pairing). Error patterns differ by resolution — the ensemble may recover some of the 5.3% test errors that neither resolution alone handles.

#### 5.5.33 DINOv3 pipeline run + SigLIP2-384 × DINOv3 ensemble grid search (run_010)


Full 5-fold CV pipeline run of DINOv3 (cfg 118 settings: unfreeze=4, lr=1e-4, LLRD=0.75, 50ep) to save probability arrays for offline ensemble analysis. After run_010, `ensemble_grid.py` was run to find the optimal soft weight between SigLIP2-384 probs (from run_009) and DINOv3 probs.

**DINOv3 pipeline result:** val acc = **93.70%**, F1 = 0.9349.

**Ensemble grid search — `(1−w)×SigLIP2-384 + w×DINOv3`:**

| w (DINOv3) | CV Acc | F1 |
|---|---|---|
| 0.00 (SigLIP2 solo) | 94.90% | 0.9476 |
| 0.05 | 94.90% | 0.9476 |
| 0.10 | 94.81% | 0.9467 |
| … | … | … |
| 0.60 | 94.81% | 0.9455 |
| **0.65** | **95.00%** | **0.9477** |
| 0.70 | 94.53% | 0.9427 |
| … | … | … |
| 1.00 (DINOv3 solo) | 93.70% | 0.9349 |

**Best: w=0.65 → `0.35×SigLIP2-384 + 0.65×DINOv3` = 95.00% CV (+0.10% over SigLIP2 solo)**

**Key findings:**

- **Optimal w=0.65 favors DINOv3** despite it being the weaker model (93.70% vs 94.90%). This is the complementarity effect: DINOv3's self-supervised patch-distillation features encode different visual patterns than SigLIP2's image-text contrastive features, so the two models' errors are partially independent.
- **Gain is modest (+0.10%)** because the 1.2-point accuracy gap limits decorrelation. When one model is significantly weaker, P(both wrong on the same example) is still high — the ensemble can only recover errors where exactly one model fails, but that set is small at 94.90% vs 93.70%.
- **Same-backbone ensemble (SigLIP2-384 + SigLIP2-512, w=0.10) got +0.19%** — slightly better than the cross-architecture ensemble (+0.10%) despite the same-backbone errors being more correlated. The 384+512 pairing benefits from resolution diversity even though the feature distributions are highly aligned.
- **Ensemble pseudo-labels for R2:** The 95.00% CV ensemble produces higher-quality pseudo-labels than either model alone. Running R2 with `PSEUDO_LABEL_ENSEMBLE_WEIGHTS={"siglip2_so400m": 0.35, "dinov3": 0.65}` is the next step (run_011).

#### 5.5.34 Ensemble pseudo-label R2 — SigLIP2 solo (run_011b)

SigLIP2 retrained in R2 using pseudo-labels generated from the soft-weighted ensemble `0.35×SigLIP2-384 + 0.65×DINOv3` (the best CV combination found in section 5.5.33, 95.00%). Pseudo-label source loaded from saved `probs/test_probs_*.npy` files without retraining DINOv3.

| Round | CV Acc | F1 | Notes |
|---|---|---|---|
| R1 | 95.00% | 0.9429 | Consistent with run_009 |
| **R2** | **95.09%** | **0.9429** | +0.09% from ensemble pseudo-labels |

**Key findings:**

- **Ensemble pseudo-labels give +0.09%** — identical to run_004c self-distillation gain (+0.09%). The higher-quality pseudo-label source (95.00% ensemble vs 94.90% solo) does not improve the R2 gain.
- **The limiting factor is the number of confident test images, not label quality.** At threshold=0.97, the ensemble accepts ~N pseudo-labels; the quality improvement from 94.90%→95.00% source adds negligibly to that count or their individual accuracy.
- **SigLIP2 R1 at 95.00% is rock-stable** across all runs (run_009: 94.90%, run_011b: 95.00%, run_004c R1: 95.18%). Variance is within run-to-run noise from stratified fold splitting.
- **Hard classes persist:** class 63 (F1=0.43), class 86 (0.48), class 88 (0.45) remain below 0.50 in R2. Pseudo-labeling cannot help hard classes where the model is uncertain — test images from those classes don't reach the 0.97 confidence threshold.
- **run_011 (SigLIP2+DINOv3 joint) crashed** at SigLIP2 fold 3 due to Windows memory fragmentation after DINOv3's 78-min prior run. Run solo (run_011b) completed cleanly — consistent with run_009 confirming SigLIP2 solo is safe within the available RAM budget.

---

#### 5.5.35 Clean val_probs regeneration + confidence-gated ensemble (run_012)

Run_012 regenerated clean `probs/val_probs_siglip2_so400m.npy` after the R2 overwrite bug corrupted the previous file. A 2D confidence-gated ensemble grid search then tested all combinations of `w ∈ [0.0, 0.5]` and `conf_threshold ∈ [0.80, 1.00]`.

**SigLIP2 R1 results (run_012):**

| CV Acc | F1 | Notes |
|---|---|---|
| 94.90% | 0.9407 | Consistent with run_009; R1 only (no pseudo-labels) |

**Confusion pairs (pooled 5-fold val):**

| Pair | Errors | Rate |
|---|---|---|
| 63 <-> 70 | 8 | 0.42 |
| 76 <-> 78 | 7 | 0.30 |
| 86 <-> 88 | 7 | 0.32 |
| 86 <-> 87 | 6 | 0.26 |
| 68 <-> 74 | 4 | 0.16 |

**2D confidence-gated ensemble grid search (SigLIP2-384 + DINOv3):**

| w (DINOv3) | conf threshold | CV Acc | gated |
|---|---|---|---|
| 0.00 | any | **94.90%** | — |
| 0.05–0.10 | any | 94.90% | flat, no change |
| 0.15 | ≥0.95 | 94.90% | 870/1079 gated in |
| 0.15–0.50 | <0.95 | 94.81–94.53% | hurt |

Best: `w=0.00` at all confidence levels — DINOv3 never improves over SigLIP2 solo regardless of gating threshold.

**Key findings:**

- **DINOv3 ensemble path definitively closed.** The confidence gate (restricting DINOv3 to high-confidence predictions) does not rescue the combination. Even w=0.05–0.10 is completely neutral, and any w≥0.15 below conf=0.95 hurts.
- **run_010's w=0.65 → 95.00% was model-weight-specific overfitting.** That result used run_009's specific SigLIP2 model weights; DINOv3 happened to correct run_009's errors. With run_012's different (though same-accuracy) weights, the same DINOv3 probs offer no complementary signal.
- **The 91.818% Kaggle score is explained.** Val-overfitted ensemble weights (w=0.65 selected on 1079 samples ≈ 10 per class) hurt on the 1036-image test set where DINOv3's systematic fine-grained errors dominated.
- **New confusion pair discovered:** class 63 ↔ 70 (rate=0.42) is the most confused pair — higher than the expected 76/78 and 86/87/88 clusters. Class 70 was not previously flagged as hard.

**SigLIP2-384 + SigLIP2-512 ensemble re-evaluation with clean probs:**

With fresh `val_probs_siglip2_so400m.npy` (run_012), the 384+512 ensemble grid was re-run using the existing `probs/val_probs_siglip2_so400m_512.npy` (run_009). Optimal weight shifted from w=0.10 (run_009 probs, 95.09%) to **w=0.20 (run_012 probs, 95.00%)**. Submitted to Kaggle: **94.545%** (−0.455 pts CV→test gap). The same-backbone ensemble also exhibits val-overfitting on 1079 samples — resolution diversity does not provide enough independent signal to generalize.

---

#### 5.5.36 SigLIP2-SO400M-384 self-distillation — run_013 (SEED=42) and run_015 (SEED=0)

Two fresh full-pipeline runs to produce independently-trained models for a different-seed ensemble. Both use identical config (SigLIP2-SO400M-384, unfreeze=2, lr=5e-5, LLRD=0.75, 10ep, TTA, pseudo-label threshold=0.97) but differ in random seed, giving independent 5-fold stratified splits.

| Run | Seed | R1 acc | R2 acc | R2 Δ | Pseudo-labels accepted |
|---|---|---|---|---|---|
| run_013 | 42 | 94.90% | 94.90% | 0.00% | 801/1036 |
| run_015 | 0 | 94.90% | 94.62% | **−0.28%** | 801/1036 |

**Key finding — pseudo-label gain is not reliable across seeds.** Across all SigLIP2 R2 runs: run_004c +0.09%, run_011b +0.09%, run_013 0.00%, run_015 −0.28%. The variance is larger than the mean gain. At threshold=0.97, both runs accept exactly 801/1036 pseudo-labels — confirming identical confidence coverage. The R2 CV accuracy fluctuates within ±0.3% of R1 from run to run, suggesting the gain (or loss) is driven by which specific test images get injected into which folds, not by any systematic signal.

**Submissions:** , 

#### 5.5.37 Different-seed ensemble grid search — run_016 (SEED=42 × SEED=0)

 run on  (SEED=42 R1 OOF) and  (SEED=0 R1 OOF). Test predictions blended from  and  (R2 test probs for each seed).

**Grid search results ():**

| w (r015) | val acc | f1 |
|---|---|---|
| 0.00 (r013 solo) | 94.90% | 0.9476 |
| 0.05–0.20 | 94.81% | — |
| 0.25 | 95.09% | 0.9494 |
| **0.30** | **95.18%** | **0.9502** |
| 0.35 | 95.09% | 0.9492 |
| 0.50 | 94.62% | 0.9448 |
| 1.00 (r015 solo) | 94.90% | 0.9480 |

**Best:  → val acc 95.18%, F1 = 0.9502 (+0.28pp over solo)**

**Submission:** 

**Key findings:**
- **+0.28pp is 3× larger than typical val-overfitting noise.** Prior val-overfit ensembles (SigLIP2-384+512, SigLIP2+DINOv3) showed +0.10–0.19% CV then hurt on Kaggle. A +0.28pp gain from two same-architecture, same-accuracy models is larger than expected from correlated errors alone, suggesting the different fold splits introduce genuine error diversity.
- **Asymmetric weight (70/30) favors SEED=42.** This is partly by chance — SEED=42 happened to produce slightly better OOF predictions. The 30% weight from SEED=0 still contributes complementary signal.
- **Kaggle result: 95.454%** — tied with the solo best (run_004c). The +0.28pp val gain was val-overfitting. Different-seed fold splits produce genuinely different OOF predictions, but the test-set errors are highly correlated (same pretrained weights, same architecture, same training recipe). **All ensemble paths are now exhausted:** same-backbone different-resolution (+0.19% CV → −0.91% Kaggle), cross-architecture (+0.10% CV → −3.6% Kaggle), same-architecture different-seed (+0.28% CV → 0% test gain). The 1079-sample val set cannot identify which test errors are independent regardless of ensemble approach.

#### 5.5.38 Per-class soft routing ensemble — `class_router.py`

After all global ensemble paths were exhausted (same-backbone different-resolution, cross-architecture, same-backbone different-seed all showing val-overfitting patterns), a per-class routing approach was developed. The key observation: global ensembles fail because the weaker model's systematic errors dominate. But at the individual class level, a weaker model can decisively outperform SigLIP2 on 5–15 specific hard classes.

**Design:**
- SigLIP2 (siglip2_so400m_r013) is the backbone for all 100 classes.
- For each class where a secondary model exceeds the backbone's per-class F1 by ≥ delta, the backbone's probability column for that class is replaced by the secondary's column.
- Multiple secondaries are supported: each class is routed to whichever secondary leads by the largest margin.
- Routing analysis requires only saved `.npy` probability arrays — no retraining needed.

**Routing analysis — SigLIP2 vs DINOv3 vs CLIP-93 (delta ≥ 0.05):**

Note: cfg 92 (88.23%) has no saved `.npy` arrays (run before prob-saving was added). Cfg 93 (87.40%, 50ep) probs are used as the CLIP secondary. Cfg 92 would likely show higher per-class F1 on the specific CLIP-leading classes.

| Class | SigLIP2 F1 | DINOv3 F1 | CLIP-93 F1 | Winner | Margin |
|---|---|---|---|---|---|
| 87 | 0.600 | **0.762** | 0.567 | DINOv3 | +0.162 |
| 52 | 0.800 | 0.875 | **0.960** | CLIP-93 | +0.160 |
| 84 | 0.842 | **1.000** | 0.900 | DINOv3 | +0.158 |
| 86 | 0.696 | 0.846 | **0.850** | CLIP-93 | +0.154 |
| 89 | 0.842 | **0.947** | 0.760 | DINOv3 | +0.105 |
| 85 | 0.480 | **0.583** | 0.381 | DINOv3 | +0.103 |
| 51 | 0.842 | 0.900 | **0.933** | CLIP-93 | +0.091 |
| 1 | 0.909 | **1.000** | 1.000 | DINOv3 | +0.091 |
| 93 | 0.909 | **1.000** | 0.933 | DINOv3 | +0.091 |
| 94 | 0.909 | **1.000** | 0.665 | DINOv3 | +0.091 |
| 3 | 0.947 | **1.000** | 0.933 | DINOv3 | +0.053 |
| 6 | 0.947 | **1.000** | 1.000 | DINOv3 | +0.053 |
| 2 | 0.909 | 0.952 | **0.960** | CLIP-93 | +0.051 |

**87 classes stay with SigLIP2.** 9 route to DINOv3, 4 to CLIP-93.

**Three routing strategies compared (SigLIP2 backbone + DINOv3 secondary, 12 routed classes):**

| Strategy | Val Acc | Delta vs backbone |
|---|---|---|
| Backbone solo | 94.90% | — |
| **Hard (w=1.0)** | **96.11%** | **+1.21pp** |
| Soft-global (single w grid-searched) | 96.11% | +1.21pp (best w=1.00) |
| Soft-per-class (independent w per class) | 95.83% | +0.93pp |

**Key findings:**
- Hard routing and soft-global routing converge to w=1.00 — full replacement is optimal, meaning the secondary model's column is more informative than any blend for these classes.
- Soft-per-class underperforms hard routing (−0.28pp): with ~10 val samples per class, per-class weight search overfits the noise. Classes 1, 6, and 51 optimise to w=0.0 despite positive per-class F1 margins, simply because the margin is within variance of those small per-class sample sizes.
- **Hard routing is the correct strategy** — simpler, no overfitting risk, equal or better results.

**Val gain caveat:** All prior global ensembles showed val gains of +0.10–0.28pp that failed to hold on the 1036-image test set. The +1.21pp val gain here is substantially larger (4–12× prior gains), but the underlying mechanism is different — per-class routing should be less susceptible to the specific-weight-overfitting pattern that caused prior ensemble failures, since each routed class has a consistent, large per-class margin rather than a global weight selected to minimise error on 1079 samples.

**Initial submission (run_017 — 3 classes only):** Routes classes 84, 86, 87 to DINOv3 (the top 3 by margin, all >0.15). Val acc = 95.37% (+0.47pp). Pending Kaggle submission.

**Infrastructure — `class_router.py`:** Implements full routing workflow: load any combination of backbone + secondary .npy files, compute per-class F1, auto-identify routing candidates above delta threshold, run hard/soft-global routing, generate submission CSV. Supports multiple secondaries simultaneously with `--secondaries model1 model2 ...` and `--analyze` flag for analysis-only mode (no submission). `search.py` now saves `val_probs_{model}_u{unfreeze}_ep{epochs}_llrd{llrd}_cfg{N}.npy` and `test_probs_...npy` after every config, enabling immediate routing analysis without a separate pipeline run.

### 5.6 Overall Best Results Summary

| Model | Best CV acc | Kaggle | Config | Key settings |
|---|---|---|---|---|
| ResNet-50 | 6.79% | — | baseline | — |
| ResNet-101 | 14.81% | — | baseline | — |
| ConvNeXt-Base | 76.27% | — | 11 | unfreeze=4, lr=3e-4 |
| ConvNeXt-Large | 77.85% | — | pipeline | unfreeze=4, prog-unfreeze |
| ViT-B/16 | 74.42% | — | 54 | unfreeze=4, LLRD=0.85, +CJ |
| SigLIP-Base | 84.61% | — | 90 | unfreeze=2, lr=1e-4, 20ep |
| CLIP-ViT-Large/14 (cfg 91) | 86.56% | — | 91 | unfreeze=2, LLRD=0.75, 30ep |
| CLIP-ViT-Large/14 (cfg 92) | 88.23% | — | 92 | unfreeze=4, LLRD=0.75, 30ep |
| CLIP-ViT-Large/14 (cfg 93) | 87.40% | — | 93 | unfreeze=4, LLRD=0.75, 50ep (regression vs 30ep) |
| CLIP-ViT-Large/14 (cfg 120) | — (killed at fold 3) | — | 120 | unfreeze=6, LLRD=0.65, 50ep |
| SigLIP2-SO400M-384 + upsample (cfg 121) | _running_ | — | 121 | unfreeze=2, lr=1e-4, LLRD=0.8, 20ep, upsample min=20 |
| DINOv2-Base | 85.54% | — | 64 | unfreeze=2, lr=1e-4, LLRD=0.85, 10ep |
| DINOv2-Large (search) | 87.39% | — | 80 | unfreeze=2, lr=1e-4, LLRD=0.75, 50ep |
| DINOv2-Large (pipeline, upsample) | 89.71% | — | run_005a | unfreeze=2, 50ep, upsample min=20 |
| DINOv3 ViT-L/16 (search) | 93.79% | — | 118 | unfreeze=4, lr=1e-4, 50ep |
| DINOv3 ViT-L/16 (pipeline) | 93.70% | — | run_010 | unfreeze=4, lr=1e-4, 50ep, upsample min=20 |
| SigLIP2-SO400M-512 (run_005) | 94.53% | — | run_005 | unfreeze=6, 15ep (suboptimal) |
| SigLIP2-SO400M-512 (cfg 112) | 94.72% | — | 112 | unfreeze=2, lr=1e-4, 20ep (optimal 512px) |
| Soft ensemble: SigLIP2-384 + SigLIP2-512 (w=0.20) | 95.00% | **94.545%** | run_012 | fresh probs re-eval; val-overfitted (+0.19→−0.45 gap) |
| SigLIP2-SO400M-384 (cfg 104) | 95.00% | — | 104 | unfreeze=2, lr=1e-4, LLRD=0.8, 20ep |
| SigLIP2-SO400M-384 R1 (run_004c) | 95.18% | — | run_004c | pipeline, unfreeze=2, 20ep |
| SigLIP2-SO400M-384 R1 (run_011b) | 95.00% | — | run_011b | pipeline, unfreeze=2, 20ep |
| Hard routing (run_005a R1) | 95.09% | — | run_005a | DINOv2-Large + SigLIP2, log-prob z-score |
| Soft ensemble: SigLIP2-384 + DINOv3 (w=0.65) | 95.00% | — | run_010 | 0.35×SigLIP2 + 0.65×DINOv3 (val-overfitted, 91.818% Kaggle) |
| SigLIP2-SO400M-384 R2 (run_011b) | 95.09% | — | run_011b | ensemble pseudo-label (0.35×SigLIP2 + 0.65×DINOv3) |
| SigLIP2-SO400M-384 R1 (run_012) | 94.90% | — | run_012 | clean val_probs run; conf-gated ensemble showed w=0 optimal |
| SigLIP2-SO400M-384 R2 | 95.27% | 95.454% | run_004c | self-distillation pseudo-label, 5th place |
| SigLIP2-SO400M-384 R1 (run_013, SEED=42) | 94.90% | — | run_013 | fresh SEED=42 run for ensemble diversity |
| SigLIP2-SO400M-384 R2 (run_013) | 94.90% | — | run_013 | self-distill, 0 gain vs R1 |
| SigLIP2-SO400M-384 R2 (run_015, SEED=0) | 94.62% | — | run_015 | self-distill, −0.28pp vs R1 |
| Different-seed ensemble (r013+r015, w=0.30) | 95.18% | **95.454%** | run_016 | 0.70×SEED=42 + 0.30×SEED=0; +0.28pp CV, 0 test gain (val-overfitted) |
| Per-class routing: SigLIP2 + DINOv3 (3 classes: 84, 86, 87) | 95.37% | _pending_ | run_017 | hard routing w=1.0; +0.47pp CV |
| Per-class routing: SigLIP2 + DINOv3 (12 classes, delta≥0.05) | 96.11% | _not submitted_ | — | hard routing; +1.21pp CV; val-overfit risk TBD |
| Per-class routing: SigLIP2 + DINOv3 + CLIP-93 (3 models) | _pending_ | — | run_018 | 3-model hard routing; cfg 93 probs available |
| SigLIP2 + upsample (cfg 121) solo routing backbone | _running_ | — | — | new backbone candidate for routing after cfg 121 finishes |

### 5.7 Pipeline Run — Class-Routed Ensemble (DINOv2-Base + DINOv2-Large)

Full 5-fold CV pipeline run with both DINOv2 models using their per-model best configs, followed by a class-routed weighted ensemble. Each model's submission CSV and the combined routed ensemble CSV were generated.

**Per-model best configs used:**
- DINOv2-Base: lr=1e-4, LLRD=0.85, unfreeze=2, 10 epochs (config 64 settings)
- DINOv2-Large: lr=1e-4, LLRD=0.75, unfreeze=2, 30 epochs (config 72 settings)

**5-fold CV results:**

| Model | Val Acc | F1 | Train Time |
|---|---|---|---|
| DINOv2-Base | 0.8434 ± — | 0.826 | 11.2 min |
| DINOv2-Large | 0.8647 ± — | 0.847 | 55.2 min |
| **Routed Ensemble** | **0.8684** | — | — |

**Class-routing analysis (51+34+15 = 100 classes):**
- DINOv2-Base leads on **51/100** classes
- DINOv2-Large leads on **34/100** classes
- Routed ensemble leads on **15/100** classes (outperforms both individuals)

The routed ensemble (+0.37% over Large alone) confirms that Base and Large are complementary: despite Large's higher overall accuracy, Base dominates more than half the classes individually. Class routing successfully leverages this complementarity.

**Submission files generated:**
- `submission_dinov2.csv` — DINOv2-Base single-model predictions
- `submission_dinov2_large.csv` — DINOv2-Large single-model predictions
- `submission_routed.csv` — Class-routed ensemble (best submission)

2. **Kaggle public leaderboard score.** Best submission: **95.454%** (5th place, tied with 3rd–4th). Source: SigLIP2-SO400M-384 R2 self-distillation pseudo-label run (run_004c).
3. **Qualitative error analysis.** The SigLIP2 pooled 5-fold confusion matrix (run_012) reveals two categories of errors: systematic pairwise confusion and hard isolated classes.

   **Top confusion pairs (pooled 5-fold val, threshold ≥4 mutual errors):**

   | Pair | Errors | Confusion rate | Likely cause |
   |---|---|---|---|
   | 63 ↔ 70 | 8 | 0.42 | Highest confusion rate; classes share strong visual similarity |
   | 86 ↔ 88 | 7 | 0.32 | Persistent pair; all models confused (F1<0.70) |
   | 76 ↔ 78 | 7 | 0.30 | Symmetric confusion; similar texture/color pattern |
   | 86 ↔ 87 | 6 | 0.26 | Both classes confusing with 86; possible subcategory overlap |
   | 68 ↔ 74 | 4 | 0.16 | Moderate confusion; both hard across architectures |

   Class 86 appears in two confusion pairs (86↔88 and 86↔87), confirming it as a "hub" of visual ambiguity — the model frequently mistakes it for one of two neighboring classes, consistent with per-class routing targeting it as a priority.

   **Hardest individual classes (SigLIP2 best-fold F1 < 0.60):**
   - Class 66: F1=0.00 across all 8 models and all configurations — uniquely hard. No model ever correctly classifies this class, even with TTA and pseudo-labeling. Likely insufficient training examples and/or high within-class visual variance.
   - Class 85: F1=0.48 (SigLIP2 OOF). DINOv3 reaches 0.58 — targeted by per-class routing.
   - Class 87: F1=0.60 (SigLIP2). DINOv3 reaches 0.76 — largest absolute routing margin (+0.162).
   - Class 63: F1=0.43 (SigLIP2 R2). Pair 63↔70 is the primary confusion source.

   **Pattern:** The hard classes and confusion pairs are stable across architectures (DINOv2, DINOv3, CLIP, SigLIP2 all struggle with 63, 66, 85, 86, 87, 88), suggesting image-level visual ambiguity rather than architecture-specific failure modes. Per-class routing can address 87, 85, 84, 86 (large secondary-model F1 advantages) but class 66's zero-F1 pattern across all models makes it unroutable — all secondary models also fail on it.

---

## 6 Discussion

1. **What worked best and why.** SigLIP2-SO400M-384 with 2 unfrozen layers, lr=1e-4, LLRD=0.8, 20 epochs, and self-distillation pseudo-labeling achieved **95.27% CV** and **95.454% Kaggle** — the final result across all 119 configs and pipeline runs. The dominant factor is pretraining quality: SigLIP2's 400M-param ViT-So400M backbone, pretrained on billions of image-text pairs at native 384px resolution, transfers to 100-class classification with minimal fine-tuning (only the top 2 of 27 layers need adapting). The pattern mirrors DINOv2: shallow unfreeze (2 blocks) consistently outperforms deeper unfreezing (unfreeze=6 loses −0.65%), and shorter training can overfit (20ep optimal, not longer). Self-distillation pseudo-labeling adds +0.09% CV (and +0.64% Kaggle test), while blended ensemble pseudo-labels produce zero gain — the stronger model must generate its own labels to avoid noise injection from weaker ensembles. DINOv2-Large at 50ep (87.39% search) remains the best DINOv2 result, but the 7.9 point gap to SigLIP2 confirms that architecture and pretraining dataset dominate hyperparameter tuning at this scale.

2. **Failure cases, overfitting/underfitting observations.** Class 66 scores F1=0 across all models — consistently the hardest class. DINOv2-base overfits past 10 epochs: 20ep → 0.8526 < 0.8554, 30ep → 0.8406 (using best config 64 params). DINOv2-Large does not show this: 30ep → 0.8684 > 20ep → 0.8564. ViT-B/16's steep LR sensitivity (2% frozen at 1e-5, 45% at 1e-4) reflects the CLS token struggling to adapt without backbone gradient flow; partial unfreezing with LLRD resolves it (+27 pts). The layer-count bug in `apply_freeze` (hardcoded 12 layers) made configs 57–58 invalid — DINOv2-Large has 24 layers, so unfreeze=2 was actually unfreezing layers 10–11 instead of 22–23.

3. **Limitations and next steps.**
   - **SigLIP2-SO400M-384 is the dominant backbone.** At 95.00% (search config 104) and 95.27% CV / 95.454% Kaggle (run_004c with pseudo-labels), it outperforms DINOv2-Large by 7.9 points. The gap reflects fundamentally better pretraining (image-text pairs at native 384px resolution vs self-supervised on ImageNet).
   - **Self-distillation pseudo-labeling gives +0.09% CV consistently.** run_004c (self-distillation, 94.90% source) and run_011b (ensemble pseudo-labels, 95.00% source) both produce exactly +0.09% R2 gain. The limiting factor is the number of test images that exceed the 0.97 confidence threshold, not the label accuracy. Zero gain when diluted by much weaker models (DINOv2, 5.5.27).
   - **Hard class routing with log-prob z-score normalisation is neutral at −0.09% CV** (5.5.29). With a dominant model (SigLIP2 at 95%) and one much weaker model (DINOv2-Large at 90%), routing cannot improve the mean — the weaker model "wins" only easy classes where both models already score F1=1.0. Routing may add test-set value on the 9 hard classes where DINOv2-Large leads by per-class val F1, but this is unverifiable from CV alone.
   - **Higher resolution consistently hurts.** SigLIP2-512: −0.47% at suboptimal settings (run_005), −0.55% even at optimal settings (cfg 112, unfreeze=2, 94.72%). DINOv2-518: −9.64% (configs 85–87). Token sequence expansion exceeds what ~1079 training images can support; positional embedding interpolation noise from 384px pretraining dominates. Native resolution is confirmed optimal across all tested models.
   - **Class weights ruled out.** Inverse-frequency weighting hurts all models (SigLIP-Base −0.75%, DINOv2-Large −2.13%). Distorts gradient balance more than it helps rare classes at this scale.
   - **Upsample balance (`USE_UPSAMPLE_BALANCE=True`, min=20)** helps DINOv2-Large significantly (+2.32 pts, 87.39% → 89.71%). Effect on SigLIP2 not yet isolated.
   - **DINOv3 best: 93.79% (cfg 118) / 93.70% (pipeline, run_010).** Strong but 1.48 pts below SigLIP2. Soft-weighted ensemble grid search (run_010) found `0.35×SigLIP2 + 0.65×DINOv3` = **95.00%** on val, but this was model-weight-specific overfitting: when re-evaluated with run_012's SigLIP2 probs, the optimal weight is **w=0** (DINOv3 never helps). The confidence-gated ensemble (restricting DINOv3 to high-confidence predictions) also gives w=0 as best across all thresholds. The 91.818% Kaggle score from the w=0.65 submission confirms DINOv3's val-overfitted weights hurt on the real test set. **DINOv3 ensemble is not a viable path forward.**
   - **CLIP-ViT-Large/14 (cfg 92): 88.23% at unfreeze=4, LLRD=0.75, 30ep.** Despite lower overall accuracy than DINOv3 (93.79%), CLIP-92 leads SigLIP2 on specific classes (52, 86) where DINOv3 does not — its image-text contrastive pretraining encodes different class discriminators. **50ep (cfg 93) regressed to 87.40%** — CLIP-ViT's 30ep recipe is already at its capacity; extended training overshoots. Best CLIP result remains cfg 92 (88.23%); cfg 93 probs are used for routing since cfg 92 has no saved `.npy` arrays.
   - **SigLIP2 + upsample balance (cfg 121) currently running.** DINOv2-Large gained +2.32pp from upsample balance; the analogous gain on SigLIP2 (already at 95%) is expected to be smaller but positive. If cfg 121 improves on the routing backbone, the run_018 3-model submission can use it.
   - **Per-class hard routing is the most promising remaining path.** With all global ensemble approaches exhausted (val-overfitting confirmed for same-backbone different-resolution, cross-architecture, and different-seed ensembles), per-class routing using `class_router.py` offers a different mechanism: SigLIP2 handles 87–88 classes, secondary models replace only specific columns where they have large, consistent per-class F1 advantages. Val gain of +1.21pp (12 classes, SigLIP2+DINOv3) and +0.47pp (3 classes, run_017) are substantially larger than prior ensemble gains, reducing the val-overfitting risk. The 3-model routing (SigLIP2 + DINOv3 + CLIP-93) is the next submission target after cfg 121 completes.
   - **Label-mixing and spatial augmentation ineffective.** Mixup (−2%), CutMix (−6%), RandAugment neutral-to-negative across all tested models. Disabled in the final pipeline.
   - **Kaggle submissions per config** available at `search_results/config_{i}_{model}.csv`.

---

## 7 Reproducibility

1. **Random seeds.** `SEED = 42` passed to `set_seed()` (HuggingFace), `train_test_split`, and `TrainingArguments`.
2. **Environment setup.**
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
3. **Training command.**
   ```
   python pipeline.py
   ```
