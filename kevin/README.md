# CSE 144 Final Project — Transfer-Learning Image Classifier (100 classes)

Fine-tunes diverse ImageNet-21k pretrained backbones into a 100-class classifier and
ensembles them (5-fold CV × multi-architecture × test-time augmentation) to maximize
held-out test accuracy on the [UCSC CSE 144 Spring 2026 Kaggle competition](https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project).

Data: **1,079** train images across **100** classes (4–41 images/class), predict **all
1,036** images in `Data/test/` (`0.jpg`–`1035.jpg`). `sample_submission.csv` is only a
format example (1000 rows) — Kaggle scoring requires the full test folder.

> ⚠️ **Label mapping (the #1 failure mode):** the folder name *is* the label
> (`train/7/...` → class `7`). We build an explicit `int(folder)` map and **assert
> `"0"→0 … "99"→99`** at load time. We never rely on `ImageFolder`'s alphabetical class
> order, which would scramble labels and produce ~random accuracy.

## Repository layout

```
src/
  data.py     # folder→label map (+assert), RGB cleaning, stratified k-fold, timm transforms, datasets
  model.py    # timm backbone factory + 100-way head (dropout), EMA (warmup), layer-wise-LR-decay groups
  train.py    # CLI: fine-tune ONE backbone across folds → per-fold raw+EMA ckpts + OOF logits + config
  predict.py  # CLI: load all ckpts, TTA + multi-fold/multi-arch softmax ensemble → validated submission.csv
  utils.py    # seeding/determinism, device select, metrics, logging, submission validator
config.yaml   # SEED + all hyperparameters (backbones, folds, aug, lr/llrd, mixup, ema, …)
model.ipynb   # thin Colab wrapper (GPU → Drive Data → train → predict → download)
requirements.txt
checkpoints/  # produced by train.py (per-backbone/per-fold ckpts + OOF + resolved config)
```

## Environment

| | Local (MPS) | Google Colab (primary) |
|---|---|---|
| Device | Apple M2 Pro, MPS | NVIDIA T4/A100, CUDA, AMP on |
| Python | 3.11.5 | Colab default |
| torch / torchvision | 2.1.2 / 0.16.2 (pinned in requirements.txt) | Colab preinstall (use as-is) |
| timm | 1.0.27 | `pip install timm==1.0.27` in notebook |

The code is device-agnostic (`cuda → mps → cpu`). AMP is auto-enabled only on CUDA.
EMA uses `foreach=False` for MPS compatibility. Use Colab GPU for full training runs.

## Google Colab (quick start)

1. **Runtime → Change runtime type → GPU.**
2. Upload `Data/` to Google Drive (keep `train/`, `test/`, `sample_submission.csv`).
3. Open `model.ipynb` in Colab (upload, or open from GitHub after push).
4. Edit in the notebook:
   - `REPO_URL` — your public GitHub clone URL (or `%cd` to a Drive copy of the whole repo and skip clone).
   - `DATA_ON_DRIVE` — path to `Data` on Drive (e.g. `/content/drive/MyDrive/CSE144/Data`).
   - `CKPT_DRIVE` — where to persist `checkpoints/` between sessions.
5. Run cells in order: install → mount Drive → train → predict → **download** `submission.csv`.
6. Submit at the [Kaggle competition page](https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project/submit) — file must have **1036 rows**.

**Session tips:** Colab disks reset when the runtime ends. Sync `checkpoints/` to Drive after each backbone. To resume inference only, restore checkpoints then run `python src/predict.py --out submission.csv`.

## Setup (local)

```bash
pip install -r requirements.txt
```

## Train

Fine-tune one backbone across all folds (saves per-fold raw + EMA checkpoints, OOF
logits, and the resolved config under `checkpoints/<backbone>/`):

```bash
python src/train.py --backbone convnext_small      # primary ConvNeXt-S (21k→1k)
python src/train.py --backbone effv2s              # EfficientNetV2-S (21k)
python src/train.py --backbone vit_base            # ViT-B/16 (augreg 21k)
```

Quick local sanity / label-map gate (subset of classes, few epochs):

```bash
python src/train.py --backbone convnext_small --folds 0 --epochs 5 --limit-classes 10 --device mps
```

Useful flags: `--folds 0,1`, `--epochs N`, `--batch-size N`, `--img-size N`,
`--limit-classes N`, `--no-mixup`, `--device cuda|mps|cpu`.

## Inference → submission.csv

Loads every fold of every backbone in `config.ensemble`, applies hflip TTA, averages
softmax, and writes a **format-validated** `submission.csv` for all **1,036** test IDs
(numeric sort: `0.jpg` … `1035.jpg`):

```bash
python src/predict.py --out submission.csv
python src/predict.py --backbones convnext_small --out submission.csv   # single backbone
```

## Reproducibility

- Fixed `SEED` in `config.yaml`; `seed_everything` seeds python/numpy/torch and sets
  `cudnn.deterministic=True`. Residual GPU nondeterminism (some CUDA/MPS kernels) means
  bitwise-identical runs aren't guaranteed, but results are stable across seeds.
- Every hyperparameter lives in `config.yaml`; each `train.py` run snapshots the resolved
  config next to its checkpoints.
- Pinned versions in `requirements.txt`. Validation is leak-free **out-of-fold (OOF)**
  accuracy over the whole train set; the Kaggle public LB (~10% of test) is used for
  format sanity only, never for tuning.

## Deliverables

- **Trained weights (Google Drive):** _<link TBD — uploaded after the final ≥96% run>_
- **Kaggle leaderboard screenshot:** _<image TBD — added after submission>_
- **Report (PDF):** _<added with the final submission>_
