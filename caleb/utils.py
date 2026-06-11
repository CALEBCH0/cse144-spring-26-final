import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback


def signal(event: str, **kwargs):
    parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"##SIGNAL## {event} {parts}", flush=True)


# ── Freeze / unfreeze ─────────────────────────────────────────────────────────

def _count_transformer_layers(model):
    """Detect number of encoder blocks by scanning named parameters."""
    max_idx = -1
    for name, _ in model.named_parameters():
        for prefix in ("encoder.layer.", "layers.", "model.layer.", "timm_model.blocks."):
            if prefix in name:
                try:
                    idx = int(name.split(prefix)[1].split(".")[0])
                    max_idx = max(max_idx, idx)
                except (ValueError, IndexError):
                    pass
    return max_idx + 1 if max_idx >= 0 else 12


def apply_freeze(model, model_name, unfreeze_blocks):
    """Freeze all params then selectively unfreeze top N blocks/stages + classifier.

    Transformers (dinov2, dinov2_large, vit): detects layer count automatically.
    ConvNeXt: unfreeze_blocks out of 4 stages from top.
    unfreeze_blocks=0 -> frozen backbone, head only.
    """
    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_blocks > 0:
        if "convnext" in model_name:
            for name, param in model.named_parameters():
                for i in range(4 - unfreeze_blocks, 4):
                    if f"encoder.stages.{i}." in name:
                        param.requires_grad = True
        else:
            n = _count_transformer_layers(model)
            for name, param in model.named_parameters():
                for i in range(n - unfreeze_blocks, n):
                    if (f"encoder.layer.{i}." in name or f"layers.{i}." in name
                            or f"model.layer.{i}." in name
                            or f"timm_model.blocks.{i}." in name):
                        param.requires_grad = True

    for name, param in model.named_parameters():
        if "classifier" in name or "timm_model.head" in name:
            param.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Layer-wise LR decay (LLRD) ────────────────────────────────────────────────

def get_llrd_optimizer(model, model_name, base_lr, decay_factor, weight_decay):
    """AdamW with per-layer learning rates that decay toward earlier layers.

    Classifier head gets base_lr. Each layer below it is multiplied by
    decay_factor, so shallow layers train more slowly and preserve pretrained
    features. Typical decay_factor: 0.65–0.85.
    """
    groups = []

    if model_name in ("dinov2", "dinov2_large", "dinov3", "vit",
                      "siglip", "clip_vit",
                      "siglip2_so400m", "siglip2_so400m_512"):
        n = _count_transformer_layers(model)
        # embeddings — lowest LR
        emb_params = [p for n_, p in model.named_parameters()
                      if ("embeddings" in n_ or "patch_embed" in n_
                          or "cls_token" in n_ or "pos_embed" in n_)
                      and p.requires_grad]
        if emb_params:
            groups.append({"params": emb_params, "lr": base_lr * (decay_factor ** n)})

        # encoder layers: layer 0 (shallowest) -> layer n-1 (deepest)
        for i in range(n):
            layer_params = [p for n_, p in model.named_parameters()
                            if (f"encoder.layer.{i}." in n_ or f"layers.{i}." in n_
                                or f"model.layer.{i}." in n_
                                or f"timm_model.blocks.{i}." in n_)
                            and p.requires_grad]
            if layer_params:
                groups.append({"params": layer_params,
                                "lr": base_lr * (decay_factor ** (n - 1 - i))})

    elif model_name == "convnext":
        emb_params = [p for n, p in model.named_parameters()
                      if "embeddings" in n and p.requires_grad]
        if emb_params:
            groups.append({"params": emb_params, "lr": base_lr * (decay_factor ** 4)})

        for i in range(4):
            stage_params = [p for n, p in model.named_parameters()
                            if f"encoder.stages.{i}." in n and p.requires_grad]
            if stage_params:
                groups.append({"params": stage_params,
                                "lr": base_lr * (decay_factor ** (3 - i))})

    # classifier head always gets the full base_lr
    head_params = [p for n, p in model.named_parameters()
                   if ("classifier" in n or "timm_model.head" in n) and p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": base_lr})

    for g in groups:
        g["weight_decay"] = weight_decay

    return torch.optim.AdamW(groups)


# ── Mixup ─────────────────────────────────────────────────────────────────────

class MixupCollator:
    """Replaces the standard collator. Blends pairs of images and creates
    soft one-hot labels for cross-entropy.  alpha=0 disables mixup."""

    def __init__(self, num_classes, alpha=0.2):
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels = torch.tensor([ex["label"] for ex in examples])

        if self.alpha <= 0 or not torch.is_grad_enabled():
            return {"pixel_values": pixel_values, "labels": labels}

        lam = float(np.random.beta(self.alpha, self.alpha))
        idx = torch.randperm(pixel_values.size(0))

        mixed = lam * pixel_values + (1 - lam) * pixel_values[idx]

        y_a = F.one_hot(labels, self.num_classes).float()
        y_b = F.one_hot(labels[idx], self.num_classes).float()
        soft_labels = lam * y_a + (1 - lam) * y_b

        return {"pixel_values": mixed, "labels": soft_labels}


class CustomTrainer(Trainer):
    """Trainer that supports LLRD and soft labels (Mixup).

    Extra constructor args:
        llrd_factor  (float): decay per layer; 1.0 = disabled (uniform LR).
        model_name   (str):   model key used by get_llrd_optimizer.
        use_mixup    (bool):  if True, expects float soft labels from MixupCollator.
    """

    def __init__(self, *args, llrd_factor=1.0, model_name="", use_mixup=False, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llrd_factor = llrd_factor
        self.model_name_for_llrd = model_name
        self.use_mixup = use_mixup
        self.class_weights = class_weights

    def create_optimizer(self):
        if self.llrd_factor < 1.0:
            self.optimizer = get_llrd_optimizer(
                self.model,
                self.model_name_for_llrd,
                self.args.learning_rate,
                self.llrd_factor,
                self.args.weight_decay,
            )
            return self.optimizer
        return super().create_optimizer()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if labels.dtype == torch.float:
            loss = F.cross_entropy(logits, labels)
        elif self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ── Per-epoch train/val accuracy logging ─────────────────────────────────────

class EpochAccuracyCallback(TrainerCallback):
    """Prints train and val accuracy at the end of every epoch.

    train_eval_dataset  — training split with val transforms (no augmentation).
    val_eval_dataset    — validation split with val transforms.
    collate_fn          — plain collate (no mixup).

    Always tracks the best val epoch and restores it at the end of training
    (in-memory, no disk writes). This prevents accuracy regression when the
    final epoch is not the best epoch.
    """

    def __init__(self, train_eval_dataset, val_eval_dataset, collate_fn, batch_size=32):
        self.train_ds = train_eval_dataset
        self.val_ds   = val_eval_dataset
        self.collate  = collate_fn
        self.batch_size = batch_size
        self._best_val_acc = -1.0
        self._best_state = None
        self._best_epoch = -1

    def _eval_acc(self, model, dataset, device):
        import copy
        from torch.utils.data import DataLoader
        from sklearn.metrics import accuracy_score
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            collate_fn=self.collate, shuffle=False)
        preds, labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                pv = batch["pixel_values"].to(device)
                out = model(pixel_values=pv)
                preds.extend(out.logits.argmax(-1).cpu().tolist())
                labels.extend(batch["labels"].tolist())
        return accuracy_score(labels, preds)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        import copy
        device = next(model.parameters()).device
        tr_acc  = self._eval_acc(model, self.train_ds, device)
        val_acc = self._eval_acc(model, self.val_ds,   device)
        gap = tr_acc - val_acc
        epoch = int(state.epoch)
        marker = ""
        if val_acc > self._best_val_acc:
            self._best_val_acc = val_acc
            self._best_epoch = epoch
            self._best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            marker = " *best"
        print(f"\n  [epoch {epoch:>2}]  train={tr_acc:.4f}  val={val_acc:.4f}  gap={gap:+.4f}{marker}")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self._best_state is not None and self._best_epoch != int(state.epoch):
            device = next(model.parameters()).device
            model.load_state_dict({k: v.to(device) for k, v in self._best_state.items()})
            print(f"\n  [BestEpoch] restored ep{self._best_epoch} val={self._best_val_acc:.4f} "
                  f"(final was ep{int(state.epoch)})")


# ── Progressive unfreezing ────────────────────────────────────────────────────

class ProgressiveUnfreezeCallback(TrainerCallback):
    """Unfreezes the full backbone after `unfreeze_epoch` epochs have completed.

    Before that epoch the backbone stays frozen (only the classifier trains),
    giving the head time to stabilise before the backbone starts moving.
    """

    def __init__(self, model_name, unfreeze_epoch):
        self.model_name = model_name
        self.unfreeze_epoch = unfreeze_epoch
        self._unfrozen = False

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not self._unfrozen and state.epoch >= self.unfreeze_epoch:
            for param in model.parameters():
                param.requires_grad = True
            self._unfrozen = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n  [ProgressiveUnfreeze] epoch {state.epoch:.0f}: "
                  f"full backbone unfrozen — {trainable:,} trainable params")


# ── Test-time augmentation (TTA) ──────────────────────────────────────────────

def predict_with_tta(trainer, base_dataset, train_tf, n_augments=5):
    """Run inference n_augments times with random train augmentations and
    average the softmax probabilities.  Returns numpy array (N, num_classes)."""

    tta_dataset = base_dataset.with_transform(train_tf)
    all_probs = []

    for _ in range(n_augments):
        out = trainer.predict(tta_dataset)
        probs = torch.softmax(torch.tensor(out.predictions), dim=-1).numpy()
        all_probs.append(probs)

    return np.mean(all_probs, axis=0)


# ── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble_predict(models, dataset, collate_fn, batch_size=32):
    """Average logits from a list of models over the same dataset.
    Returns numpy array (N, num_classes) of averaged logits."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    all_logits = []
    for model in models:
        model.eval()
        model.to(device)
        fold_logits = []
        with torch.no_grad():
            for batch in loader:
                pv = batch["pixel_values"].to(device)
                out = model(pixel_values=pv)
                fold_logits.append(out.logits.cpu())
        all_logits.append(torch.cat(fold_logits, dim=0))
        model.cpu()
        torch.cuda.empty_cache()

    return torch.stack(all_logits).mean(0).numpy()


# ── CutMix ────────────────────────────────────────────────────────────────────

class CutMixCollator:
    """Pastes a random rectangular patch from one image into another and
    mixes labels proportional to the actual cut area.  alpha=0 disables."""

    def __init__(self, num_classes, alpha=1.0):
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels = torch.tensor([ex["label"] for ex in examples])

        if self.alpha <= 0 or not torch.is_grad_enabled():
            return {"pixel_values": pixel_values, "labels": labels}

        lam = float(np.random.beta(self.alpha, self.alpha))
        idx = torch.randperm(pixel_values.size(0))

        _, _, H, W = pixel_values.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        mixed = pixel_values.clone()
        mixed[:, :, y1:y2, x1:x2] = pixel_values[idx, :, y1:y2, x1:x2]

        # Recompute lam from actual box size so label mix is accurate
        lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

        y_a = F.one_hot(labels, self.num_classes).float()
        y_b = F.one_hot(labels[idx], self.num_classes).float()
        soft_labels = lam * y_a + (1 - lam) * y_b

        return {"pixel_values": mixed, "labels": soft_labels}


class MixupCutMixCollator:
    """Randomly applies Mixup or CutMix per batch (50/50 by default)."""

    def __init__(self, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0, mixup_prob=0.5):
        self.mixup = MixupCollator(num_classes, alpha=mixup_alpha)
        self.cutmix = CutMixCollator(num_classes, alpha=cutmix_alpha)
        self.mixup_prob = mixup_prob

    def __call__(self, examples):
        if np.random.random() < self.mixup_prob:
            return self.mixup(examples)
        return self.cutmix(examples)


# ── Test-image inference (Kaggle submission) ──────────────────────────────────

def predict_test_images(models, image_paths, val_transform, batch_size=32):
    """Ensemble-predict test images. Returns numpy array (N,) of class indices.

    models        — list of trained PyTorch models (1 = single model, N = ensemble)
    image_paths   — paths to test images in submission order
    val_transform — _Transform object from build_transforms; .tf is the actual callable
    """
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = [val_transform.tf(Image.open(p).convert("RGB")) for p in image_paths]

    all_logits = []
    for model in models:
        model.eval()
        model.to(device)
        fold_logits = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(device)
            with torch.no_grad():
                out = model(pixel_values=batch)
            fold_logits.append(out.logits.cpu())
        all_logits.append(torch.cat(fold_logits, dim=0))
        model.cpu()
        torch.cuda.empty_cache()

    return torch.stack(all_logits).mean(0).argmax(dim=1).numpy()


def get_test_probs(models, image_paths, val_transform, batch_size=32):
    """Like predict_test_images but returns averaged softmax probs (N, C) for class-routed ensemble."""
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = [val_transform.tf(Image.open(p).convert("RGB")) for p in image_paths]

    all_logits = []
    for model in models:
        model.eval()
        model.to(device)
        fold_logits = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(device)
            with torch.no_grad():
                out = model(pixel_values=batch)
            fold_logits.append(out.logits.cpu())
        all_logits.append(torch.cat(fold_logits, dim=0))
        model.cpu()
        torch.cuda.empty_cache()

    mean_logits = torch.stack(all_logits).mean(0)
    return torch.softmax(mean_logits, dim=-1).numpy()


# ── LoRA ──────────────────────────────────────────────────────────────────────

# Target modules verified by inspecting named_modules() of each loaded model:
#   DINOv2 (facebook/dinov2-base):    query, value  in dinov2.encoder.layer.{i}.attention.attention.*
#   timm ViT (vit_base_patch16_224):  qkv           in timm_model.blocks.{i}.attn.qkv
_LORA_TARGETS = {
    "dinov2": {"target_modules": ["query", "value"], "modules_to_save": ["classifier"]},
    "vit":    {"target_modules": ["qkv"],            "modules_to_save": ["head"]},
}


def apply_lora(model, model_name, r=8, lora_alpha=16, lora_dropout=0.05):
    """Wrap model with LoRA adapters. Backbone is frozen by PEFT; only LoRA
    weights and the task head remain trainable.  Returns the PEFT model."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("Install peft: pip install 'peft>=0.6.0'")

    if model_name not in _LORA_TARGETS:
        raise ValueError(
            f"LoRA not configured for '{model_name}'. "
            f"Supported: {list(_LORA_TARGETS)}. "
            "Add target_modules to _LORA_TARGETS in utils.py."
        )

    cfg = _LORA_TARGETS[model_name]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=cfg["target_modules"],
        modules_to_save=cfg["modules_to_save"],
        bias="none",
    )
    return get_peft_model(model, lora_config)


def top_confusion_pairs(true_labels, pred_labels, class_names, top_n=10):
    """Return the top_n most-confused class pairs from pooled CV predictions.

    Returns list of (class_a, class_b, n_errors, error_rate) sorted by n_errors desc.
    error_rate = symmetric confusion / (total predictions on both classes).
    """
    from sklearn.metrics import confusion_matrix
    n = len(class_names)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(n)))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            n_confused = cm[i, j] + cm[j, i]
            if n_confused == 0:
                continue
            rate = n_confused / max(cm[i].sum() + cm[j].sum(), 1)
            pairs.append((int(class_names[i]), int(class_names[j]), n_confused, rate))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_n]
