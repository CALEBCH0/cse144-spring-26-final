"""Model factory: timm backbone + new 100-way head with dropout, EMA wrapper,
and layer-wise-LR-decay parameter groups for the optimizer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm
from timm.optim import param_groups_layer_decay, param_groups_weight_decay
from timm.utils import ModelEmaV3


def create_model(
    timm_name: str,
    num_classes: int = 100,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    img_size: int | None = None,
) -> nn.Module:
    """Build a timm backbone with a fresh `num_classes` head.

    `drop_rate` -> classifier dropout, `drop_path_rate` -> stochastic depth.
    `img_size` is passed to models that accept dynamic input sizing (ViT/EVA);
    for CNNs it is ignored by timm.
    """
    kwargs = dict(
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    try:
        return timm.create_model(timm_name, img_size=img_size, **kwargs) if img_size else \
            timm.create_model(timm_name, **kwargs)
    except TypeError:
        # Backbone doesn't accept img_size (most CNNs) — input size is set via transforms.
        return timm.create_model(timm_name, **kwargs)


def build_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    layer_decay: float | None,
) -> list[dict]:
    """Optimizer param groups.

    With `layer_decay` in (0,1): layer-wise LR decay (backbone layers get
    progressively smaller LR than the head) — the standard fine-tuning recipe for
    pretrained backbones. Otherwise a plain weight-decay split (no decay on
    norms/biases). The head always trains at the full `lr`.
    """
    no_decay_skip = getattr(model, "no_weight_decay", lambda: set())()
    if layer_decay and 0.0 < layer_decay < 1.0:
        return param_groups_layer_decay(
            model,
            weight_decay=weight_decay,
            layer_decay=layer_decay,
            no_weight_decay_list=no_decay_skip,
        )
    return param_groups_weight_decay(model, weight_decay=weight_decay)


def create_ema(model: nn.Module, decay: float) -> ModelEmaV3:
    """EMA of weights — the selected/reported model.

    use_warmup=True ramps the effective decay (low early -> `decay` late) so the
    EMA tracks the model from the start. Critical here: total training is only
    ~1k steps, so a fixed high decay (e.g. 0.9998 => ~5k-step window) would leave
    the EMA stuck near random init. Caller must pass `step` to update().
    """
    # foreach=False: the fused _foreach_lerp_ kernel is unimplemented on MPS; the
    # per-tensor path is correct everywhere (the speed delta is negligible at this scale).
    return ModelEmaV3(model, decay=decay, use_warmup=True, foreach=False)


@torch.no_grad()
def freeze_backbone(model: nn.Module, freeze: bool) -> None:
    """Freeze everything except the classifier head (for optional head-only warmup)."""
    classifier = model.get_classifier()
    head_params = set(id(p) for p in classifier.parameters())
    for p in model.parameters():
        p.requires_grad = not freeze
    for p in classifier.parameters():
        p.requires_grad = True
    # keep head trainable regardless
    _ = head_params
