import torch
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Lambda,
    Normalize,
    RandAugment,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import TimmWrapperImageProcessor


class _Transform:
    """Picklable batch transform wrapper (needed for Windows multiprocessing)."""

    def __init__(self, tf):
        self.tf = tf

    def __call__(self, batch):
        batch["pixel_values"] = [self.tf(img.convert("RGB")) for img in batch["image"]]
        return batch


def build_transforms(image_processor,
                     use_color_jitter=False,
                     use_randaugment=False,
                     use_random_erasing=False):
    """Build train and val transform callables for a given image processor.

    Pre-tensor PIL augmentations (ColorJitter, RandAugment) are injected before
    the base transform; RandomErasing is injected after ToTensor/Normalize.
    Works for both TimmWrapperImageProcessor and standard HuggingFace processors.
    """
    extra_pre = []
    if use_color_jitter:
        extra_pre.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    if use_randaugment:
        extra_pre.append(RandAugment(num_ops=2, magnitude=6))

    extra_post = []
    if use_random_erasing:
        extra_post.append(RandomErasing(p=0.2))

    if isinstance(image_processor, TimmWrapperImageProcessor):
        base_train = image_processor.train_transforms
        base_val = image_processor.val_transforms
        _train_tf = Compose(extra_pre + [base_train] + extra_post) if (extra_pre or extra_post) else base_train
        _val_tf = base_val
    else:
        if "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            size = (image_processor.size["height"], image_processor.size["width"])

        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std"):
            normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        else:
            normalize = Lambda(lambda x: x)

        _train_tf = Compose(extra_pre + [RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize] + extra_post)
        _val_tf = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    return _Transform(_train_tf), _Transform(_val_tf)
