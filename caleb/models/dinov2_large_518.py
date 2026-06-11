from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "facebook/dinov2-large"


def get_model(num_labels: int, label2id: dict, id2label: dict):
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
    )
    return AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
    )


def get_processor():
    # 518px = DINOv2's native pretraining resolution (patch_size=14 → 37×37 patches)
    # Default 224px crops down from native, discarding spatial detail
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    processor.size = {"shortest_edge": 518}
    if hasattr(processor, "crop_size"):
        processor.crop_size = {"height": 518, "width": 518}
    return processor
