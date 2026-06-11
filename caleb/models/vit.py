from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "timm/vit_base_patch16_224.augreg_in21k_ft_in1k"


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
    return AutoImageProcessor.from_pretrained(MODEL_ID)
