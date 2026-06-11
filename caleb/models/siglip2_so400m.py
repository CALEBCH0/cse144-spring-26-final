from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

# SO400M = 400M-param ViT-So400M, pretrained at 384px on image-text pairs
# Top team achieved 94.5% OOF with a frozen probe on this backbone
MODEL_ID = "google/siglip2-so400m-patch14-384"


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
    # Naturally 384px — do not override; this is the pretraining resolution
    return AutoImageProcessor.from_pretrained(MODEL_ID)
