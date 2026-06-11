from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

# SO400M at 512px — larger spatial resolution than 384px variant (patch16 → 1024 tokens)
# Needs batch_size=8 to stay within 16GB VRAM
MODEL_ID = "google/siglip2-so400m-patch16-512"


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
