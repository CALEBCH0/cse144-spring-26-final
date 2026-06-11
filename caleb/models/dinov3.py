import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from transformers.modeling_outputs import ImageClassifierOutput

MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class DINOv3Classifier(nn.Module):
    """DINOv3 ViT-L backbone with a linear classification head.

    DINOv3 is a pretrain-only checkpoint (no classification head in HF),
    so we wrap AutoModel and add a linear head on top of pooler_output.
    Architecture matches CLIP-ViT wrapper pattern.
    """

    def __init__(self, num_labels: int, label2id: dict, id2label: dict):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_ID)
        self.config = self.backbone.config
        self.config.num_labels = num_labels
        self.config.label2id = label2id
        self.config.id2label = id2label
        hidden_size = self.config.hidden_size  # 1024 for ViT-L
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return ImageClassifierOutput(loss=loss, logits=logits)


def get_model(num_labels: int, label2id: dict, id2label: dict):
    return DINOv3Classifier(num_labels, label2id, id2label)


def get_processor():
    return AutoImageProcessor.from_pretrained(MODEL_ID)
