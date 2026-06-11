import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPImageProcessor
from transformers.modeling_outputs import ImageClassifierOutput

MODEL_ID = "openai/clip-vit-large-patch14"


class CLIPClassifier(nn.Module):
    """CLIP vision encoder with a linear classification head."""

    def __init__(self, num_labels: int):
        super().__init__()
        clip = CLIPModel.from_pretrained(MODEL_ID)
        self.vision_model = clip.vision_model
        self.config = self.vision_model.config  # required by apply_freeze / LLRD
        hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output  # (B, hidden_size)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return ImageClassifierOutput(loss=loss, logits=logits)


def get_model(num_labels: int, label2id: dict, id2label: dict):
    return CLIPClassifier(num_labels)


def get_processor():
    return CLIPImageProcessor.from_pretrained(MODEL_ID)
