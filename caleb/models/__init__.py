from models.siglip2_so400m import MODEL_ID as SIGLIP2_SO400M_MODEL_ID
from models.siglip2_so400m import get_model as get_siglip2_so400m
from models.siglip2_so400m import get_processor as get_siglip2_so400m_processor
from models.siglip2_so400m_512 import MODEL_ID as SIGLIP2_SO400M_512_MODEL_ID
from models.siglip2_so400m_512 import get_model as get_siglip2_so400m_512
from models.siglip2_so400m_512 import get_processor as get_siglip2_so400m_512_processor
from models.siglip import MODEL_ID as SIGLIP_MODEL_ID
from models.siglip import get_model as get_siglip
from models.siglip import get_processor as get_siglip_processor
from models.clip_vit import MODEL_ID as CLIP_VIT_MODEL_ID
from models.clip_vit import get_model as get_clip_vit
from models.clip_vit import get_processor as get_clip_vit_processor
from models.dinov2_giant import MODEL_ID as DINOV2_GIANT_MODEL_ID
from models.dinov2_giant import get_model as get_dinov2_giant
from models.dinov2_giant import get_processor as get_dinov2_giant_processor
from models.convnext import MODEL_ID as CONVNEXT_MODEL_ID
from models.convnext import get_model as get_convnext
from models.convnext import get_processor as get_convnext_processor
from models.convnext_tiny import MODEL_ID as CONVNEXT_TINY_MODEL_ID
from models.convnext_tiny import get_model as get_convnext_tiny
from models.convnext_tiny import get_processor as get_convnext_tiny_processor
from models.dinov2 import MODEL_ID as DINOV2_MODEL_ID
from models.dinov2 import get_model as get_dinov2
from models.dinov2 import get_processor as get_dinov2_processor
from models.dinov3 import MODEL_ID as DINOV3_MODEL_ID
from models.dinov3 import get_model as get_dinov3
from models.dinov3 import get_processor as get_dinov3_processor
from models.dinov2_large import MODEL_ID as DINOV2_LARGE_MODEL_ID
from models.dinov2_large import get_model as get_dinov2_large
from models.dinov2_large import get_processor as get_dinov2_large_processor
from models.dinov2_large_518 import MODEL_ID as DINOV2_LARGE_518_MODEL_ID
from models.dinov2_large_518 import get_model as get_dinov2_large_518
from models.dinov2_large_518 import get_processor as get_dinov2_large_518_processor
from models.convnext_large import MODEL_ID as CONVNEXT_LARGE_MODEL_ID
from models.convnext_large import get_model as get_convnext_large
from models.convnext_large import get_processor as get_convnext_large_processor
from models.resnet50 import MODEL_ID as RESNET50_MODEL_ID
from models.resnet50 import get_model as get_resnet50
from models.resnet50 import get_processor as get_resnet50_processor
from models.resnet101 import MODEL_ID as RESNET101_MODEL_ID
from models.resnet101 import get_model as get_resnet101
from models.resnet101 import get_processor as get_resnet101_processor
from models.vit import MODEL_ID as VIT_MODEL_ID
from models.vit import get_model as get_vit
from models.vit import get_processor as get_vit_processor

MODELS = [
    {
        "name": "dinov3",
        "model_id": DINOV3_MODEL_ID,
        "get_model": get_dinov3,
        "get_processor": get_dinov3_processor,
        "output_dir": "checkpoints/dinov3",
        "learning_rate": 1e-4,
        "llrd_factor": 0.75,     # same as dinov2_large (same model size)
        "num_epochs": 50,         # cfg 118 best: 50ep → 93.79%
        "unfreeze_blocks": 4,     # cfg 118 best: unfreeze=4 (monotonically better than 2 for DINOv3)
    },
    {
        "name": "dinov2",
        "model_id": DINOV2_MODEL_ID,
        "get_model": get_dinov2,
        "get_processor": get_dinov2_processor,
        "output_dir": "checkpoints/dinov2",
        "learning_rate": 1e-4,   # cfg 64: best base config (85.54%)
        "llrd_factor": 0.85,
        "num_epochs": 10,        # base peaks at 10ep; 20ep regresses
        "unfreeze_blocks": 2,
    },
    {
        "name": "dinov2_large",
        "model_id": DINOV2_LARGE_MODEL_ID,
        "get_model": get_dinov2_large,
        "get_processor": get_dinov2_large_processor,
        "output_dir": "checkpoints/dinov2_large",
        "learning_rate": 1e-4,   # cfg 80: best large config (87.39%)
        "llrd_factor": 0.75,
        "num_epochs": 50,        # cfg 80: 50ep best; 30ep was 86.47%
        "unfreeze_blocks": 2,
    },
    {
        "name": "dinov2_large_518",
        "model_id": DINOV2_LARGE_518_MODEL_ID,
        "get_model": get_dinov2_large_518,
        "get_processor": get_dinov2_large_518_processor,
        "output_dir": "checkpoints/dinov2_large_518",
        "learning_rate": 1e-4,
        "llrd_factor": 0.75,
        "num_epochs": 30,
        "unfreeze_blocks": 2,
    },
    {
        "name": "convnext_tiny",
        "model_id": CONVNEXT_TINY_MODEL_ID,
        "get_model": get_convnext_tiny,
        "get_processor": get_convnext_tiny_processor,
        "output_dir": "checkpoints/convnext_tiny",
    },
    {
        "name": "resnet50",
        "model_id": RESNET50_MODEL_ID,
        "get_model": get_resnet50,
        "get_processor": get_resnet50_processor,
        "output_dir": "checkpoints/resnet50",
        "learning_rate": 1e-4,
    },
    {
        "name": "resnet101",
        "model_id": RESNET101_MODEL_ID,
        "get_model": get_resnet101,
        "get_processor": get_resnet101_processor,
        "output_dir": "checkpoints/resnet101",
        "learning_rate": 1e-4,
    },
    {
        "name": "convnext",
        "model_id": CONVNEXT_MODEL_ID,
        "get_model": get_convnext,
        "get_processor": get_convnext_processor,
        "output_dir": "checkpoints/convnext",
        "learning_rate": 3e-4,  # best from LR sweep (config 11: 76.27%)
    },
    {
        "name": "convnext_large",
        "model_id": CONVNEXT_LARGE_MODEL_ID,
        "get_model": get_convnext_large,
        "get_processor": get_convnext_large_processor,
        "output_dir": "checkpoints/convnext_large",
        "learning_rate": 3e-4,  # same family; apply same LR finding
    },
    {
        "name": "vit",
        "model_id": VIT_MODEL_ID,
        "get_model": get_vit,
        "get_processor": get_vit_processor,
        "output_dir": "checkpoints/vit",
    },
    {
        "name": "siglip2_so400m",
        "model_id": SIGLIP2_SO400M_MODEL_ID,
        "get_model": get_siglip2_so400m,
        "get_processor": get_siglip2_so400m_processor,
        "output_dir": "checkpoints/siglip2_so400m",
        "learning_rate": 1e-4,
        "llrd_factor": 0.8,
        "unfreeze_blocks": 2,   # cfg 104 best: unfreeze=2, 20ep → 95.00% (vs unfreeze=4 94.81%, unfreeze=6 94.35%)
        "num_epochs": 20,
    },
    {
        "name": "siglip2_so400m_512",
        "model_id": SIGLIP2_SO400M_512_MODEL_ID,
        "get_model": get_siglip2_so400m_512,
        "get_processor": get_siglip2_so400m_512_processor,
        "output_dir": "checkpoints/siglip2_so400m_512",
        "learning_rate": 1e-4,
        "llrd_factor": 0.8,
        "unfreeze_blocks": 2,   # cfg 112 best: unfreeze=2, 20ep → 94.72% (unfreeze=6 hurts same as 384px)
        "num_epochs": 20,
        "batch_size": 8,        # 1024 tokens at 512px — needs smaller batch
    },
    {
        "name": "siglip",
        "model_id": SIGLIP_MODEL_ID,
        "get_model": get_siglip,
        "get_processor": get_siglip_processor,
        "output_dir": "checkpoints/siglip",
        "learning_rate": 1e-4,
        "unfreeze_blocks": 2,
        "num_epochs": 10,
    },
    {
        "name": "clip_vit",
        "model_id": CLIP_VIT_MODEL_ID,
        "get_model": get_clip_vit,
        "get_processor": get_clip_vit_processor,
        "output_dir": "checkpoints/clip_vit",
        "learning_rate": 1e-4,
        "unfreeze_blocks": 2,
        "num_epochs": 10,
    },
    # TODO: requires gradient_checkpointing, ~4h/run — add to SELECTED_MODELS when ready
    # {
    #     "name": "dinov2_giant",
    #     "model_id": DINOV2_GIANT_MODEL_ID,
    #     "get_model": get_dinov2_giant,
    #     "get_processor": get_dinov2_giant_processor,
    #     "output_dir": "checkpoints/dinov2_giant",
    #     "learning_rate": 1e-4,
    #     "llrd_factor": 0.75,
    #     "num_epochs": 30,
    #     "unfreeze_blocks": 2,
    #     "batch_size": 8,
    # },
]
