# Trained Model Weights

The final trained checkpoints are too large for GitHub.

Before submitting the repository, upload the final checkpoint folder to Google
Drive and replace the placeholder below with a shareable link:

**Google Drive weights link:** https://drive.google.com/drive/folders/1B2p5ubG2Yc7x4FiWw4xVz9-ZXNZAlHzD?usp=share_link

Expected local checkpoint layout:

```text
checkpoints_local/
  convnext_base/
    fold0_ema.pt
    fold1_ema.pt
    fold2_ema.pt
    fold3_ema.pt
    fold4_ema.pt
    oof.npz
    resolved_config.yaml
    resolved_backbone_spec.yaml
```

The report uses ConvNeXt-B as the best offline model.
