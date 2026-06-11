CSE 144 Final Project Report
Transfer Learning Challenge
Group Members (max 3):
Member 1 Name (CruzID: XXXXX)
Member 2 Name (CruzID: XXXXX)
Member 3 Name (CruzID: XXXXX)
Spring 2026
1 Introduction
1. Problem goal and setting.
2. Why transfer learning is appropriate.
3. Brief summary of your approach and main result.
2 Dataset
1. Number of classes and dataset sizes (train/val/test).
2. Directory structure and label mapping details (ensure labels 0–99 match folders).
3. Preprocessing (resize, normalization) and data augmentation used.
3 Implementation
3.1 Model
1. Pretrained backbone used (e.g., ResNet/EfficientNet/ViT) and why.
2. Architecture changes (classifier head, pooling, dropout, etc.).
3. Fine-tuning strategy (frozen layers, unfreezing schedule).
1
3.2 Training
1. Loss function and optimizer.
2. Learning rate, scheduler, batch size, epochs, weight decay.
3. Hardware/software environment (GPU, PyTorch version).
4 Experiments
1. Baseline setup.
2. Hyperparameter tuning plan and validation method.
3. Ablations (augmentations, model size, freezing strategy, LR, etc.).
5 Results
1. Training/validation accuracy (and loss) and any key curves/plots (optional).
2. Kaggle public leaderboard score (and submission settings).
3. Brief qualitative examples or error analysis highlights (optional).
6 Discussion
1. What worked best and why.
2. Failure cases, overfitting/underfitting observations.
3. Limitations and concrete next improvements.
7 Reproducibility
1. Random seeds and determinism settings.
2. Package versions / environment setup steps.
3. Exact commands to train and to generate submission.csv.
2
8 Team Contributions
1. Member 1: . . .
2. Member 2: . . .
3. Member 3: . . .
9 References
1. TorchVision documentation / pretrained model sources.
2. Any papers or tutorials used.
3