• Label 0 for class “0”
• Label 1 for class “1”
• And so on
• More advanced architectures
• More sophisticated data augmentations
• Tuned hyperparameters
• Whether the model fits in GPU memory
• How long training takes
• Using a fixed random seed
• Reporting averaged metrics overall multiple runs
• Providing detailed execution instructions



0, if no submission,
70 + max(0, x − 60), otherwise,



p, if there’s a presentation,
−10, if no presentation.



r, if report+code+weights are submitted,
−10, otherwise.
.
 # CSE 144 Final Project

 Transfer learning pipeline for the UCSC CSE 144 Spring 2026 final image classification challenge.

 Competition page: <https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project>

 ## Overview

 This repository contains experiments for a 100-class image classification task built from a small sampled dataset. The project focuses on transfer learning with pretrained vision backbones and compares several families of models, including:

 - DINOv2
 - ConvNeXt
 - ViT
 - ResNet baselines

 The codebase includes:

 - A cross-validation training pipeline
 - A hyperparameter search script
 - Model definitions and preprocessing utilities
 - Saved checkpoints for multiple model families
 - A project report with experiment notes and results

 ## Dataset

 The competition dataset is expected under `data/`.

 ```text
 data/
	 train/
		 0/
		 1/
		 ...
		 99/
	 test/
	 sample_submission.csv
 ```

 Important label convention:

 - Folder `0` must map to label `0`
 - Folder `1` must map to label `1`
 - ...
 - Folder `99` must map to label `99`

 The current pipeline loads the training split from `data/train` and performs stratified cross-validation.

 ## Environment Setup

 Install dependencies:

 ```bash
 pip install -r requirements.txt
 ```

 Main packages used:

 - PyTorch
 - TorchVision
 - Transformers
 - Datasets
 - timm
 - peft
 - scikit-learn

 ## Repository Layout

 ```text
 config.py        Default training/search configuration
 pipeline.py      Main training and cross-validation entry point
 search.py        Hyperparameter search runner
 resulter.py      Result formatting/export helpers
 transforms.py    Image preprocessing and augmentation setup
 utils.py         Training utilities, TTA, ensemble helpers, LoRA helpers
 models/          Model factory implementations
 checkpoints/     Saved checkpoints for trained models
 report.md        Project report draft and experiment summary
 docs/            Background notes on models and prior methods
 ```

 ## Running Experiments

 ### 1. Configure the run

 Edit `config.py` to choose:

 - `SELECTED_MODELS`
 - `NUM_EPOCHS`
 - `BATCH_SIZE`
 - `UNFREEZE_BLOCKS`
 - augmentation toggles such as `USE_MIXUP` and `USE_RANDAUGMENT`
 - advanced options such as TTA, ensemble prediction, progressive unfreezing, and LoRA

 Current defaults are tuned toward the best search findings in this repository.

 ### 2. Run cross-validation training

 ```bash
 python pipeline.py
 ```

 To also export a text summary:

 ```bash
 python pipeline.py --export results.txt
 ```

 What `pipeline.py` does:

 - Loads `data/train` with the Hugging Face `imagefolder` loader
 - Runs stratified `N_FOLDS` cross-validation
 - Fine-tunes the selected pretrained models
 - Reports accuracy, precision, recall, and macro F1
 - Optionally applies TTA and fold ensembling

 ### 3. Run hyperparameter search

 ```bash
 python search.py
 ```

 The search script:

 - Evaluates predefined experiment configs
 - Uses resume-friendly logging in `search_results.csv`
 - Skips completed configs already written to the CSV
 - Reports mean and standard deviation across folds

 ## Current Configuration Notes

 The default config in `config.py` currently enables:

 - Mixup
 - Progressive unfreezing
 - Test-time augmentation
 - Fold ensembling
 - RandAugment

 The default config currently disables:

 - LLRD, because search results showed little benefit
 - Color jitter, because it hurt ConvNeXt performance in this dataset
 - Random erasing, pending stronger evidence from search
 - LoRA, unless explicitly selected for DINOv2 or ViT experiments

 ## Results Summary

 Highlights from `report.md` and `search_results.csv`:

 - ConvNeXt-Base and ViT-B/16 reached about 70.7% mean validation accuracy in 5-fold CV
 - ConvNeXt-Large improved that to 77.85% mean validation accuracy in the stronger multi-technique setup
 - The best search result in the repository is DINOv2 with `unfreeze_blocks=2`, reaching 84.24% mean CV accuracy

 These results suggest that selective partial unfreezing is more effective than full fine-tuning on this small dataset.

 ## Notes on Submission

 The assignment requires a Kaggle submission file with columns:

 - `ID`
 - `Label`

 This repository currently focuses on training, evaluation, checkpointing, and search. If you need a dedicated inference script for generating `submission.csv` from `data/test`, that should be added as a separate step.

 ## References

 - Competition page: <https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project>
 - Report draft: `report.md`
 - Model and SOTA notes: `docs/`