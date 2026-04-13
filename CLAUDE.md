# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BrainIAC is a 3D Vision Transformer (ViT-B) foundation model for structural Brain MRI analysis, pretrained with SimCLR contrastive learning. The repository provides training and inference pipelines for multiple downstream tasks on 3D brain MRI (NIFTI format, 96×96×96 voxels).

## Environment Setup

```bash
conda create -n brainiac python=3.9
conda activate brainiac
pip install -r requirements.txt
```

All scripts are run from the `src/` directory. The BrainIAC pretrained checkpoint (`BrainIAC.ckpt`) must be placed in `src/checkpoints/`.

## Running Training and Inference

All training scripts accept a `--config` argument pointing to a YAML config file. Commands are run from the repo root:

```bash
# Train a downstream task (example: brain age)
python src/train_lightning_brainage.py --config src/config_finetune.yml

# Run inference across multiple tasks
python src/test_inference_finetune.py

# Run segmentation inference
python src/test_segmentation.py \
    --config src/config_finetune_segmentation.yml \
    --test_csv "/path/to/test.csv" \
    --checkpoint_path "/path/to/checkpoint.ckpt"

# Extract BrainIAC features (768-dim CLS token embeddings)
python src/get_brainiac_features.py \
    --input_csv data.csv \
    --output_csv features.csv \
    --root_dir /path/to/images

# Generate saliency maps
python src/generate_brainage_vit_saliency.py
```

## Architecture

### Core Model (`src/model.py`)
- **`ViTBackboneNet`**: Loads the SimCLR-pretrained ViT-B backbone from a checkpoint. The backbone is MONAI's `ViT` (96³ input, 16³ patches, hidden_size=768, 12 layers, 12 heads). Returns the CLS token (768-dim) from `forward()`.
- **`Classifier`**: Linear head (`nn.Linear(768, num_classes)`).
- **`SingleScanModel`**: Backbone + dropout + classifier for single-scan tasks.
- **`SingleScanModelBP`**: Processes 2 scans (stacked on dim=1), mean-pools their CLS tokens.
- **`SingleScanModelQuad`**: Processes 4 scans, mean-pools their CLS tokens (used for overall survival with t1c/t1n/t2f/t2w).

### Segmentation Model (`src/segmentation_model.py`)
- **`ViTUNETRSegmentationModel`**: Loads BrainIAC weights into a MONAI UNETR encoder, with a U-Net decoder head. Used for tumor segmentation from FLAIR.

### Training (`src/train_lightning_*.py`)
Each task has its own PyTorch Lightning module. All use WandB logging and `ModelCheckpoint`. The `train.freeze` config key (`"yes"`/`"no"`) controls whether the ViT backbone is frozen (linear probing vs. end-to-end fine-tuning).

Training scripts available:
- `train_lightning_brainage.py` — brain age regression (T1w, MAE)
- `train_lightning_mci.py` — MCI classification (T1w, AUC)
- `train_lightning_idh.py` — IDH mutation classification (dual-scan, AUC)
- `train_lightning_os.py` — overall survival prediction (quad-scan, AUC)
- `train_lightning_multiclass.py` — MR sequence classification (4-class, AUC)
- `train_lightning_segmentation.py` — tumor segmentation (FLAIR, Dice)
- `train_lightning_segperf.py` — segmentation performance prediction
- `train_lightning_rt.py` / `train_lightning_rt_binary.py` — RT-related tasks

### Data (`src/dataset.py`, `src/dataset_segmentation.py`)
- **`BrainAgeDataset`** / **`MCIStrokeDataset`**: Single-image datasets. CSV must have `pat_id` and `label` columns.
- **`DualImageDataset`**: For IDH (two MRI sequences). CSV needs two image path columns.
- **`QuadImageDataset`**: For overall survival (four sequences: t1c, t1n, t2f, t2w).
- **`SequenceDataset`**: For MR sequence classification.
- All datasets use MONAI transforms: load → resize to 96³ → normalize intensity (nonzero, channel-wise) → augment (train only).

### Preprocessing (`src/preprocessing/`)
- `mri_preprocess_3d_simple.py`: Full preprocessing pipeline — registration to atlas (SimpleITK) → brain extraction (HD-BET, a UNet-based DL skull-stripping method).
- Atlas templates in `src/preprocessing/atlases/`.
- HD-BET model weights at `src/preprocessing/hd-bet_params/0.model`.

## Configuration

Two config files control training:
- `src/config_finetune.yml`: For classification/regression tasks. Key fields: `data.csv_file`, `data.val_csv`, `data.root_dir`, `simclrvit.ckpt_path`, `train.freeze`, `optim.lr`.
- `src/config_finetune_segmentation.yml`: For segmentation. Key fields: `data.train_csv`, `data.val_csv`, `pretrain.simclr_checkpoint_path`, `training.freeze`.

## Inference Configuration (`src/test_inference_finetune.py`)

The `DATASETS` dict at the top of the file configures which tasks to evaluate. Each entry specifies `checkpoint_path`, `test_csv_path`, `root_dir`, `output_csv_path`, `task_type` (`"regression"`, `"classification"`, `"multiclass"`), `image_type` (`"single"`, `"dual"`, `"quad"`), and `num_classes`.

## Downstream Tasks Summary

| Task | Input | Model variant | Metric |
|------|-------|--------------|--------|
| Brain Age | T1w (single) | `SingleScanModel` | MAE |
| MCI Classification | T1w (single) | `SingleScanModel` | AUC |
| IDH Mutation | T1w+T2w (dual) | `SingleScanModelBP` | AUC |
| Overall Survival | t1c+t1n+t2f+t2w (quad) | `SingleScanModelQuad` | AUC |
| MR Sequence | Single | `SingleScanModel` (4-class) | AUC |
| Tumor Segmentation | FLAIR (single) | `ViTUNETRSegmentationModel` | Dice |
| Time-to-Stroke | T1w (single) | `SingleScanModel` | MAE |
