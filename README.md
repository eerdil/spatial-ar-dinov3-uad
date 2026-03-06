# AutoregressiveAnomDino

Source code for the paper **"Autoregressive Anomaly Detection with DINOv3 Features"**.

## Overview

This project detects anomalies in images by training a lightweight **autoregressive (AR) model** to predict [DINOv3](https://github.com/facebookresearch/dinov2) feature maps. The AR model is trained exclusively on normal (healthy) images. At test time, anomalous regions produce feature patterns that the AR model cannot predict well — resulting in high reconstruction error, which directly serves as the anomaly map.

**Key design choices:**
- Frozen DINOv3 backbone (no fine-tuning) extracts rich semantic features
- PixelCNN-style masked convolutions or transformer with causal/neighborhood attention for autoregressive prediction
- Trained with MSE loss on normal images only — no anomaly labels required at training time
- Pixel-level anomaly maps evaluated with AUROC and AUPR

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Pretrained DINOv3 Checkpoints

Download the DINOv3 checkpoints and place them in `pretrained_models/` at the project root. The expected filenames are:

| Model | Filename |
|-------|----------|
| ViT-S/16 | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| ViT-S/16+ | `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` |
| ViT-B/16 | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| ViT-L/16 | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |
| ViT-H/16+ | `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` |
| ViT-7B/16 | `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` |
| ConvNeXt-Tiny | `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` |
| ConvNeXt-Small | `dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth` |
| ConvNeXt-Base | `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` |
| ConvNeXt-Large | `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth` |

If your checkpoints are stored elsewhere, set the environment variable:

```bash
export DINO_CHECKPOINTS_PATH=/path/to/your/checkpoints
```

---

## Data Setup

Place your dataset inside the `data/` directory. The expected folder structure is:

```
data/
  <dataset_name>/
    train/
      good/
        img/
          image1.png
          image2.png
          ...
    valid/
      good/
        img/   *.png
        label/ *.png    (binary masks, 0=normal, 255=anomalous)
      Ungood/
        img/   *.png
        label/ *.png
    test/
      good/
        img/   *.png
        label/ *.png
      Ungood/
        img/   *.png
        label/ *.png
```

Supported dataset names: `brats`, `bmad`, `resc`.

To use a custom dataset path, set the corresponding environment variable:

```bash
export DATASET_BRATS_PATH=/path/to/brats
export DATASET_BMAD_PATH=/path/to/bmad
export DATASET_RESC_PATH=/path/to/resc
```

---

## Training

```bash
python train.py \
  --dataset_name brats \
  --model dinov3_vits16 \
  --img_size 448 \
  --batch_size 64 \
  --epochs 40 \
  --lr 1e-3 \
  --kernel_size 3 \
  --dilation_schedule 4 4 4 4 \
  --center_masked_first \
  --wandb_project_name my_project
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `dinov3_vits16` | DINOv3 backbone variant |
| `--ar_model` | `conv` | AR model type: `conv` or `transformer` |
| `--dataset_name` | `brats` | Dataset: `brats`, `bmad`, `resc` |
| `--img_size` | `448` | Input image resolution |
| `--batch_size` | `64` | Training batch size |
| `--epochs` | `40` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate (AdamW) |
| `--kernel_size` | `3` | Convolution kernel size |
| `--dilation_schedule` | `4 4 4 4` | Per-layer dilation values |
| `--non_causal` | off | Use standard (non-causal) convolutions |
| `--center_masked_first` | off | Bidirectional mode: predict center from all neighbors |
| `--seed` | `42` | Random seed |
| `--wandb_project_name` | — | W&B project name for logging |

Results and checkpoints are saved to `results/<experiment_name>/`.

---

## Testing

```bash
python test.py \
  --dataset_name brats \
  --model dinov3_vits16 \
  --experiment_name <experiment_name> \
  --ckpt_name model_best_val_aupr.pth
```

`<experiment_name>` must match a folder in `results/`. Available checkpoint names:
- `model_best_val_loss.pth`
- `model_best_val_aupr.pth`
- `model_best_test_aupr.pth`

---

## Path Configuration

All paths have sensible defaults relative to the project root. To override any of them, use environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_PATH` | project root | Base directory for results |
| `DINO_CHECKPOINTS_PATH` | `pretrained_models/` | Where DINOv3 `.pth` files are stored |
| `DATASET_BRATS_PATH` | `data/brats/` | BraTS dataset root |
| `DATASET_BMAD_PATH` | `data/bmad/` | BMAD dataset root |
| `DATASET_RESC_PATH` | `data/resc/` | RESC dataset root |

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{TODO,
  title={TODO},
  author={TODO},
  year={2025}
}
```
