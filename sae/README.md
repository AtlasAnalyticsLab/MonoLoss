# MonoLoss

Training Sparse Autoencoders (SAEs) with a novel **monosemanticity loss** that encourages neurons to represent semantically coherent concepts.

## Installation

```bash
pip install torch numpy wandb tqdm lmdb open_clip_torch transformers pandas pillow
```

## Quick Start

### 1. Extract Features

Extract vision features to LMDB format:

```bash
# ImageNet (6 commands: 3 models × 2 splits)
python extract_imagenet.py --model clip --split train
python extract_imagenet.py --model clip --split validation
python extract_imagenet.py --model siglip2 --split train
python extract_imagenet.py --model siglip2 --split validation
python extract_imagenet.py --model vit --split train
python extract_imagenet.py --model vit --split validation

# OpenImages (9 commands: 3 models × 3 splits)
python extract_open_images.py --model clip --split train --dataset_dir /path/to/OpenImages
python extract_open_images.py --model clip --split validation --dataset_dir /path/to/OpenImages
python extract_open_images.py --model clip --split test --dataset_dir /path/to/OpenImages
python extract_open_images.py --model siglip2 --split train --dataset_dir /path/to/OpenImages
python extract_open_images.py --model siglip2 --split validation --dataset_dir /path/to/OpenImages
python extract_open_images.py --model siglip2 --split test --dataset_dir /path/to/OpenImages
python extract_open_images.py --model vit --split train --dataset_dir /path/to/OpenImages
python extract_open_images.py --model vit --split validation --dataset_dir /path/to/OpenImages
python extract_open_images.py --model vit --split test --dataset_dir /path/to/OpenImages
```

**Supported models:** `clip`, `siglip2`, `vit`

### 2. Create Dataset Config

Create a JSON config file pointing to your LMDB files:

```json
{
  "train_path": "/path/to/features/train/clip_image_features.lmdb",
  "val_path": "/path/to/features/validation/clip_image_features.lmdb",
  "test_path": "/path/to/features/test/clip_image_features.lmdb"
}
```

### 3. Train SAE

```bash
# Baseline (no monosemanticity loss)
python main.py --dataset_config config/imagenet_clip.json --model batch_topk --batch_size 2048 --mono_coef 0.0 --exp_name baseline

# With MonoLoss
python main.py --dataset_config config/imagenet_clip.json --model batch_topk --batch_size 2048 --mono_coef 0.0003 --exp_name monoloss
```

## Reproducing Paper Results

Run all experiments with optimal hyperparameters:

```bash
# ImageNet experiments (24 runs: 4 models × 3 encoders × 2 conditions)
bash scripts/train_imagenet.sh

# OpenImages experiments
bash scripts/train_openimages.sh
```

## Training Arguments

| Argument | Description |
|----------|-------------|
| `--dataset_config` | JSON file with LMDB paths (required) |
| `--model` | `topk`, `batch_topk`, `vanilla`, `jumprelu` |
| `--batch_size` | Batch size (default: 2048) |
| `--mono_coef` | Monosemanticity loss coefficient (0.0 = baseline) |
| `--topk_k` | Top-K sparsity (default: 64) |
| `--n_latents` | Dictionary size (default: 8192) |
| `--l1_coef` | Sparsity penalty: L1 for vanilla, L0 for jumprelu (default: 0.0001) |
| `--bandwidth` | JumpReLU threshold smoothness (default: 0.001) |
| `--normalize` | Normalize inputs to zero mean, unit variance |
| `--dead_check_interval` | Dead neuron check interval for topk (default: 1000, paper uses 10000) |

## Project Structure

```
sae/
├── main.py                  # CLI entry point
├── train_topk.py            # Training loop for TopKSAE
├── train_other.py           # Training loop for BatchTopK/Vanilla/JumpReLU
├── loss.py                  # Monosemanticity loss functions
├── mono_loss.py             # Custom autograd for full-dataset mono loss
├── extract_imagenet.py      # ImageNet feature extraction
├── extract_open_images.py   # OpenImages feature extraction
├── models/                  # SAE architectures
├── dataset/                 # LMDB data loaders
├── config/                  # Dataset configurations
└── scripts/                 # Training scripts
```

## Output

Training outputs to `checkpoints/<exp_name>/`:
- `autoencoder_final.pt` - Model checkpoint
- `training_config.json` - Configuration
- `results.json` - Final metrics (R², monosemanticity scores)
