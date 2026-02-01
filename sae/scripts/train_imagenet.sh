#!/bin/bash
# Train SAEs on ImageNet features - Baseline vs MonoLoss
# These are the optimal hyperparameters found via grid search
set -e

OUTPUT_DIR="./checkpoints/imagenet"
mkdir -p "$OUTPUT_DIR"

###############################################################################
# BatchTopK
###############################################################################

# CLIP
python main.py --dataset_config config/imagenet_clip.json --model batch_topk --batch_size 2048 --mono_coef 0.0 --exp_name clip_batch_topk_baseline --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_clip.json --model batch_topk --batch_size 2048 --mono_coef 0.0003 --exp_name clip_batch_topk_mono_0.0003 --output_dir "$OUTPUT_DIR" --skip_existing

# SigLIP2
python main.py --dataset_config config/imagenet_siglip.json --model batch_topk --batch_size 2048 --mono_coef 0.0 --exp_name siglip_batch_topk_baseline --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_siglip.json --model batch_topk --batch_size 2048 --mono_coef 0.0001 --exp_name siglip_batch_topk_mono_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing

# ViT
python main.py --dataset_config config/imagenet_vit.json --model batch_topk --batch_size 2048 --mono_coef 0.0 --exp_name vit_batch_topk_baseline --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_vit.json --model batch_topk --batch_size 2048 --mono_coef 0.0004 --exp_name vit_batch_topk_mono_0.0004 --output_dir "$OUTPUT_DIR" --skip_existing

###############################################################################
# TopK (uses dead_check_interval=10000)
###############################################################################

# CLIP
python main.py --dataset_config config/imagenet_clip.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.0 --exp_name imagenet_clip_topk_baseline --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_clip.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.09 --exp_name imagenet_clip_topk_mono_0.09 --output_dir "$OUTPUT_DIR" --skip_existing

# SigLIP2
python main.py --dataset_config config/imagenet_siglip.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.0 --exp_name imagenet_siglip_topk_baseline --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_siglip.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.14 --exp_name imagenet_siglip_topk_mono_0.14 --output_dir "$OUTPUT_DIR" --skip_existing

# ViT
python main.py --dataset_config config/imagenet_vit.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.0 --exp_name imagenet_vit_topk_baseline --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_vit.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.09 --exp_name imagenet_vit_topk_mono_0.09 --output_dir "$OUTPUT_DIR" --skip_existing

###############################################################################
# JumpReLU (l1=0.001, bandwidth=0.001, normalize)
###############################################################################

# CLIP
python main.py --dataset_config config/imagenet_clip.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0 --exp_name clip_jumprelu_baseline_l1_0.001_bwd_0.001_normalize --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_clip.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0005 --exp_name clip_jumprelu_mono_0.0005_l1_0.001_bwd_0.001_normalize --output_dir "$OUTPUT_DIR" --skip_existing

# SigLIP2
python main.py --dataset_config config/imagenet_siglip.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0 --exp_name siglip_jumprelu_baseline_l1_0.001_bwd_0.001_normalize --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_siglip.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0001 --exp_name siglip_jumprelu_mono_0.0001_l1_0.001_bwd_0.001_normalize --output_dir "$OUTPUT_DIR" --skip_existing

# ViT
python main.py --dataset_config config/imagenet_vit.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0 --exp_name vit_jumprelu_baseline_l1_0.001_bwd_0.001_normalize --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_vit.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0007 --exp_name vit_jumprelu_mono_0.0007_l1_0.001_bwd_0.001_normalize --output_dir "$OUTPUT_DIR" --skip_existing

###############################################################################
# Vanilla (l1=0.0001)
###############################################################################

# CLIP
python main.py --dataset_config config/imagenet_clip.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.0 --exp_name clip_vanilla_baseline_l1_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_clip.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.0001 --exp_name clip_vanilla_mono_0.0001_l1_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing

# SigLIP2
python main.py --dataset_config config/imagenet_siglip.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.0 --exp_name siglip_vanilla_baseline_l1_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_siglip.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.00003 --exp_name siglip_vanilla_mono_0.00003_l1_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing

# ViT
python main.py --dataset_config config/imagenet_vit.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.0 --exp_name vit_vanilla_baseline_l1_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/imagenet_vit.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.00009 --exp_name vit_vanilla_mono_0.00009_l1_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing

echo "ImageNet training complete!"
