#!/bin/bash
# Train SAEs on OpenImages features - Baseline vs MonoLoss
# These are the optimal hyperparameters found via grid search
set -e

OUTPUT_DIR="./checkpoints/openimages"
mkdir -p "$OUTPUT_DIR"

###############################################################################
# BatchTopK (topk_k=64)
###############################################################################

# CLIP
python main.py --dataset_config config/open_images_clip_image.json --model batch_topk --batch_size 2048 --topk_k 64 --mono_coef 0.0 --exp_name no_mono_loss_clip_image_batch_topk_topk64 --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_clip_image.json --model batch_topk --batch_size 2048 --topk_k 64 --mono_coef 0.0005 --exp_name mono_loss_clip_image_batch_topk_topk64_mono_00005_period_1 --output_dir "$OUTPUT_DIR" --skip_existing

# SigLIP2
python main.py --dataset_config config/open_images_siglip2_image.json --model batch_topk --batch_size 2048 --topk_k 64 --mono_coef 0.0 --exp_name no_mono_loss_siglip2_batch_topk_topk64 --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_siglip2_image.json --model batch_topk --batch_size 2048 --topk_k 64 --mono_coef 0.0002 --exp_name mono_loss_siglip2_batch_topk_topk64_mono_00002_period_1 --output_dir "$OUTPUT_DIR" --skip_existing

# ViT
python main.py --dataset_config config/open_images_vit.json --model batch_topk --batch_size 2048 --topk_k 64 --mono_coef 0.0 --exp_name no_mono_loss_vit_batch_topk_topk64 --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_vit.json --model batch_topk --batch_size 2048 --topk_k 64 --mono_coef 0.0007 --exp_name mono_loss_vit_batch_topk_topk64_mono_00007_period_1 --output_dir "$OUTPUT_DIR" --skip_existing

###############################################################################
# TopK (uses dead_check_interval=10000)
###############################################################################

# CLIP
python main.py --dataset_config config/open_images_clip_image.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.0 --exp_name openimages_clip_topk_baseline --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_clip_image.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.5 --exp_name openimages_clip_topk_mono_0.5 --output_dir "$OUTPUT_DIR" --skip_existing

# SigLIP2
python main.py --dataset_config config/open_images_siglip2_image.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.0 --exp_name openimages_siglip_topk_baseline --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_siglip2_image.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.3 --exp_name openimages_siglip_topk_mono_0.3 --output_dir "$OUTPUT_DIR" --skip_existing

# ViT
python main.py --dataset_config config/open_images_vit.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.0 --exp_name openimages_vit_topk_baseline --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_vit.json --model topk --batch_size 2048 --dead_check_interval 10000 --mono_coef 0.9 --exp_name openimages_vit_topk_mono_0.9 --output_dir "$OUTPUT_DIR" --skip_existing

###############################################################################
# JumpReLU (l1=0.001, bandwidth=0.001, normalize)
###############################################################################

# CLIP
python main.py --dataset_config config/open_images_clip_image.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0 --exp_name no_mono_loss_clip_image_jumprelu_l1_0.001_bw_0.001_normalize --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_clip_image.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0005 --exp_name mono_loss_clip_image_jumprelu_l1_0.001_bw_0.001_mono_0.0005_normalize --output_dir "$OUTPUT_DIR" --skip_existing

# SigLIP2
python main.py --dataset_config config/open_images_siglip2_image.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0 --exp_name no_mono_loss_siglip2_image_jumprelu_l1_0.001_bw_0.001_normalize --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_siglip2_image.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0001 --exp_name mono_loss_siglip2_image_jumprelu_l1_0.001_bw_0.001_mono_0.0001_normalize --output_dir "$OUTPUT_DIR" --skip_existing

# ViT
python main.py --dataset_config config/open_images_vit.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0 --exp_name no_mono_loss_vit_jumprelu_l1_0.001_bw_0.001_normalize --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_vit.json --model jumprelu --batch_size 2048 --l1_coef 0.001 --bandwidth 0.001 --normalize --mono_coef 0.0005 --exp_name mono_loss_vit_jumprelu_l1_0.001_bw_0.001_mono_0.0005_normalize --output_dir "$OUTPUT_DIR" --skip_existing

###############################################################################
# Vanilla (l1=0.0001)
###############################################################################

# CLIP
python main.py --dataset_config config/open_images_clip_image.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.0 --exp_name no_mono_loss_clip_image_vanilla_l1_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_clip_image.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.0001 --exp_name mono_loss_clip_image_vanilla_l1_0.0001_mono_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing

# SigLIP2
python main.py --dataset_config config/open_images_siglip2_image.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.0 --exp_name no_mono_loss_siglip_vanilla_l1_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_siglip2_image.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.00003 --exp_name mono_loss_siglip_vanilla_l1_0.0001_mono_0.00003 --output_dir "$OUTPUT_DIR" --skip_existing

# ViT
python main.py --dataset_config config/open_images_vit.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.0 --exp_name no_mono_loss_vit_vanilla_l1_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing
python main.py --dataset_config config/open_images_vit.json --model vanilla --batch_size 2048 --l1_coef 0.0001 --mono_coef 0.0001 --exp_name mono_loss_vit_vanilla_l1_0.0001_mono_0.0001 --output_dir "$OUTPUT_DIR" --skip_existing

echo "OpenImages training complete!"
