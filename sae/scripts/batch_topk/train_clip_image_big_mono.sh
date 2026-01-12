#!/bin/bash

# Model/Group configuration
MODEL_NAME="clip_image_batch_topk"

# Common arguments
CONFIG="config/open_images_clip_image.json"
WANDB_PROJECT="monoloss"
WANDB_GROUP="$MODEL_NAME"
N_LATENTS=8192
BATCH_SIZE=2048
NUM_EPOCHS=50
MONO_PERIOD=1
MODEL_TYPE="batch_topk"
NORMALIZE="--normalize"  # Set to "--normalize" to enable layer normalization, or leave empty to disable



# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name no_mono_loss_${MODEL_NAME}_topk64
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.1 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_01_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.3 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_03_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.5 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_05_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.7 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_07_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.9 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_09_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 1.0 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_1_period_1

# TopK 128 experiments
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --topk_k 128 --model $MODEL_TYPE $NORMALIZE --exp_name no_mono_loss_${MODEL_NAME}_topk128
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.1 --topk_k 128 --model $MODEL_TYPE $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_01_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.3 --topk_k 128 --model $MODEL_TYPE $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_03_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.5 --topk_k 128 --model $MODEL_TYPE $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_05_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.7 --topk_k 128 --model $MODEL_TYPE $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_07_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.9 --topk_k 128 --model $MODEL_TYPE $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_09_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 1.0 --topk_k 128 --model $MODEL_TYPE $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_1_period_1
