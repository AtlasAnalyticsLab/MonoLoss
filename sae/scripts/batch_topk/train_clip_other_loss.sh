#!/bin/bash

# Model/Group configuration
MODEL_NAME="test_other_loss_clip_image"

# Common arguments
CONFIG="config/open_images_clip_image.json"
WANDB_PROJECT="monoloss"
WANDB_GROUP="$MODEL_NAME"
N_LATENTS=8192
BATCH_SIZE=2048
NUM_EPOCHS=50
MONO_PERIOD=1
MODEL_TYPE="batch_topk"
NORMALIZE=""  # Set to "--normalize" to enable layer normalization, or leave empty to disable

# TopK 64 experiments with higher mono_coef values (100x+ increase)
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 1 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_topk64_mono_1_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 2 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_topk64_mono_2_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 3 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_topk64_mono_3_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 5 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_topk64_mono_5_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 7 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_topk64_mono_7_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 10 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_topk64_mono_10_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 20 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_topk64_mono_20_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 50 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_topk64_mono_50_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 100 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_topk64_mono_100_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 1000 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_topk64_mono_1000_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name ${MODEL_NAME}_no_mono_loss_topk64
