#!/bin/bash

# Model/Group configuration
MODEL_NAME="dino2_batch_topk_higher_batch"

# Common arguments
CONFIG="config/open_images_dino.json"
WANDB_PROJECT="monoloss"
WANDB_GROUP="$MODEL_NAME"
N_LATENTS=8192
BATCH_SIZE=4096
NUM_EPOCHS=50
MONO_PERIOD=1
MODEL_TYPE="batch_topk"
NORMALIZE=""  # Set to "--normalize" to enable layer normalization, or leave empty to disable

# TopK 64 experiments with small mono_coef values
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name no_mono_loss_${MODEL_NAME}_topk64_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0001 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_00001_period_1_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0002 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_00002_period_1_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0003 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_00003_period_1_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0005 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_00005_period_1_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0007 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_00007_period_1_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.001 --model $MODEL_TYPE --topk_k 64 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk64_mono_0001_period_1_batch${BATCH_SIZE}

# TopK 128 experiments with small mono_coef values
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --model $MODEL_TYPE --topk_k 128 $NORMALIZE --exp_name no_mono_loss_${MODEL_NAME}_topk128_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0001 --model $MODEL_TYPE --topk_k 128 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_00001_period_1_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0002 --model $MODEL_TYPE --topk_k 128 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_00002_period_1_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0003 --model $MODEL_TYPE --topk_k 128 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_00003_period_1_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0005 --model $MODEL_TYPE --topk_k 128 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_00005_period_1_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0007 --model $MODEL_TYPE --topk_k 128 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_00007_period_1_batch${BATCH_SIZE}
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.001 --model $MODEL_TYPE --topk_k 128 $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_topk128_mono_0001_period_1_batch${BATCH_SIZE}

