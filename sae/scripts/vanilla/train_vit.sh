#!/bin/bash

# Model/Group configuration
MODEL_NAME="vit_vanilla"
CONFIG="config/open_images_vit.json"
WANDB_PROJECT="monoloss"
WANDB_GROUP="$MODEL_NAME"
N_LATENTS=8192
BATCH_SIZE=2048
NUM_EPOCHS=50
MONO_PERIOD=1
MODEL_TYPE="vanilla"
L1_COEF=0.0001
NORMALIZE=""


python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --model $MODEL_TYPE --l1_coef $L1_COEF $NORMALIZE --exp_name no_mono_loss_${MODEL_NAME}_l1_${L1_COEF}
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0001 --model $MODEL_TYPE --l1_coef $L1_COEF $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_mono_0.0001
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.00001 --model $MODEL_TYPE --l1_coef $L1_COEF $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_mono_0.00001
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.00003 --model $MODEL_TYPE --l1_coef $L1_COEF $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_mono_0.00003
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.00005 --model $MODEL_TYPE --l1_coef $L1_COEF $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_mono_0.00005
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.00007 --model $MODEL_TYPE --l1_coef $L1_COEF $NORMALIZE --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_mono_0.00007
