#!/bin/bash

# Model/Group configuration
MODEL_NAME="vit_jumprelu"
CONFIG="config/open_images_vit.json"
WANDB_PROJECT="monoloss"
WANDB_GROUP="$MODEL_NAME"
N_LATENTS=8192
BATCH_SIZE=2048
NUM_EPOCHS=50
MONO_PERIOD=1
MODEL_TYPE="jumprelu"
L1_COEF=0.001
BANDWIDTH=0.001

# JumpReLU experiments with mono_coef ablation
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name no_mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_normalize

# Lower values
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.000001 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.000001_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.000003 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.000003_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.000005 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.000005_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.00001 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.00001_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.00003 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.00003_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.00005 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.00005_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0001 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.0001_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0003 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.0003_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0005 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.0005_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.001 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.001_normalize

# Higher values (above l1_coef)
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.003 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.003_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.005 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.005_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.007 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.007_normalize
python main.py --skip_existing --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.01 --model $MODEL_TYPE --l1_coef $L1_COEF --bandwidth $BANDWIDTH --normalize --exp_name mono_loss_${MODEL_NAME}_l1_${L1_COEF}_bw_${BANDWIDTH}_mono_0.01_normalize