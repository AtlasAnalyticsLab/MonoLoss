#!/bin/bash

# Model/Group configuration
MODEL_NAME="vit_topk"

# Common arguments
CONFIG="config/open_images_vit.json"
WANDB_PROJECT="monoloss"
WANDB_GROUP="$MODEL_NAME"
N_LATENTS=8192
BATCH_SIZE=2048
NUM_EPOCHS=50
MONO_PERIOD=1

# TopK 64 experiments
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --topk 64 --exp_name no_mono_loss_${MODEL_NAME}_topk64
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.1 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_01_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.3 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_03_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.5 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_05_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.7 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_07_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.9 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_09_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 1.0 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_1_period_1

# TopK 128 experiments
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --topk 128 --exp_name no_mono_loss_${MODEL_NAME}_topk128
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.1 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_01_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.3 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_03_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.5 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_05_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.7 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_07_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.9 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_09_period_1
python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 1.0 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_1_period_1


# # TopK 64 experiments
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --topk 64 --exp_name no_mono_loss_${MODEL_NAME}_topk64
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0001 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_0001_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0003 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_0003_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0005 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_0005_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0007 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_0007_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0009 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_0009_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.001 --topk 64 --exp_name mono_loss_${MODEL_NAME}_topk64_mono_001_period_1

# # TopK 128 experiments
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0 --topk 128 --exp_name no_mono_loss_${MODEL_NAME}_topk128
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0001 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_0001_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0003 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_0003_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0005 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_0005_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0007 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_0007_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.0009 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_0009_period_1
# python main.py --dataset_config $CONFIG --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP --n_latents $N_LATENTS --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --mono_period $MONO_PERIOD --mono_coef 0.001 --topk 128 --exp_name mono_loss_${MODEL_NAME}_topk128_mono_001_period_1
