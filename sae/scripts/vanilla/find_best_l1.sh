#!/bin/bash

# Model/Group configuration
MODEL_NAME="clip_image_vanilla_find_best_l1"

# Common arguments
CONFIG="config/open_images_clip_image.json"
WANDB_PROJECT="monoloss"
WANDB_GROUP="$MODEL_NAME"
N_LATENTS=8192
BATCH_SIZE=2048
NUM_EPOCHS=50
MONO_PERIOD=1
MODEL_TYPE="vanilla"
NORMALIZE_FLAGS=("" "--normalize")  # "" = no normalize, "--normalize" = enable normalization

L1_VALUES=(1e-06 3e-06 1e-05 3e-05 0.0001 0.0003 0.001)

for NORMALIZE in "${NORMALIZE_FLAGS[@]}"; do
    if [ -z "$NORMALIZE" ]; then
        NORM_TAG="nonorm"
    else
        NORM_TAG="norm"
    fi

    for L1 in "${L1_VALUES[@]}"; do
        python main.py \
            --dataset_config $CONFIG \
            --wandb_project $WANDB_PROJECT \
            --wandb_group $WANDB_GROUP \
            --n_latents $N_LATENTS \
            --batch_size $BATCH_SIZE \
            --num_epochs $NUM_EPOCHS \
            --mono_period $MONO_PERIOD \
            --mono_coef 0 \
            --model $MODEL_TYPE \
            --l1_coef $L1 \
            $NORMALIZE \
            --exp_name no_mono_loss_${MODEL_NAME}_${NORM_TAG}_l1_${L1}
    done
done

