RDZV_PORT="${1:?rdzv_port required (e.g., 29653)}"
LAMBDA="${2:?monoloss_lambda required (e.g., 0.0)}"

torchrun --nproc_per_node=1 --rdzv-endpoint="localhost:${RDZV_PORT}" finetuning/train.py \
  --lr 0.01 --monoloss_lambda "${LAMBDA}" \
  --intermediate-layer --ex-factor 1 --act-type topk --k 768 \
  --data-path /scratch/ntanh/cifar100/ --output-dir /scratch/ntanh/MonoLoss_results_cifar100 \
  --model clip_vit_b_32 --epochs 90 --batch-size 1024 --opt sgd --wd 0.0001 \
  --lr-scheduler steplr --lr-warmup-method "constant" --lr-warmup-epochs 0 \
  --lr-warmup-decay 0.01 --amp --label-smoothing 0 --mixup-alpha 0 \
  --cutmix-alpha 0 --model-ema --workers 16 \
  --interpolation bicubic --val-resize-size 224 --val-crop-size 224 --wandb-project monoloss-finetuning-vit --print-freq 10 \
  --pre-extracted-train-features-path pre_extracted_features/cifar100_train_features_clip_vit_base_patch32_no_abs_path.pt \
  --pre-extracted-val-features-path pre_extracted_features/cifar100_test_features_clip_vit_base_patch32_no_abs_path.pt 