RDZV_PORT="${1:?rdzv_port required (e.g., 29653)}"
LAMBDA="${2:?monoloss_lambda required (e.g., 0.0)}"

torchrun --nproc_per_node=1 --rdzv-endpoint="localhost:${RDZV_PORT}" finetuning/train.py \
  --monoloss_lambda "${LAMBDA}" \
  --intermediate-layer --ex-factor 1 --act-type topk --k 2048 \
  --lr 0.1 \
  --data-path /scratch/ntanh/cifar10/ --output-dir /scratch/ntanh/MonoLoss_results_cifar10 \
  --model resnet50 --batch-size 1024 \
  --lr-scheduler cosineannealinglr --lr-warmup-epochs 0 --lr-warmup-method linear \
  --auto-augment ta_wide --epochs 90 --random-erase 0.1 --weight-decay 0.00002 \
  --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
  --train-crop-size 176 --model-ema --val-resize-size 232 \
  --ra-sampler --ra-reps=4 --workers 16 --print-freq 10 \
  --pre-extracted-train-features-path pre_extracted_features/cifar10_train_features_clip_vit_base_patch32_no_abs_path.pt \
  --pre-extracted-val-features-path pre_extracted_features/cifar10_test_features_clip_vit_base_patch32_no_abs_path.pt 