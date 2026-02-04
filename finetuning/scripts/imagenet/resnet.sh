RDZV_PORT="${1:?rdzv_port required (e.g., 29653)}"
LAMBDA="${2:?monoloss_lambda required (e.g., 0.0)}"

torchrun --nproc_per_node=1 --rdzv-endpoint="localhost:${RDZV_PORT}" train.py \
  --monoloss_lambda "${LAMBDA}" \
  --intermediate-layer --ex-factor 1 --act-type topk --k 2048 \
  --data-path /scratch/ntanh/imagenet/ --output-dir /scratch/ntanh/MonoLoss_results_resnet_int \
  --model resnet50 --batch-size 1024 --lr 1e-3 \
  --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
  --auto-augment ta_wide --epochs 90 --random-erase 0.1 --weight-decay 0.00002 \
  --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --workers 16\
  --train-crop-size 176 --model-ema --val-resize-size 232 \
  --pre-extracted-train-features-path pre_extracted_features/imagenet_train_features_clip_vit_base_patch32_no_abs_path.pt \
  --pre-extracted-val-features-path pre_extracted_features/imagenet_test_features_clip_vit_base_patch32_no_abs_path.pt 