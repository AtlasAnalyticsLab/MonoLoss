# Using MonoLoss for finetuning

Finetuning ResNet50 and CLIP-ViT-B/32 with MonoLoss as a regularizer on ImageNet-1K, CIFAR-10, and CIFAR-100.


## Download pre-extracted CLIP-ViT-L/14 features
The pre-extracted features can be downloaded at this [link](https://drive.google.com/drive/folders/1lSKyASqXvL3Cp9e1TfptJQZE7d_KBh0f?usp=drive_link). Please place all of them in the folder `MonoLoss/finetuning/pre_extracted_features`.


## Run the finetuning experiments
The script for finetuning loops is in the file `finetuning/scripts/run_multiple_jobs.sh`.
Each loop runs all the experiments with different lambda for one model and one dataset. T

### ResNet50 on ImageNet-1K
```bash
BASE_PORT=29650
i=0
for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/imagenet/resnet.sh "${port}" "${lam}"
done
```

### CLIP-ViT-B/32 on ImageNet-1K
```bash
BASE_PORT=29650
i=0
for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/imagenet/clip_vit.sh "${port}" "${lam}"
done
```

### ResNet50 on CIFAR-100
```bash
BASE_PORT=29650
i=0
for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/cifar100/resnet.sh "${port}" "${lam}"
done
```

### CLIP-ViT-B/32 on CIFAR-100
```bash
BASE_PORT=29650
i=0
for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/cifar100/clip_vit.sh "${port}" "${lam}"
done
```

### ResNet50 on CIFAR-10
```bash
BASE_PORT=29650
i=0
for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/cifar10/resnet.sh "${port}" "${lam}"
done
```

### CLIP-ViT-B/32 on CIFAR-10
```bash
BASE_PORT=29650
i=0
for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/cifar10/clip_vit.sh "${port}" "${lam}"
done

