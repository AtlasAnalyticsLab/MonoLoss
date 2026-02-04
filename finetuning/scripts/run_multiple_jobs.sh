BASE_PORT=29650
i=0

# ImageNet
for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/imagenet/resnet.sh "${port}" "${lam}"
done

for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/imagenet/clip_vit.sh "${port}"
done

# CIFAR-100
for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/cifar100/resnet.sh "${port}" "${lam}"
done

for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/cifar100/clip_vit.sh "${port}"
done

# CIFAR-10
for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/cifar10/resnet.sh "${port}" "${lam}"
done

for lam in 0.0 0.03 0.05 0.07 0.1; do
  port=$((BASE_PORT + i))
  i=$((i + 1))
  sh scripts/cifar10/clip_vit.sh "${port}"
done