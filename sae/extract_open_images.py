"""Extract vision features from OpenImages and save to LMDB format."""

import os
import gc
import json
import argparse
import lmdb
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import open_clip
from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoProcessor,
    AutoFeatureExtractor,
    SwinModel,
)


class OpenImagesDataset(Dataset):
    """OpenImages dataset with multi-label annotations."""

    def __init__(self, dataset_dir, split='train', transform=None, check_images=False):
        """
        Args:
            dataset_dir: Root directory for the dataset
            split: Dataset split ('train', 'test', 'validation')
            transform: Optional transform to be applied on images
            check_images: Whether to check if images exist during initialization
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(dataset_dir, 'images', split)
        self.cap_dir = os.path.join(dataset_dir, 'captions', split)
        self.labels_dir = os.path.join(dataset_dir, 'labels')

        json_path = os.path.join(self.cap_dir, f'simplified_open_images_{split}_localized_narratives.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Simplified JSON file not found: {json_path}")

        print(f"Loading caption data from {json_path}...")
        with open(json_path, 'r') as f:
            self.image_id_to_captions = json.load(f)

        self._load_label_data()

        self.samples = []
        for image_id, captions in self.image_id_to_captions.items():
            for caption_idx, _ in enumerate(captions):
                self.samples.append((image_id, caption_idx))

        if check_images:
            self._check_images()

    def _load_label_data(self):
        """Load class descriptions and annotations for the dataset."""
        print("Loading label data...")

        class_file = os.path.join(self.labels_dir, 'oidv7-class-descriptions-boxable.csv')
        if not os.path.exists(class_file):
            raise FileNotFoundError(f"Class descriptions file not found: {class_file}")

        class_df = pd.read_csv(class_file)
        self.label_to_class = {row['LabelName']: row['DisplayName']
                              for _, row in class_df.iterrows()}
        self.class_to_label = {v: k for k, v in self.label_to_class.items()}

        self.all_classes = sorted(list(self.label_to_class.values()))
        self.num_classes = len(self.all_classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.all_classes)}

        print(f"Total number of classes: {self.num_classes}")

        annotations_file = os.path.join(self.labels_dir,
                                       f'{self.split}-annotations-human-imagelabels-boxable.csv')
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

        print(f"Loading annotations from {annotations_file}...")
        annotations_df = pd.read_csv(annotations_file)

        if 'Confidence' in annotations_df.columns:
            annotations_df = annotations_df[annotations_df['Confidence'] == 1]

        self.image_to_labels = {}
        self.image_to_label_tensor = {}

        for image_id, group in annotations_df.groupby('ImageID'):
            label_names = group['LabelName'].tolist()
            class_names = [self.label_to_class.get(label, label) for label in label_names]
            self.image_to_labels[image_id] = class_names

            label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
            for class_name in class_names:
                if class_name in self.class_to_idx:
                    label_tensor[self.class_to_idx[class_name]] = 1.0

            self.image_to_label_tensor[image_id] = label_tensor

        print(f"Loaded labels for {len(self.image_to_labels)} images")

    def _check_images(self):
        """Check if all images in the dataset exist."""
        print("Checking image files...")
        unique_image_ids = set(image_id for image_id, _ in self.samples)

        valid_image_ids = set()
        for image_id in tqdm(unique_image_ids):
            image_path = os.path.join(self.img_dir, f"{image_id}.jpg")
            if os.path.isfile(image_path):
                valid_image_ids.add(image_id)

        self.samples = [(image_id, caption_idx) for image_id, caption_idx in self.samples
                        if image_id in valid_image_ids]

        print(f"Found {len(valid_image_ids)} valid images out of {len(unique_image_ids)}")
        print(f"Dataset now contains {len(self.samples)} valid image-caption pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id, caption_idx = self.samples[idx]
        caption_data = self.image_id_to_captions[image_id][caption_idx]
        caption = caption_data['caption']

        labels_text = self.image_to_labels.get(image_id, [])

        if image_id in self.image_to_label_tensor:
            labels_tensor = self.image_to_label_tensor[image_id]
        else:
            labels_tensor = torch.zeros(self.num_classes, dtype=torch.float32)

        image_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'captions': caption,
            'image_id': image_id,
            'labels_text': labels_text,
            'labels': labels_tensor,
            'idx': idx
        }


def pil_collate(batch):
    """Collate function that keeps PIL images."""
    return {
        'image': [b['image'] for b in batch],
        'captions': [b['captions'] for b in batch],
        'idx': torch.tensor([b['idx'] for b in batch], dtype=torch.long),
    }


def write_to_lmdb(lmdb_path, features_dict, feat_shape):
    """
    Write features to LMDB in raw bytes format (compatible with LMDBFeatureDataset).

    Args:
        lmdb_path: Path to LMDB directory
        features_dict: Dict mapping idx -> numpy array (float32)
        feat_shape: Shape of each feature vector (tuple)
    """
    num_samples = len(features_dict)

    sample_size = np.prod(feat_shape) * 4  # float32 = 4 bytes
    map_size = int(sample_size * num_samples * 2.5 + 1024**3)  # 2.5x + 1GB buffer

    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, feat in tqdm(features_dict.items(), desc="Writing to LMDB"):
            key = str(idx).encode('ascii')
            value = feat.astype(np.float32).tobytes()
            txn.put(key, value)

        txn.put(b'__len__', str(num_samples).encode('ascii'))
        txn.put(b'__shape__', str(feat_shape).encode('ascii'))

    env.close()
    print(f"Saved {num_samples} samples to {lmdb_path}")


def extract_features_streaming(loader, model_fn, total_samples, device, desc="Extracting"):
    """Extract features in streaming fashion."""
    features_dict = {}
    feat_shape = None

    for batch in tqdm(loader, desc=desc):
        idxs = batch['idx'].numpy()

        with torch.no_grad():
            feats = model_fn(batch)

        if feat_shape is None:
            feat_shape = tuple(feats.shape[1:])

        for i, idx in enumerate(idxs):
            features_dict[int(idx)] = feats[i]

    return features_dict, feat_shape


def extract_dino_features(dataset_dir, split, out_dir, model_name, tag, batch_size, device):
    """Extract DINOv2 features."""
    loader = DataLoader(
        OpenImagesDataset(dataset_dir, split, dino_transform),
        batch_size=batch_size, shuffle=False, num_workers=4
    )

    model = torch.hub.load('facebookresearch/dinov2', model_name).to(device).eval()

    def model_fn(batch):
        imgs = batch['image'].to(device)
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            feats = model(imgs)
        return feats.cpu().float().numpy()

    features_dict, feat_shape = extract_features_streaming(
        loader, model_fn, len(loader.dataset), device, f'DINO ({tag})'
    )

    lmdb_path = str(out_dir / f'dino_features_{tag}.lmdb')
    write_to_lmdb(lmdb_path, features_dict, feat_shape)

    del model; torch.cuda.empty_cache(); gc.collect()


def extract_clip_features(dataset_dir, split, out_dir, model_name, tag, batch_size, device):
    """Extract CLIP image features."""
    model, _, preprocess_clip = open_clip.create_model_and_transforms(model_name)
    model = model.to(device).eval()

    loader = DataLoader(
        OpenImagesDataset(dataset_dir, split),
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=pil_collate,
    )

    def model_fn(batch):
        pil_imgs = batch['image']
        img_tensor = torch.stack([preprocess_clip(im) for im in pil_imgs]).to(device)
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            feats = model.encode_image(img_tensor)
        return feats.cpu().float().numpy()

    features_dict, feat_shape = extract_features_streaming(
        loader, model_fn, len(loader.dataset), device, f'CLIP ({tag})'
    )

    lmdb_path = str(out_dir / f'clip_image_features_{tag}.lmdb')
    write_to_lmdb(lmdb_path, features_dict, feat_shape)

    del model; torch.cuda.empty_cache(); gc.collect()


def extract_siglip2_features(dataset_dir, split, out_dir, model_name, tag, batch_size, device):
    """Extract SigLIP2 image features."""
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    loader = DataLoader(
        OpenImagesDataset(dataset_dir, split),
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=pil_collate,
    )

    def model_fn(batch):
        pil_imgs = batch['image']
        inputs = processor(images=pil_imgs, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            feats = model.get_image_features(pixel_values=inputs['pixel_values'])
        return feats.cpu().float().numpy()

    features_dict, feat_shape = extract_features_streaming(
        loader, model_fn, len(loader.dataset), device, f'SigLIP2 ({tag})'
    )

    lmdb_path = str(out_dir / f'siglip2_image_features_{tag}.lmdb')
    write_to_lmdb(lmdb_path, features_dict, feat_shape)

    del model; torch.cuda.empty_cache(); gc.collect()


def extract_vit_features(dataset_dir, split, out_dir, model_name, tag, batch_size, device):
    """Extract ViT features (CLS token)."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(device).eval()

    loader = DataLoader(
        OpenImagesDataset(dataset_dir, split),
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=pil_collate,
    )

    def model_fn(batch):
        pil_imgs = batch['image']
        inputs = processor(images=pil_imgs, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :]
        return feats.cpu().float().numpy()

    features_dict, feat_shape = extract_features_streaming(
        loader, model_fn, len(loader.dataset), device, f'ViT ({tag})'
    )

    lmdb_path = str(out_dir / f'vit_features_{tag}.lmdb')
    write_to_lmdb(lmdb_path, features_dict, feat_shape)

    del model; torch.cuda.empty_cache(); gc.collect()


def extract_swin_features(dataset_dir, split, out_dir, model_name, tag, batch_size, device):
    """Extract Swin Transformer features (pooler output)."""
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = SwinModel.from_pretrained(model_name).to(device).eval()

    loader = DataLoader(
        OpenImagesDataset(dataset_dir, split),
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=pil_collate,
    )

    def model_fn(batch):
        pil_imgs = batch['image']
        inputs = processor(images=pil_imgs, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            outputs = model(**inputs)
            feats = outputs.pooler_output
        return feats.cpu().float().numpy()

    features_dict, feat_shape = extract_features_streaming(
        loader, model_fn, len(loader.dataset), device, f'Swin ({tag})'
    )

    lmdb_path = str(out_dir / f'swin_features_{tag}.lmdb')
    write_to_lmdb(lmdb_path, features_dict, feat_shape)

    del model; torch.cuda.empty_cache(); gc.collect()


# Transforms
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class MaybeToTensor(T.ToTensor):
    def __call__(self, pic):
        return pic if isinstance(pic, torch.Tensor) else super().__call__(pic)

normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
dino_transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    MaybeToTensor(),
    normalize,
])


# Model configurations
MODEL_CONFIGS = {
    'dinov2': ('dinov2_vitl14_reg', 'dinov2_vitl14_reg'),
    'clip': ('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'CLIP-ViT-L-14-DataComp'),
    'siglip2': ('google/siglip2-so400m-patch14-384', 'siglip2-so400m-p14-384'),
    'vit': ('google/vit-large-patch16-224', 'vit-large-p16-224'),
    'swin': ('microsoft/swin-large-patch4-window12-384', 'swin-large-p4-w12-384'),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract OpenImages features to LMDB")
    parser.add_argument('--model', choices=['dinov2', 'clip', 'siglip2', 'vit', 'swin'], required=True)
    parser.add_argument('--split', choices=['train', 'test', 'validation'], required=True)
    parser.add_argument('--dataset_dir', type=str, default='/home/atlas-gp/OpenImages/',
                       help='Path to OpenImages dataset')
    parser.add_argument('--output_dir', type=str, default='./features/open_images',
                       help='Output directory for features')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    OUT_DIR = Path(args.output_dir) / args.split
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_name, tag = MODEL_CONFIGS[args.model]

    if args.model == 'dinov2':
        extract_dino_features(args.dataset_dir, args.split, OUT_DIR, model_name, tag, args.batch_size, DEVICE)
    elif args.model == 'clip':
        extract_clip_features(args.dataset_dir, args.split, OUT_DIR, model_name, tag, args.batch_size, DEVICE)
    elif args.model == 'siglip2':
        extract_siglip2_features(args.dataset_dir, args.split, OUT_DIR, model_name, tag, args.batch_size, DEVICE)
    elif args.model == 'vit':
        extract_vit_features(args.dataset_dir, args.split, OUT_DIR, model_name, tag, args.batch_size, DEVICE)
    elif args.model == 'swin':
        extract_swin_features(args.dataset_dir, args.split, OUT_DIR, model_name, tag, args.batch_size, DEVICE)

    print("Done!")
