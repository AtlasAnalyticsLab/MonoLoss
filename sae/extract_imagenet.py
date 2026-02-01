"""Extract features from ImageNet-1k using various vision models and save to LMDB."""

import os
import gc
import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from datasets import load_dataset
import argparse

# Vision model imports
import open_clip  # For CLIP models
from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoProcessor,
    Dinov2Model,
    SwinModel,
)  # All use HuggingFace AutoImageProcessor for proper transforms


class ImageNetHFDataset(Dataset):
    """Wrapper around HuggingFace ImageNet-1k dataset."""

    def __init__(self, hf_dataset, transform=None, return_pil=False):
        """
        Args:
            hf_dataset: HuggingFace dataset (train or validation split)
            transform: Optional transform to apply to images
            return_pil: If True, return PIL images (for HF processors)
        """
        self.dataset = hf_dataset
        self.transform = transform
        self.return_pil = return_pil

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')  # Ensure RGB (some ImageNet images are grayscale)
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label,
            'idx': idx
        }


def pil_collate(batch):
    """Collate function that keeps PIL images (for models with their own preprocessing)."""
    return {
        'image': [b['image'] for b in batch],
        'label': torch.tensor([b['label'] for b in batch], dtype=torch.long),
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

    # Estimate map size (need extra space for LMDB B-tree overhead)
    sample_size = np.prod(feat_shape) * 4  # float32 = 4 bytes
    map_size = int(sample_size * num_samples * 2.5 + 1024**3)  # 2.5x + 1GB buffer

    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, feat in tqdm(features_dict.items(), desc="Writing to LMDB"):
            key = str(idx).encode('ascii')
            # Store as raw float32 bytes (NOT pickle!)
            value = feat.astype(np.float32).tobytes()
            txn.put(key, value)

        # Store metadata
        txn.put(b'__len__', str(num_samples).encode('ascii'))
        txn.put(b'__shape__', str(feat_shape).encode('ascii'))

    env.close()
    print(f"Saved {num_samples} samples to {lmdb_path}")


def extract_features_streaming(loader, model_fn, total_samples, device, desc="Extracting"):
    """
    Extract features in streaming fashion to avoid OOM.

    Args:
        loader: DataLoader
        model_fn: Function that takes batch and returns features (numpy array)
        total_samples: Total number of samples
        device: torch device
        desc: Description for progress bar

    Returns:
        features_dict: Dict mapping idx -> numpy array
        feat_shape: Shape of feature vectors
    """
    features_dict = {}
    feat_shape = None

    for batch in tqdm(loader, desc=desc):
        idxs = batch['idx'].numpy()

        with torch.no_grad():
            feats = model_fn(batch)  # Returns numpy array

        if feat_shape is None:
            feat_shape = tuple(feats.shape[1:])

        for i, idx in enumerate(idxs):
            features_dict[int(idx)] = feats[i]

    return features_dict, feat_shape


def extract_dino_features(hf_dataset, split, out_dir, model_name, tag, batch_size, device):
    """Extract DINOv2 features using HuggingFace transformers."""
    # Use HuggingFace processor and model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name).to(device).eval()

    loader = DataLoader(
        ImageNetHFDataset(hf_dataset, transform=None),  # Keep PIL, processor handles transform
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=pil_collate
    )

    def model_fn(batch):
        pil_imgs = batch['image']
        inputs = processor(images=pil_imgs, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            outputs = model(**inputs)
            # Use CLS token (first token of last_hidden_state)
            feats = outputs.last_hidden_state[:, 0, :]
        return feats.cpu().float().numpy()

    features_dict, feat_shape = extract_features_streaming(
        loader, model_fn, len(hf_dataset), device, f'DINO ({tag})'
    )

    lmdb_path = str(out_dir / f'dino_features_{tag}.lmdb')
    write_to_lmdb(lmdb_path, features_dict, feat_shape)

    del model; torch.cuda.empty_cache(); gc.collect()


def extract_clip_features(hf_dataset, split, out_dir, model_name, tag, batch_size, device):
    """Extract CLIP image features."""
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    model = model.to(device).eval()

    # Create dataset with CLIP preprocessing
    loader = DataLoader(
        ImageNetHFDataset(hf_dataset, preprocess),
        batch_size=batch_size, shuffle=False, num_workers=4
    )

    def model_fn(batch):
        imgs = batch['image'].to(device)
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            feats = model.encode_image(imgs)
        return feats.cpu().float().numpy()

    features_dict, feat_shape = extract_features_streaming(
        loader, model_fn, len(hf_dataset), device, f'CLIP ({tag})'
    )

    lmdb_path = str(out_dir / f'clip_image_features_{tag}.lmdb')
    write_to_lmdb(lmdb_path, features_dict, feat_shape)

    del model; torch.cuda.empty_cache(); gc.collect()


def resolve_hf_cache_path(model_name):
    """Resolve HuggingFace model name to local cache snapshot path."""
    import os
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    model_dir = model_name.replace('/', '--')
    snapshot_dir = os.path.join(hf_home, 'hub', f'models--{model_dir}', 'snapshots')
    if os.path.exists(snapshot_dir):
        snapshots = os.listdir(snapshot_dir)
        if snapshots:
            return os.path.join(snapshot_dir, snapshots[0])
    return model_name  # Fallback to original name


def extract_siglip2_features(hf_dataset, split, out_dir, model_name, tag, batch_size, device):
    """Extract SigLIP2 image features."""
    # Use local cache path to avoid tokenizer network calls in offline mode
    local_path = resolve_hf_cache_path(model_name)
    model = AutoModel.from_pretrained(local_path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(local_path)

    loader = DataLoader(
        ImageNetHFDataset(hf_dataset, transform=None),  # Keep PIL images
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=pil_collate
    )

    def model_fn(batch):
        pil_imgs = batch['image']
        inputs = processor(images=pil_imgs, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            feats = model.get_image_features(pixel_values=inputs['pixel_values'])
        return feats.cpu().float().numpy()

    features_dict, feat_shape = extract_features_streaming(
        loader, model_fn, len(hf_dataset), device, f'SigLIP2 ({tag})'
    )

    lmdb_path = str(out_dir / f'siglip2_image_features_{tag}.lmdb')
    write_to_lmdb(lmdb_path, features_dict, feat_shape)

    del model; torch.cuda.empty_cache(); gc.collect()


def extract_vit_features(hf_dataset, split, out_dir, model_name, tag, batch_size, device):
    """Extract ViT features (CLS token) using HuggingFace transformers."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    loader = DataLoader(
        ImageNetHFDataset(hf_dataset, transform=None),  # Keep PIL, processor handles transform
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=pil_collate
    )

    def model_fn(batch):
        pil_imgs = batch['image']
        inputs = processor(images=pil_imgs, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            outputs = model(**inputs)
            # CLS token (first token of last_hidden_state)
            feats = outputs.last_hidden_state[:, 0, :]
        return feats.cpu().float().numpy()

    features_dict, feat_shape = extract_features_streaming(
        loader, model_fn, len(hf_dataset), device, f'ViT ({tag})'
    )

    lmdb_path = str(out_dir / f'vit_features_{tag}.lmdb')
    write_to_lmdb(lmdb_path, features_dict, feat_shape)

    del model; torch.cuda.empty_cache(); gc.collect()


def extract_swin_features(hf_dataset, split, out_dir, model_name, tag, batch_size, device):
    """Extract Swin Transformer features (pooler output) using HuggingFace transformers."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SwinModel.from_pretrained(model_name).to(device).eval()

    loader = DataLoader(
        ImageNetHFDataset(hf_dataset, transform=None),  # Keep PIL, processor handles transform
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=pil_collate
    )

    def model_fn(batch):
        pil_imgs = batch['image']
        inputs = processor(images=pil_imgs, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            outputs = model(**inputs)
            # Swin uses pooler_output (adaptive avg pool of last hidden state)
            feats = outputs.pooler_output
        return feats.cpu().float().numpy()

    features_dict, feat_shape = extract_features_streaming(
        loader, model_fn, len(hf_dataset), device, f'Swin ({tag})'
    )

    lmdb_path = str(out_dir / f'swin_features_{tag}.lmdb')
    write_to_lmdb(lmdb_path, features_dict, feat_shape)

    del model; torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ImageNet features to LMDB")
    parser.add_argument('--model', choices=['dinov2', 'clip', 'siglip2', 'vit', 'swin'], required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--output_dir', type=str, default='./features/imagenet')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Model configurations: (hf_model_id, output_tag)
    # All models use HuggingFace AutoImageProcessor for transforms
    MODEL_CONFIGS = {
        'dinov2': ('facebook/dinov2-large', 'dinov2-large'),
        'clip': ('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'CLIP-ViT-L-14-DataComp'),
        'siglip2': ('google/siglip2-so400m-patch14-384', 'siglip2-so400m-p14-384'),
        'vit': ('google/vit-large-patch16-224', 'vit-large-p16-224'),
        'swin': ('microsoft/swin-large-patch4-window12-384', 'swin-large-p4-w12-384'),
    }

    # Load ImageNet from HuggingFace
    print(f"Loading ImageNet-1k {args.split} split from HuggingFace...")
    hf_dataset = load_dataset("imagenet-1k", split=args.split, trust_remote_code=True)
    print(f"Loaded {len(hf_dataset)} samples")

    # Create output directory
    OUT_DIR = Path(args.output_dir) / args.split
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_name, tag = MODEL_CONFIGS[args.model]

    if args.model == 'dinov2':
        extract_dino_features(hf_dataset, args.split, OUT_DIR, model_name, tag, args.batch_size, DEVICE)
    elif args.model == 'clip':
        extract_clip_features(hf_dataset, args.split, OUT_DIR, model_name, tag, args.batch_size, DEVICE)
    elif args.model == 'siglip2':
        extract_siglip2_features(hf_dataset, args.split, OUT_DIR, model_name, tag, args.batch_size, DEVICE)
    elif args.model == 'vit':
        extract_vit_features(hf_dataset, args.split, OUT_DIR, model_name, tag, args.batch_size, DEVICE)
    elif args.model == 'swin':
        extract_swin_features(hf_dataset, args.split, OUT_DIR, model_name, tag, args.batch_size, DEVICE)

    print("Done!")
