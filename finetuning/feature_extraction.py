from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
from utils import FolderDatasetWithDir

import glob
import torch
from torch.utils.data import DataLoader

import argparse
import tqdm

import os
os.environ['HF_HOME'] = '/data/add_disk1/anhng/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet_root_folder", type=str, default="/dev/shm/an",
        help="Path pattern to training image files."
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to use for feature extraction (e.g., 'train' or 'val')."
    )
    parser.add_argument(
        "--model", type=str, default="resnet50", help="Model architecture to use for feature extraction."
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for feature extraction."
    )
    parser.add_argument(
        "--output_path", type=str, default="/home/anhnguyen/MonoLoss/finetuning/pre_extracted_features",
        help="Path to save extracted features."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    torch.hub.set_dir('/project/rrg-msh/anhnguyen/.cache')

    if args.split == "train":
        args.file_pattern = f"{args.imagenet_root_folder}/train/*/*.JPEG"
    elif args.split == "val":
        args.file_pattern = f"{args.imagenet_root_folder}/val/*/*.JPEG"

    # Initialize model with the best available weights
    if args.model == "resnet50":
        print("Using ResNet50 model for feature extraction")
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        preprocess = weights.transforms()
    elif args.model == 'clip_vit_large_patch14_336':
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336", use_fast=True)
    elif args.model == 'clip_vit_large_patch14':
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
    elif args.model == 'clip_vit_base_patch32':
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    else:
        raise ValueError(f"Model {args.model} not supported for feature extraction.")
    model.eval()
    model.to(args.device)

    # Step 1: Load validation image file paths
    val_dataset = FolderDatasetWithDir(file_pattern=args.file_pattern, transform=preprocess)
    print(f"Found {len(val_dataset)} images for feature extraction.")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Step 3: Apply inference preprocessing transforms
    batch_size = args.batch_size
    all_features = []
    all_paths = []
    with torch.no_grad():
        for images, paths in tqdm.tqdm(val_loader):
            images = images.to(args.device)
            if 'clip_vit' in args.model:
                features = model.get_image_features(images)
            else:
                features = model(images)
            all_features.append(features)
            all_paths.extend(paths)
    all_features = torch.cat(all_features, dim=0)

    # Step 4: Save extracted features
    output_path = f"{args.output_path}/imagenet_{args.split}_features_{args.model}.pt"
    torch.save({
        'features': all_features,
        'paths': all_paths
    }, output_path)
    print(f"Extracted features saved to {output_path} with shape {all_features.shape}")

if __name__ == "__main__":
    main()

