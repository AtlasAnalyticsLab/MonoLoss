"""Open Images dataset with images, captions, and labels."""

import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class OpenImagesDataset(Dataset):
    """Open Images dataset with images, captions, and multi-label annotations."""
    
    def __init__(self, dataset_dir, split='train', transform=None, check_images=False):
        """
        Args:
            dataset_dir (str): Root directory for the dataset
            split (str): Dataset split ('train', 'test', 'validation')
            transform (callable, optional): Optional transform to be applied on images
            check_images (bool): Whether to check if images exist during initialization
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        
        # Define paths
        self.img_dir = os.path.join(dataset_dir, 'images', split)
        self.cap_dir = os.path.join(dataset_dir, 'captions', split)
        self.labels_dir = os.path.join(dataset_dir, 'labels')
        
        # Find the simplified JSON file
        json_path = os.path.join(self.cap_dir, f'simplified_open_images_{split}_localized_narratives.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Simplified JSON file not found: {json_path}")
        
        # Load the caption data
        print(f"Loading caption data from {json_path}...")
        with open(json_path, 'r') as f:
            self.image_id_to_captions = json.load(f)
        
        # Load label data
        self._load_label_data()
        
        # Create a flat list of image_ids and caption indices for __getitem__
        self.samples = []
        for image_id, captions in self.image_id_to_captions.items():
            for caption_idx, _ in enumerate(captions):
                self.samples.append((image_id, caption_idx))
        
        if check_images:
            self._check_images()
    
    def _load_label_data(self):
        """Load class descriptions and annotations for the dataset."""
        print("Loading label data...")
        
        # Load class descriptions
        class_file = os.path.join(self.labels_dir, 'oidv7-class-descriptions-boxable.csv')
        if not os.path.exists(class_file):
            raise FileNotFoundError(f"Class descriptions file not found: {class_file}")
        
        class_df = pd.read_csv(class_file)
        self.label_to_class = {row['LabelName']: row['DisplayName'] 
                              for _, row in class_df.iterrows()}
        self.class_to_label = {v: k for k, v in self.label_to_class.items()}
        
        # Create ordered list of all classes for tensor conversion
        self.all_classes = sorted(list(self.label_to_class.values()))
        self.num_classes = len(self.all_classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.all_classes)}
        
        print(f"Total number of classes: {self.num_classes}")
        
        # Load annotations for the split
        annotations_file = os.path.join(self.labels_dir, 
                                       f'{self.split}-annotations-human-imagelabels-boxable.csv')
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        
        print(f"Loading annotations from {annotations_file}...")
        annotations_df = pd.read_csv(annotations_file)
        
        # Filter for positive labels (Confidence=1) if that column exists
        if 'Confidence' in annotations_df.columns:
            annotations_df = annotations_df[annotations_df['Confidence'] == 1]
        
        # Group labels by image ID
        self.image_to_labels = {}
        self.image_to_label_tensor = {}
        
        for image_id, group in annotations_df.groupby('ImageID'):
            label_names = group['LabelName'].tolist()
            # Convert label IDs to human-readable class names
            class_names = [self.label_to_class.get(label, label) for label in label_names]
            self.image_to_labels[image_id] = class_names
            
            # Create multi-hot encoding tensor for this image
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
        
        # Filter samples to only include valid images
        self.samples = [(image_id, caption_idx) for image_id, caption_idx in self.samples 
                        if image_id in valid_image_ids]
        
        print(f"Found {len(valid_image_ids)} valid images out of {len(unique_image_ids)}")
        print(f"Dataset now contains {len(self.samples)} valid image-caption pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: Dictionary containing image tensor, caption string, image ID, and labels
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image_id and caption_idx
        image_id, caption_idx = self.samples[idx]
        
        # Get caption data
        caption_data = self.image_id_to_captions[image_id][caption_idx]
        caption = caption_data['caption']
        
        # Get labels for this image (both text and tensor versions)
        labels_text = self.image_to_labels.get(image_id, [])
        
        # Get or create the multi-hot encoded label tensor
        if image_id in self.image_to_label_tensor:
            labels_tensor = self.image_to_label_tensor[image_id]
        else:
            # If image has no labels in our dataset, create empty tensor
            labels_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        
        # Load image
        image_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'captions': caption,
            'image_id': image_id,
            'labels_text': labels_text,
            'labels': labels_tensor,  # This is the multi-hot encoded tensor
            'idx': idx
        }

