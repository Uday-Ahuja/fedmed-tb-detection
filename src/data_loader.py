# src/data_loader.py
"""
Data loading and preprocessing for TB X-ray dataset.
Handles image loading, augmentation, train/val/test splitting.
"""

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src.config import *

class TBDataset(Dataset):
    """
    Custom PyTorch Dataset for TB X-rays.
    
    Args:
        image_paths: List of file paths to images
        labels: List of labels (0=Normal, 1=TB)
        transform: torchvision transforms to apply
    """
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """Return total number of samples"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and return one sample (image, label).
        Called by DataLoader for each batch.
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB (3 channels)
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms (resize, augment, normalize)
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(train=True):
    """
    Get image transformations for preprocessing.
    
    Args:
        train: If True, apply data augmentation. If False, just resize/normalize.
    
    Returns:
        torchvision.transforms.Compose object
    """
    
    if train:
        # Training: Apply data augmentation to increase dataset variety
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to 224x224
            transforms.RandomHorizontalFlip(p=0.5),   # Flip 50% of images
            transforms.RandomRotation(10),             # Rotate ±10 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust colors
            transforms.ToTensor(),                     # Convert to PyTorch tensor
            transforms.Normalize(MEAN, STD)            # Normalize with ImageNet stats
        ])
    else:
        # Validation/Test: No augmentation, just resize and normalize
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])


def load_shenzhen_data(data_dir="data/raw/shenzhen"):
    """
    Load Shenzhen dataset and split into train/val/test.
    
    Filename format: CHNCXR_XXXX_Y.png
    where Y is 0 (Normal) or 1 (TB)
    
    Returns:
        train_data: (image_paths, labels) for training
        val_data: (image_paths, labels) for validation
        test_data: (image_paths, labels) for testing
    """
    
    print(f"Loading data from: {data_dir}")
    
    # Get all PNG files
    all_files = glob.glob(os.path.join(data_dir, "*.png"))
    print(f"Found {len(all_files)} total images")
    
    image_paths = []
    labels = []
    
    # Parse filenames to extract labels
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Extract label from filename: CHNCXR_XXXX_Y.png -> Y
        try:
            label = int(filename.split('_')[-1].split('.')[0])
            
            # Validate label (should be 0 or 1)
            if label in [0, 1]:
                image_paths.append(filepath)
                labels.append(label)
        except:
            print(f"Warning: Could not parse label from {filename}, skipping")
            continue
    
    print(f"Valid images: {len(image_paths)}")
    print(f"  Normal (0): {labels.count(0)}")
    print(f"  TB (1): {labels.count(1)}")
    
    # Split data: 70% train, 15% val, 15% test
    # stratify=labels ensures class balance in each split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=temp_labels
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_paths)} ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Val: {len(val_paths)} ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Test: {len(test_paths)} ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_data_loaders(data_dir="data/raw/shenzhen", batch_size=32):

    # Load and split data
    train_data, val_data, test_data = load_shenzhen_data(data_dir)

    train_dataset = TBDataset(
        train_data[0], train_data[1],
        transform=get_transforms(train=True)
    )

    val_dataset = TBDataset(
        val_data[0], val_data[1],
        transform=get_transforms(train=False)
    )

    test_dataset = TBDataset(
        test_data[0], test_data[1],
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader