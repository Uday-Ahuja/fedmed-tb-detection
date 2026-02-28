# scripts/explore_data.py
"""
Quick script to visualize the dataset.
Shows sample images and data distribution.
"""

import sys
import os
# Add project root to path so we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from PIL import Image
from src.data_loader import load_shenzhen_data
from src.config import PLOTS_DIR

def main():
    """Explore and visualize the dataset"""
    
    print("="*50)
    print("EXPLORING TB X-RAY DATASET")
    print("="*50)
    
    # Load data splits
    train_data, val_data, test_data = load_shenzhen_data()
    train_paths, train_labels = train_data
    
    # Print statistics
    total_images = len(train_paths) + len(val_data[0]) + len(test_data[0])
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total images: {total_images}")
    print(f"Training: {len(train_paths)} images")
    print(f"  - Normal: {train_labels.count(0)}")
    print(f"  - TB: {train_labels.count(1)}")
    print(f"Validation: {len(val_data[0])} images")
    print(f"Test: {len(test_data[0])} images")
    
    # Visualize 6 sample images (3 normal, 3 TB)
    print("\n" + "="*50)
    print("CREATING VISUALIZATION")
    print("="*50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sample TB X-ray Images', fontsize=16, fontweight='bold')
    
    # Get indices for normal and TB images
    normal_indices = [i for i, label in enumerate(train_labels) if label == 0]
    tb_indices = [i for i, label in enumerate(train_labels) if label == 1]
    
   # Plot 3 normal images (top row)
    valid_normal = []
    for idx in normal_indices:
        try:
            img = Image.open(train_paths[idx])
            valid_normal.append((idx, img))
            if len(valid_normal) == 3:
                break
        except:
            continue

    for i, (idx, img) in enumerate(valid_normal):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title('Normal', fontsize=14, color='green')
        axes[0, i].axis('off')

    # Plot 3 TB images (bottom row)
    valid_tb = []
    for idx in tb_indices:
        try:
            img = Image.open(train_paths[idx])
            valid_tb.append((idx, img))
            if len(valid_tb) == 3:
                break
        except:
            continue

    for i, (idx, img) in enumerate(valid_tb):
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title('Tuberculosis', fontsize=14, color='red')
        axes[1, i].axis('off')

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(PLOTS_DIR, 'data_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")

    # Also show it
    plt.show()

    print("\n" + "="*50)
    print("EXPLORATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()