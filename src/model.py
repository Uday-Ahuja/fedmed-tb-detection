"""
Neural network model definition for TB detection.
Uses ResNet-50 pretrained on ImageNet, fine-tuned for our binary classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES

def get_model(pretrained=True):
    """
    Create ResNet-50 model for TB detection.
    
    ResNet-50:
    - 50 layers deep (very powerful feature extractor)
    - Pretrained on ImageNet (1.2M general images)
    - We modify final layer for our 2-class problem (Normal vs TB)
    
    Args:
        pretrained: If True, load weights from ImageNet training.
                   This gives us a head start - model already knows
                   how to recognize edges, shapes, textures.
    
    Returns:
        PyTorch model ready for training
    """
    
    print(f"Loading ResNet-50 (pretrained={pretrained})...")
    
    # Load pretrained ResNet-50 from torchvision
    # weights='IMAGENET1K_V1' is the standard pretrained weights
    if pretrained:
        model = models.resnet50(weights='IMAGENET1K_V1')
    else:
        model = models.resnet50(weights=None)
    
    # ResNet-50 architecture:
    # - Input: 224x224x3 image
    # - 50 convolutional layers extract features
    # - Global average pooling reduces to 2048 features
    # - Final fully connected (fc) layer: 2048 -> 1000 classes (ImageNet)
    
    # We need to modify the final layer for our task:
    # Original: 2048 features -> 1000 classes (ImageNet)
    # Our task: 2048 features -> 2 classes (Normal/TB)
    
    num_features = model.fc.in_features  # Get input size (2048)
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    print(f"✓ Model created: ResNet-50 with {NUM_CLASSES}-class output")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def get_model_summary(model):
    """
    Print model architecture summary.
    Useful for debugging and understanding the network structure.
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")


# Example usage (for testing)
if __name__ == "__main__":
    """
    Test script to verify model creation works.
    Run: python src/model.py
    """
    from src.config import DEVICE
    
    # Create model
    model = get_model(pretrained=True)
    model = model.to(DEVICE)
    
    # Print summary
    get_model_summary(model)
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)  # Batch of 1 image
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    print("\n✓ Model test passed!")