# src/train_baseline.py
"""
Training pipeline for centralized (non-federated) baseline model.
This is our comparison benchmark - standard ML training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
from src.config import *
from src.model import get_model
from src.data_loader import get_data_loaders

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch (one pass through all training data).
    
    Training loop:
    1. Get batch of images and labels
    2. Forward pass: model predicts
    3. Calculate loss (how wrong the predictions are)
    4. Backward pass: compute gradients
    5. Update weights to reduce loss
    
    Args:
        model: Neural network
        train_loader: DataLoader with training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimization algorithm (Adam)
        device: CPU/GPU/MPS
    
    Returns:
        epoch_loss: Average loss across all batches
        epoch_acc: Training accuracy (%)
    """
    
    model.train()  # Set model to training mode (enables dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # tqdm creates a progress bar
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        # Move data to device (MPS/GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: get model predictions
        outputs = model(images)  # Shape: [batch_size, 2]
        
        # Calculate loss (how far predictions are from true labels)
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update model weights
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        
        # Get predicted class (0 or 1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate average loss and accuracy for this epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Evaluate model on validation set.
    
    No training happens here - just measure performance.
    Used to:
    1. Track if model is improving
    2. Detect overfitting (train acc high, val acc low)
    3. Save best model checkpoint
    
    Args:
        model: Neural network
        val_loader: DataLoader with validation data
        criterion: Loss function
        device: CPU/GPU/MPS
    
    Returns:
        val_loss: Average validation loss
        val_acc: Validation accuracy (%)
    """
    
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # torch.no_grad() disables gradient computation (saves memory, faster)
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass only (no backward pass in validation)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def train_baseline_model(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    """
    Complete training pipeline for baseline model.
    
    Steps:
    1. Load data
    2. Create model
    3. Define loss function and optimizer
    4. Training loop (multiple epochs)
    5. Save best model
    6. Evaluate on test set
    
    Args:
        num_epochs: How many times to go through full dataset
        batch_size: Images per batch
        learning_rate: Step size for weight updates
    
    Returns:
        model: Trained model
        history: Training metrics (loss, accuracy per epoch)
        test_acc: Final test set accuracy
    """
    
    print("\n" + "="*60)
    print("STARTING BASELINE MODEL TRAINING")
    print("="*60)
    
    # Setup
    device = DEVICE
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)
    
    # Create model
    print("\nCreating model...")
    model = get_model(pretrained=True)
    model = model.to(device)
    
    # Loss function: CrossEntropyLoss for classification
    # Combines softmax + negative log likelihood
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam (adaptive learning rate, works well in practice)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler: reduce LR when validation loss plateaus
    # This helps model converge better
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    
    # Training history
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    # Main training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model (highest validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(MODELS_DIR, 'baseline_best.pth')
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    # Save training history
    history_path = os.path.join(METRICS_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n✓ Saved training history to {history_path}")
    
    # Test evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'baseline_best.pth')))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return model, history, test_acc


# Example usage
if __name__ == "__main__":
    """
    Run training directly from this file.
    Usage: python src/train_baseline.py
    """
    model, history, test_acc = train_baseline_model()