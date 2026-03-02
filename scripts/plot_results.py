# scripts/plot_results.py
"""
Visualize training results and model performance.
Creates publication-quality plots for your report.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import METRICS_DIR, PLOTS_DIR

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 10

def plot_training_history(history):
    """
    Create training curves showing loss and accuracy over epochs.
    
    These plots help you understand:
    1. Is the model learning? (loss should decrease)
    2. Is it overfitting? (train acc high, val acc low)
    3. Has it converged? (curves flatten)
    
    Args:
        history: Dictionary with train/val loss and accuracy per epoch
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for best validation loss
    best_val_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    best_val_loss = min(history['val_loss'])
    ax1.annotate(f'Best Val Loss: {best_val_loss:.4f}\nEpoch {best_val_epoch}',
                xy=(best_val_epoch, best_val_loss),
                xytext=(best_val_epoch + 1, best_val_loss + 0.05),
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=6)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for best validation accuracy
    best_val_acc_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
    best_val_acc = max(history['val_acc'])
    ax2.annotate(f'Best Val Acc: {best_val_acc:.2f}%\nEpoch {best_val_acc_epoch}',
                xy=(best_val_acc_epoch, best_val_acc),
                xytext=(best_val_acc_epoch + 1, best_val_acc - 5),
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(PLOTS_DIR, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves to: {output_path}")
    
    return fig


def plot_final_comparison(history):
    """
    Create bar chart comparing final train vs validation performance.
    Useful for quickly seeing the final results.
    
    Args:
        history: Training history dictionary
    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get final epoch metrics
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    # Create bar chart
    categories = ['Training', 'Validation']
    accuracies = [final_train_acc, final_val_acc]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Final Model Performance', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at target accuracy (85%)
    ax.axhline(y=85, color='green', linestyle='--', linewidth=2, label='Target: 85%')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(PLOTS_DIR, 'final_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved final comparison to: {output_path}")
    
    return fig


def generate_report(history):
    """
    Generate text report summarizing training results.
    Saves to results/metrics/training_report.txt
    
    Args:
        history: Training history dictionary
    """
    
    report_path = os.path.join(METRICS_DIR, 'training_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(" "*15 + "TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Training summary
        f.write("TRAINING SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total epochs: {len(history['train_loss'])}\n")
        f.write(f"Final train loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final train accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Final val loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"Final val accuracy: {history['val_acc'][-1]:.2f}%\n\n")
        
        # Best validation performance
        best_val_acc = max(history['val_acc'])
        best_val_epoch = history['val_acc'].index(best_val_acc) + 1
        best_val_loss = history['val_loss'][best_val_epoch - 1]
        
        f.write("BEST VALIDATION PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best val accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Best val loss: {best_val_loss:.4f}\n")
        f.write(f"Achieved at epoch: {best_val_epoch}\n\n")
        
        # Overfitting analysis
        train_val_gap = history['train_acc'][-1] - history['val_acc'][-1]
        f.write("OVERFITTING ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train-Val accuracy gap: {train_val_gap:.2f}%\n")
        if train_val_gap < 5:
            f.write("Status: ✓ Good generalization (gap < 5%)\n")
        elif train_val_gap < 10:
            f.write("Status: ⚠ Moderate overfitting (5% < gap < 10%)\n")
        else:
            f.write("Status: ✗ Significant overfitting (gap > 10%)\n")
        f.write("\n")
        
        # Convergence check
        last_3_val_acc = history['val_acc'][-3:]
        val_acc_std = sum((x - sum(last_3_val_acc)/3)**2 for x in last_3_val_acc) ** 0.5
        
        f.write("CONVERGENCE CHECK:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Last 3 epochs val acc std: {val_acc_std:.2f}%\n")
        if val_acc_std < 1:
            f.write("Status: ✓ Converged (std < 1%)\n")
        else:
            f.write("Status: ⚠ Still improving (std >= 1%)\n")
        f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("Report generated successfully.\n")
        f.write("Check results/plots/ for visualizations.\n")
        f.write("="*60 + "\n")
    
    print(f"✓ Saved training report to: {report_path}")


def main():
    """Main function to generate all plots and reports"""
    
    print("\n" + "="*60)
    print(" "*15 + "GENERATING RESULTS")
    print("="*60 + "\n")
    
    # Load training history
    history_path = os.path.join(METRICS_DIR, 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"ERROR: Training history not found at {history_path}")
        print("Please run training first: python scripts/train.py")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print("Loaded training history")
    print(f"  Epochs: {len(history['train_loss'])}")
    print(f"  Final train acc: {history['train_acc'][-1]:.2f}%")
    print(f"  Final val acc: {history['val_acc'][-1]:.2f}%\n")
    
    # Generate plots
    print("Creating visualizations...")
    plot_training_history(history)
    plot_final_comparison(history)
    
    # Generate text report
    print("\nGenerating report...")
    generate_report(history)
    
    print("\n" + "="*60)
    print("ALL RESULTS GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"\nCheck these locations:")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Metrics: {METRICS_DIR}")
    print(f"  Model: {os.path.join(RESULTS_DIR, 'models')}")
    print("\nNext steps:")
    print("1. Review training curves in results/plots/")
    print("2. Read training report in results/metrics/")
    print("3. Use these for your Review 2 presentation")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
