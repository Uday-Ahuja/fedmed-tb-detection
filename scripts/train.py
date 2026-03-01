# scripts/train.py
"""
Simple script to start baseline training.
This is what you run to train your model.
"""

import sys
import os
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_baseline import train_baseline_model

def main():
    """Main training entry point"""
    
    print("="*70)
    print(" "*20 + "FEDMED BASELINE TRAINING")
    print("="*70)
    print("\nThis will train a centralized ResNet-50 model for TB detection.")
    print("Expected time: 20-30 minutes on M1 Mac")
    print("\nPress Ctrl+C to stop training at any time.\n")
    
    try:
        # Start training
        model, history, test_acc = train_baseline_model()
        
        print("\n" + "="*70)
        print(f"TRAINING SUCCESSFUL!")
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print("="*70)
        
        print("\nNext steps:")
        print("1. Run: python scripts/plot_results.py")
        print("2. Check results/plots/ for visualizations")
        print("3. Check results/models/ for saved model")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial results may be saved in results/")
    
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()