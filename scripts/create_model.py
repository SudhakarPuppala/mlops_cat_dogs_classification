#!/usr/bin/env python3
"""
Create a dummy model for demonstration purposes
"""
import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import SimpleCNN

def main():
    """Create and save a dummy model"""
    print("Creating dummy model for demonstration...")
    
    # Create model
    model = SimpleCNN(num_classes=2)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/model.pt'
    torch.save(model.state_dict(), model_path)
    
    print(f"✓ Dummy model saved to {model_path}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nNote: This is an untrained model for demonstration purposes only.")
    print("To use a trained model, run: python scripts/train_model.py")

if __name__ == '__main__':
    main()
