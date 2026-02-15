#!/usr/bin/env python3
"""
Main training script for Cats vs Dogs classifier
"""
import os
import sys
import argparse
import torch
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.train import Trainer
from src.data_preprocessing import prepare_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs classifier')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--model-save-path', type=str, default='models/model.pt',
                       help='Path to save the trained model')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # For demonstration, create dummy data loaders
    # In real scenario, you would load actual data
    logger.warning("Using dummy data for demonstration. Replace with actual data loading.")
    
    # Create dummy data
    import numpy as np
    num_train = 100
    num_val = 20
    
    train_paths = [f"dummy_train_{i}.jpg" for i in range(num_train)]
    train_labels = [i % 2 for i in range(num_train)]
    val_paths = [f"dummy_val_{i}.jpg" for i in range(num_val)]
    val_labels = [i % 2 for i in range(num_val)]
    
    # Create data loaders (this will fail with dummy paths, but shows the structure)
    try:
        train_loader, val_loader = prepare_data_loaders(
            train_paths, train_labels,
            val_paths, val_labels,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        logger.info("Creating minimal dummy loaders for demonstration")
        
        # Create minimal dummy loaders
        from torch.utils.data import TensorDataset, DataLoader
        dummy_train_data = torch.randn(num_train, 3, 224, 224)
        dummy_train_labels = torch.tensor(train_labels)
        dummy_val_data = torch.randn(num_val, 3, 224, 224)
        dummy_val_labels = torch.tensor(val_labels)
        
        train_dataset = TensorDataset(dummy_train_data, dummy_train_labels)
        val_dataset = TensorDataset(dummy_val_data, dummy_val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = get_model(num_classes=2)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        experiment_name='cats-dogs-classification'
    )
    
    # Train model
    logger.info("Starting training...")
    results = trainer.train(model_save_path=args.model_save_path)
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    logger.info(f"Model saved to: {args.model_save_path}")


if __name__ == '__main__':
    main()
