#!/usr/bin/env python3
"""
Main training script for Cats vs Dogs classifier
Loads actual dataset from PetImages directory
"""
import os
import sys
import argparse
import torch
from pathlib import Path
import logging
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split

import random
import sys
from dvclive import Live

with Live(save_dvc_exp=True) as live:
    epochs = int(sys.argv[1])
    live.log_param("epochs", epochs)
    for epoch in range(epochs):
        live.log_metric("train/accuracy", epoch + random.random())
        live.log_metric("train/loss", epochs - epoch - random.random())
        live.log_metric("val/accuracy",epoch + random.random() )
        live.log_metric("val/loss", epochs - epoch - random.random())
        live.next_step()
        
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.train import Trainer
from src.data_preprocessing import prepare_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset_from_petimages(
    root_dir: str = 'PetImages',
    train_split: float = 0.8,
    random_seed: int = 42
) -> tuple:
    """
    Load dataset from PetImages directory structure.
    Handles corrupted images gracefully.
    
    Args:
        root_dir: Path to PetImages root directory
        train_split: Fraction of data to use for training (default 0.8 for 80% train, 20% val)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels)
    """
    logger.info(f"Loading dataset from {root_dir}...")
    
    all_image_paths = []
    all_labels = []
    
    # Class labels: 0 for Cat, 1 for Dog
    class_labels = {'Cat': 0, 'Dog': 1}
    class_stats = defaultdict(int)
    corrupted_images = defaultdict(int)
    
    # Scan each class directory
    for class_name, label in class_labels.items():
        class_dir = os.path.join(root_dir, class_name)
        
        if not os.path.exists(class_dir):
            logger.warning(f"Directory not found: {class_dir}")
            continue
            
        logger.info(f"\nProcessing {class_name} images from {class_dir}...")
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        image_files = [
            f for f in os.listdir(class_dir) 
            if os.path.isfile(os.path.join(class_dir, f)) and 
            f.lower().endswith(image_extensions)
        ]
        
        logger.info(f"Found {len(image_files)} image files in {class_name} directory")
        
        # Validate and load image paths
        valid_count = 0
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(class_dir, image_file)
            
            try:
                # Try to open and verify the image
                with Image.open(image_path) as img:
                    # Verify it's a valid image with content
                    if img.size[0] > 0 and img.size[1] > 0:
                        all_image_paths.append(image_path)
                        all_labels.append(label)
                        valid_count += 1
            except (IOError, OSError, Exception) as e:
                corrupted_images[class_name] += 1
                if idx < 5:  # Log only first few corrupted images
                    logger.debug(f"Corrupted image skipped: {image_path} - {str(e)}")
            
            # Progress update every 1000 images
            if (idx + 1) % 1000 == 0:
                logger.info(f"  Processed {idx + 1}/{len(image_files)} {class_name} images...")
        
        class_stats[class_name] = valid_count
        logger.info(f"✓ {class_name}: {valid_count} valid images, {corrupted_images[class_name]} corrupted")
    
    # Print dataset statistics
    total_images = sum(class_stats.values())
    total_corrupted = sum(corrupted_images.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Statistics:")
    logger.info(f"{'='*60}")
    logger.info(f"Total valid images: {total_images}")
    logger.info(f"Total corrupted images: {total_corrupted}")
    for class_name, count in class_stats.items():
        logger.info(f"  {class_name}: {count} images")
    logger.info(f"{'='*60}\n")
    
    if total_images == 0:
        logger.error("No valid images found in PetImages directory!")
        raise ValueError("Dataset is empty")
    
    # Split data into train and validation sets
    logger.info(f"Splitting data: {int(train_split*100)}% train, {int((1-train_split)*100)}% validation...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths,
        all_labels,
        test_size=1 - train_split,
        random_state=random_seed,
        stratify=all_labels  # Ensure balanced split between classes
    )
    
    logger.info(f"Training set: {len(train_paths)} images")
    logger.info(f"Validation set: {len(val_paths)} images")
    
    # Verify split balance
    train_cat_count = sum(1 for label in train_labels if label == 0)
    train_dog_count = sum(1 for label in train_labels if label == 1)
    val_cat_count = sum(1 for label in val_labels if label == 0)
    val_dog_count = sum(1 for label in val_labels if label == 1)
    
    logger.info(f"\nTrain set balance - Cats: {train_cat_count}, Dogs: {train_dog_count}")
    logger.info(f"Val set balance - Cats: {val_cat_count}, Dogs: {val_dog_count}\n")
    
    return train_paths, train_labels, val_paths, val_labels


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs classifier')
    parser.add_argument('--data-dir', type=str, default='PetImages',
                       help='Directory containing PetImages dataset (Cat/ and Dog/ subdirectories)')
    parser.add_argument('--model-save-path', type=str, default='models/model.pt',
                       help='Path to save the trained model')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction of data to use for training (default 0.8)')
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
    
    # Load actual dataset from PetImages directory
    logger.info(f"\nLoading dataset from '{args.data_dir}'...")
    try:
        train_paths, train_labels, val_paths, val_labels = load_dataset_from_petimages(
            root_dir=args.data_dir,
            train_split=args.train_split
        )
    except ValueError as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Create data loaders with actual images
    logger.info("\nPreparing data loaders...")
    try:
        train_loader, val_loader = prepare_data_loaders(
            train_paths, train_labels,
            val_paths, val_labels,
            batch_size=args.batch_size
        )
        logger.info("✓ Data loaders created successfully!")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create model
    logger.info("\nInitializing model...")
    model = get_model(num_classes=2)
    model_info = model.get_model_info()
    logger.info(f"Model: {model_info['model_name']}")
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Create trainer
    logger.info("\nInitializing trainer...")
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
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60)
    try:
        results = trainer.train(model_save_path=args.model_save_path)
        
        logger.info("\n" + "="*60)
        logger.info("Training completed!")
        logger.info("="*60)
        logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
        logger.info(f"Model saved to: {args.model_save_path}")
        logger.info("="*60 + "\n")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return



if __name__ == '__main__':
    main()
