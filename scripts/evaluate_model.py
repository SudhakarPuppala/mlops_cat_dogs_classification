#!/usr/bin/env python3
"""
Model evaluation script for Cats vs Dogs classifier
Evaluates trained model and outputs metrics to metrics/metrics.json
"""
import os
import sys
import torch
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )

from src.model import get_model
from src.data_preprocessing import prepare_data_loaders
from src.train import Trainer

def main():
    model_path = 'models/model.pt'
    data_dir = 'PetImages'
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Load dataset
    print("Preparing evaluation dataset...")
    all_image_paths = []
    all_labels = []
    class_labels = {'Cat': 0, 'Dog': 1}
    for class_name, label in class_labels.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_image_paths.append(os.path.join(class_dir, fname))
                all_labels.append(label)

    val_loader, _ = prepare_data_loaders(
        all_image_paths, all_labels,
        [], [],
        batch_size=batch_size
    )

    # Load model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluate
    trainer = Trainer(
        model=model,
        train_loader=None,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        num_epochs=1,
        experiment_name='cats-dogs-evaluation'
    )
    val_loss, val_acc, metrics = trainer.validate()

    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    metrics_path = 'metrics/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=4))

if __name__ == '__main__':
    main()
