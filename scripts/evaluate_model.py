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

from src.model import get_model
from src.data_preprocessing import prepare_data_loaders
from src.train import Trainer

def main():
    model_path = 'models/model.pt'
    data_dir = 'PetImages'
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 10  # Set number of evaluation epochs

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

    # Evaluate for multiple epochs (simulate for DVC plots)
    metrics_list = []
    for epoch in range(1, num_epochs + 1):
        val_loss, val_acc, metrics = Trainer(
            model=model,
            train_loader=None,
            val_loader=val_loader,
            device=device,
            learning_rate=0.001,
            num_epochs=1,
            experiment_name='cats-dogs-evaluation'
        ).validate()

        # Convert confusion_matrix to list for JSON serialization
        if 'confusion_matrix' in metrics:
            import numpy as np
            if isinstance(metrics['confusion_matrix'], np.ndarray):
                metrics['confusion_matrix'] = metrics['confusion_matrix'].tolist()

        # Add epoch and loss/accuracy keys
        metrics_record = {
            'epoch': epoch,
            'val_loss': val_loss,
            'val_accuracy': metrics.get('accuracy', None),
            'val_precision': metrics.get('precision', None),
            'val_recall': metrics.get('recall', None),
            'val_f1_score': metrics.get('f1_score', None),
            'confusion_matrix': metrics.get('confusion_matrix', None)
        }
        metrics_list.append(metrics_record)

    # Save metrics as a list
    os.makedirs('metrics', exist_ok=True)
    metrics_path = 'metrics/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_list, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_path}")
    print(json.dumps(metrics_list, indent=4))

if __name__ == '__main__':
    main()
