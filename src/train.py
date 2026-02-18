"""
Training script with MLflow experiment tracking
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for model training with MLflow tracking"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        experiment_name: str = 'cats-dogs-classification'
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            experiment_name: MLflow experiment name
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.experiment_name = experiment_name
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # MLflow setup
        mlflow.set_experiment(experiment_name)
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float, Dict]:
        """
        Validate the model
        
        Returns:
            Tuple of (average_loss, accuracy, metrics_dict)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='binary'),
            'recall': recall_score(all_labels, all_preds, average='binary'),
            'f1_score': f1_score(all_labels, all_preds, average='binary')
        }
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        metrics['confusion_matrix'] = cm
        
        return epoch_loss, metrics['accuracy'], metrics
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str):
        """
        Plot and save confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Cat', 'Dog'], 
                   yticklabels=['Cat', 'Dog'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_training_history(self, save_path: str):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Training history saved to {save_path}")
    
    def train(self, model_save_path: str = 'models/model.pt') -> Dict:
        """
        Full training loop with MLflow tracking
        
        Args:
            model_save_path: Path to save the best model
            
        Returns:
            Dictionary with training results
        """
        best_val_acc = 0.0
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("num_epochs", self.num_epochs)
            mlflow.log_param("batch_size", self.train_loader.batch_size)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("model_architecture", "SimpleCNN")
            
            for epoch in range(self.num_epochs):
                print(f"\nEpoch {epoch+1}/{self.num_epochs}")
                print("="*60)
                logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
                
                # Train
                train_loss, train_acc = self.train_epoch()
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)
                
                # Validate
                val_loss, val_acc, metrics = self.validate()
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Log metrics to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                mlflow.log_metric("val_precision", metrics['precision'], step=epoch)
                mlflow.log_metric("val_recall", metrics['recall'], step=epoch)
                mlflow.log_metric("val_f1_score", metrics['f1_score'], step=epoch)
                
                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                logger.info(f"Val Precision: {metrics['precision']:.4f}, Val Recall: {metrics['recall']:.4f}, Val F1: {metrics['f1_score']:.4f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), model_save_path)
                    logger.info(f"Best model saved with validation accuracy: {best_val_acc:.4f}")
            
            # Plot and log artifacts
            cm_path = 'confusion_matrix.png'
            self.plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
            mlflow.log_artifact(cm_path)
            
            history_path = 'training_history.png'
            self.plot_training_history(history_path)
            mlflow.log_artifact(history_path)
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            # Clean up temporary files
            if os.path.exists(cm_path):
                os.remove(cm_path)
            if os.path.exists(history_path):
                os.remove(history_path)
        
        return {
            'best_val_accuracy': best_val_acc,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'metrics': metrics
        }
