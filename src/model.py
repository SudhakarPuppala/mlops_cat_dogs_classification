"""
CNN Model architecture for Cats vs Dogs classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Simple CNN architecture for binary image classification"""
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize the CNN model
        
        Args:
            num_classes: Number of output classes (default: 2 for cats vs dogs)
        """
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'SimpleCNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': str(self)
        }


def get_model(num_classes: int = 2, pretrained: bool = False) -> SimpleCNN:
    """
    Get model instance
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights (not implemented for custom model)
        
    Returns:
        Model instance
    """
    model = SimpleCNN(num_classes=num_classes)
    logger.info(f"Created SimpleCNN model with {num_classes} classes")
    
    # Log model information
    info = model.get_model_info()
    logger.info(f"Total parameters: {info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {info['trainable_parameters']:,}")
    
    return model


def load_model(model_path: str, device: str = 'cpu') -> SimpleCNN:
    """
    Load a trained model from file
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


def save_model(model: nn.Module, save_path: str):
    """
    Save model to file
    
    Args:
        model: Model to save
        save_path: Path to save the model
    """
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
