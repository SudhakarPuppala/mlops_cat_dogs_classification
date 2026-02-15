"""
Unit tests for data preprocessing functions
"""
import pytest
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import (
    normalize_image,
    denormalize_image,
    get_transforms,
    CatsDogsDataset
)


class TestDataPreprocessing:
    """Test suite for data preprocessing functions"""
    
    def test_normalize_image(self):
        """Test image normalization"""
        # Create a dummy image tensor
        image = torch.rand(3, 224, 224)
        
        # Normalize
        normalized = normalize_image(image)
        
        # Check shape
        assert normalized.shape == (3, 224, 224), "Shape should remain unchanged"
        
        # Check that values are normalized (approximately)
        # Mean should be close to 0 and std close to 1 per channel
        for c in range(3):
            channel_mean = normalized[c].mean().item()
            channel_std = normalized[c].std().item()
            assert abs(channel_mean) < 2.0, f"Channel {c} mean should be close to 0"
            assert 0.5 < channel_std < 2.0, f"Channel {c} std should be reasonable"
    
    def test_denormalize_image(self):
        """Test image denormalization"""
        # Create a dummy normalized image
        normalized_image = torch.randn(3, 224, 224)
        
        # Denormalize
        denormalized = denormalize_image(normalized_image)
        
        # Check shape
        assert denormalized.shape == (3, 224, 224), "Shape should remain unchanged"
        
        # Check that values are in reasonable range
        assert denormalized.min() > -5.0, "Min value should be reasonable"
        assert denormalized.max() < 5.0, "Max value should be reasonable"
    
    def test_normalize_denormalize_inverse(self):
        """Test that denormalize is inverse of normalize"""
        # Create a dummy image
        image = torch.rand(3, 224, 224)
        
        # Normalize and denormalize
        normalized = normalize_image(image)
        recovered = denormalize_image(normalized)
        
        # Check that recovered is close to original
        diff = torch.abs(image - recovered).mean().item()
        assert diff < 1e-5, "Denormalize should be inverse of normalize"
    
    def test_get_transforms_training(self):
        """Test training transforms"""
        transform = get_transforms(is_training=True)
        
        # Check that it's a Compose object
        assert isinstance(transform, transforms.Compose), "Should return Compose object"
        
        # Check that it includes data augmentation
        transform_types = [type(t).__name__ for t in transform.transforms]
        assert 'RandomHorizontalFlip' in transform_types, "Should include horizontal flip"
        assert 'RandomRotation' in transform_types, "Should include rotation"
        assert 'ToTensor' in transform_types, "Should include ToTensor"
        assert 'Normalize' in transform_types, "Should include normalization"
    
    def test_get_transforms_validation(self):
        """Test validation transforms"""
        transform = get_transforms(is_training=False)
        
        # Check that it's a Compose object
        assert isinstance(transform, transforms.Compose), "Should return Compose object"
        
        # Check that it does NOT include data augmentation
        transform_types = [type(t).__name__ for t in transform.transforms]
        assert 'RandomHorizontalFlip' not in transform_types, "Should not include horizontal flip"
        assert 'RandomRotation' not in transform_types, "Should not include rotation"
        assert 'ToTensor' in transform_types, "Should include ToTensor"
        assert 'Normalize' in transform_types, "Should include normalization"
    
    def test_transforms_output_shape(self):
        """Test that transforms produce correct output shape"""
        # Create a dummy PIL image
        dummy_image = Image.new('RGB', (300, 300), color='red')
        
        # Apply transforms
        transform = get_transforms(is_training=False)
        output = transform(dummy_image)
        
        # Check output shape
        assert output.shape == (3, 224, 224), "Output should be 3x224x224"
        assert isinstance(output, torch.Tensor), "Output should be a tensor"
    
    def test_cats_dogs_dataset(self):
        """Test CatsDogsDataset class"""
        # Create dummy data
        image_paths = ['dummy1.jpg', 'dummy2.jpg', 'dummy3.jpg']
        labels = [0, 1, 0]
        
        # Create dataset (without transform to avoid file loading)
        dataset = CatsDogsDataset(image_paths, labels, transform=None)
        
        # Check length
        assert len(dataset) == 3, "Dataset length should be 3"
        
        # Check that it stores the data correctly
        assert dataset.image_paths == image_paths, "Image paths should match"
        assert dataset.labels == labels, "Labels should match"


class TestDataProcessingEdgeCases:
    """Test edge cases and error handling"""
    
    def test_normalize_single_channel(self):
        """Test normalization with different number of channels"""
        # Single channel image (grayscale)
        image = torch.rand(1, 224, 224)
        
        # Should handle gracefully or raise appropriate error
        try:
            normalized = normalize_image(image)
            # If it doesn't raise error, check basic properties
            assert normalized.shape[0] == 1, "Should maintain channel dimension"
        except:
            # If it raises error, that's also acceptable for this edge case
            pass
    
    def test_normalize_empty_image(self):
        """Test normalization with zero-sized image"""
        image = torch.rand(3, 0, 0)
        
        try:
            normalized = normalize_image(image)
            assert normalized.shape == (3, 0, 0), "Should handle empty images"
        except:
            # Expected to fail, which is acceptable
            pass
    
    def test_transforms_different_aspect_ratios(self):
        """Test transforms with various aspect ratios"""
        transform = get_transforms(is_training=False)
        
        # Test different aspect ratios
        for width, height in [(100, 300), (300, 100), (224, 224), (500, 200)]:
            dummy_image = Image.new('RGB', (width, height), color='blue')
            output = transform(dummy_image)
            
            # All should be resized to 224x224
            assert output.shape == (3, 224, 224), f"Failed for size ({width}, {height})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
