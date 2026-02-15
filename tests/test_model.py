"""
Unit tests for model and inference functions
"""
import pytest
import torch
import torch.nn as nn
import sys
import os
from io import BytesIO
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import SimpleCNN, get_model, save_model, load_model


class TestModelArchitecture:
    """Test suite for model architecture"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = SimpleCNN(num_classes=2)
        assert isinstance(model, nn.Module), "Model should be a PyTorch Module"
    
    def test_model_forward_pass(self):
        """Test forward pass with dummy input"""
        model = SimpleCNN(num_classes=2)
        model.eval()
        
        # Create dummy input (batch_size=4, channels=3, height=224, width=224)
        dummy_input = torch.randn(4, 3, 224, 224)
        
        # Forward pass
        output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (4, 2), "Output shape should be (batch_size, num_classes)"
    
    def test_model_output_range(self):
        """Test that model outputs are logits (not probabilities)"""
        model = SimpleCNN(num_classes=2)
        model.eval()
        
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        
        # Logits can be any real number
        assert output.dtype == torch.float32, "Output should be float32"
    
    def test_model_parameters(self):
        """Test that model has trainable parameters"""
        model = SimpleCNN(num_classes=2)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0, "Model should have parameters"
        assert trainable_params > 0, "Model should have trainable parameters"
        assert trainable_params == total_params, "All parameters should be trainable initially"
    
    def test_model_different_num_classes(self):
        """Test model with different number of classes"""
        for num_classes in [2, 5, 10]:
            model = SimpleCNN(num_classes=num_classes)
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            
            assert output.shape == (1, num_classes), f"Output should have {num_classes} classes"
    
    def test_get_model_function(self):
        """Test get_model utility function"""
        model = get_model(num_classes=2)
        
        assert isinstance(model, SimpleCNN), "Should return SimpleCNN instance"
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 2), "Model should work correctly"
    
    def test_model_get_info(self):
        """Test get_model_info method"""
        model = SimpleCNN(num_classes=2)
        info = model.get_model_info()
        
        assert 'model_name' in info, "Info should contain model_name"
        assert 'total_parameters' in info, "Info should contain total_parameters"
        assert 'trainable_parameters' in info, "Info should contain trainable_parameters"
        assert info['total_parameters'] > 0, "Should have positive parameter count"


class TestModelSaveLoad:
    """Test model saving and loading"""
    
    def test_save_model(self, tmp_path):
        """Test model saving"""
        model = SimpleCNN(num_classes=2)
        save_path = tmp_path / "test_model.pt"
        
        # Save model
        save_model(model, str(save_path))
        
        # Check file exists
        assert save_path.exists(), "Model file should be created"
        assert save_path.stat().st_size > 0, "Model file should not be empty"
    
    def test_load_model(self, tmp_path):
        """Test model loading"""
        # Create and save a model
        model1 = SimpleCNN(num_classes=2)
        save_path = tmp_path / "test_model.pt"
        torch.save(model1.state_dict(), save_path)
        
        # Load the model
        model2 = load_model(str(save_path))
        
        assert isinstance(model2, SimpleCNN), "Loaded object should be SimpleCNN"
        
        # Test that loaded model works
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model2(dummy_input)
        assert output.shape == (1, 2), "Loaded model should work correctly"
    
    def test_save_load_preserves_weights(self, tmp_path):
        """Test that save and load preserves weights"""
        # Create and save a model
        model1 = SimpleCNN(num_classes=2)
        save_path = tmp_path / "test_model.pt"
        save_model(model1, str(save_path))
        
        # Load the model
        model2 = load_model(str(save_path))
        
        # Compare weights
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), 
            model2.named_parameters()
        ):
            assert name1 == name2, "Parameter names should match"
            assert torch.allclose(param1, param2), f"Weights for {name1} should match"


class TestModelInference:
    """Test model inference capabilities"""
    
    def test_inference_mode(self):
        """Test model in eval mode"""
        model = SimpleCNN(num_classes=2)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Should work in eval mode
        with torch.no_grad():
            output = model(dummy_input)
            assert output.shape == (1, 2), "Inference should work in eval mode"
    
    def test_batch_inference(self):
        """Test inference with different batch sizes"""
        model = SimpleCNN(num_classes=2)
        model.eval()
        
        for batch_size in [1, 2, 4, 8, 16]:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                output = model(dummy_input)
                assert output.shape == (batch_size, 2), f"Should handle batch size {batch_size}"
    
    def test_prediction_consistency(self):
        """Test that same input gives same output in eval mode"""
        model = SimpleCNN(num_classes=2)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output1 = model(dummy_input)
            output2 = model(dummy_input)
        
        assert torch.allclose(output1, output2), "Same input should give same output in eval mode"
    
    def test_softmax_probabilities(self):
        """Test that softmax gives valid probabilities"""
        model = SimpleCNN(num_classes=2)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            logits = model(dummy_input)
            probs = torch.softmax(logits, dim=1)
        
        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(1)), "Probabilities should sum to 1"
        
        # Check probabilities are between 0 and 1
        assert (probs >= 0).all(), "Probabilities should be >= 0"
        assert (probs <= 1).all(), "Probabilities should be <= 1"


class TestModelGradients:
    """Test model gradient computation"""
    
    def test_backward_pass(self):
        """Test that gradients can be computed"""
        model = SimpleCNN(num_classes=2)
        model.train()
        
        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_target = torch.tensor([0, 1])
        
        # Forward pass
        output = model(dummy_input)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient for {name} should exist"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} should not contain NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
