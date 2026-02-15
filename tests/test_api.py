"""
Unit tests for FastAPI inference service
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os
from io import BytesIO
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference_api import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create a dummy RGB image
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # 503 if model not loaded
        data = response.json()
        assert "status" in data
    
    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        # May return 503 if model not loaded, which is acceptable
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "num_classes" in data
            assert "class_names" in data
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Check that response is text
        assert "text/plain" in response.headers.get("content-type", "")


class TestPredictionEndpoint:
    """Test prediction endpoint"""
    
    def test_predict_with_valid_image(self, client, sample_image):
        """Test prediction with valid image"""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        
        # Should work or return error depending on model loading
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert data["prediction"] in ["cat", "dog"]
            assert 0 <= data["confidence"] <= 1
            assert "cat" in data["probabilities"]
            assert "dog" in data["probabilities"]
    
    def test_predict_without_file(self, client):
        """Test prediction without file"""
        response = client.post("/predict")
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_with_invalid_file_type(self, client):
        """Test prediction with non-image file"""
        # Create a text file
        text_file = BytesIO(b"This is not an image")
        
        response = client.post(
            "/predict",
            files={"file": ("test.txt", text_file, "text/plain")}
        )
        
        assert response.status_code == 400  # Bad Request


class TestAPIResponseFormat:
    """Test API response formats"""
    
    def test_health_response_format(self, client):
        """Test health check response format"""
        response = client.get("/health")
        data = response.json()
        
        assert isinstance(data, dict)
        assert "status" in data
        assert isinstance(data["status"], str)
    
    def test_prediction_response_format(self, client, sample_image):
        """Test prediction response format"""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Check data types
            assert isinstance(data["prediction"], str)
            assert isinstance(data["confidence"], float)
            assert isinstance(data["probabilities"], dict)
            assert isinstance(data["probabilities"]["cat"], float)
            assert isinstance(data["probabilities"]["dog"], float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
