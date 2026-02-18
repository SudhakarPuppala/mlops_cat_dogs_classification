"""
FastAPI inference service for Cats vs Dogs classification
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import io
import time
from typing import Dict
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import os

# Import model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import SimpleCNN

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests')
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
PREDICTION_COUNT = Counter('predictions_by_class', 'Predictions by class', ['class_name'])

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="API for classifying images as cats or dogs",
    version="1.0.0"
)

# Global variables
MODEL = None
DEVICE = None
TRANSFORM = None
CLASS_NAMES = ['cat', 'dog']


def load_model(model_path: str = 'models/model.pt'):
    """Load the trained model"""
    global MODEL, DEVICE, TRANSFORM
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")
    
    # Load model
    MODEL = SimpleCNN(num_classes=2)
    if os.path.exists(model_path):
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model file not found at {model_path}. Using untrained model.")
    
    MODEL.to(DEVICE)
    MODEL.eval()
    
    # Define transform
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    logger.info("Model loaded successfully")

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="API for classifying images as cats or dogs",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Cats vs Dogs Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status of the service
    """
    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "Model not loaded"
            }
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict endpoint for image classification
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with class probabilities
    """
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        # Transform image
        image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        # Make prediction
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        # Get class name and confidence
        class_name = CLASS_NAMES[predicted_class]
        confidence = probabilities[predicted_class].item()
        # Update metrics
        PREDICTION_COUNT.labels(class_name=class_name).inc()
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        # Prepare response
        response = {
            "prediction": class_name,
            "confidence": float(confidence),
            "probabilities": {
                "cat": float(probabilities[0]),
                "dog": float(probabilities[1])
            },
            "latency_seconds": latency
        }
        logger.info(f"Prediction: {class_name} (confidence: {confidence:.4f})")
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    
    Returns:
        Prometheus metrics in text format
    """
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/model/info")
async def model_info():
    """
    Get model information
    
    Returns:
        Model metadata and statistics
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in MODEL.parameters())
    trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    
    return {
        "model_name": "SimpleCNN",
        "num_classes": 2,
        "class_names": CLASS_NAMES,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(DEVICE),
        "input_size": [224, 224]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
