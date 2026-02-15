# MLOps Pipeline for Cats vs Dogs Classification

## Project Overview

This project implements a complete end-to-end MLOps pipeline for a binary image classification task (Cats vs Dogs) designed for a pet adoption platform. The pipeline includes model development, experiment tracking, containerization, CI/CD, deployment, and monitoring.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Details](#module-details)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Contributing](#contributing)

## Features

### M1: Model Development & Experiment Tracking ✅
- **Data Versioning**: DVC integration for dataset tracking
- **Code Versioning**: Git-based source control
- **Model Architecture**: Custom CNN with batch normalization and dropout
- **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts
- **Metrics Logged**: Loss curves, accuracy, precision, recall, F1-score, confusion matrix

### M2: Model Packaging & Containerization ✅
- **REST API**: FastAPI-based inference service
- **Endpoints**: 
  - `/` - Root endpoint with API information
  - `/health` - Health check
  - `/predict` - Image classification
  - `/model/info` - Model metadata
  - `/metrics` - Prometheus metrics
- **Environment Specification**: Pinned dependencies in requirements.txt
- **Containerization**: Production-ready Dockerfile with multi-stage optimization

### M3: CI Pipeline ✅
- **Automated Testing**: Comprehensive unit tests with pytest
- **GitHub Actions**: CI workflow for testing, building, and publishing
- **Docker Image Publishing**: Automatic push to Docker Hub/Container Registry
- **Code Quality**: Test coverage reporting

### M4: CD Pipeline & Deployment ✅
- **Kubernetes Deployment**: Complete manifests with HPA, Service, Deployment
- **Docker Compose**: Alternative deployment for local/VM environments
- **GitOps**: Automated deployment on main branch changes
- **Smoke Tests**: Post-deployment health and functionality checks

### M5: Monitoring & Logging ✅
- **Request Logging**: Structured logging for all API requests
- **Prometheus Metrics**: Request count, latency, predictions by class
- **Health Checks**: Liveness and readiness probes
- **Performance Tracking**: Latency monitoring and resource utilization

## Project Structure

```
mlops-cats-dogs-project/
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data loading and augmentation
│   ├── model.py                  # CNN model architecture
│   ├── train.py                  # Training loop with MLflow
│   └── inference_api.py          # FastAPI service
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py     # Data preprocessing tests
│   ├── test_model.py            # Model architecture tests
│   └── test_api.py              # API endpoint tests
├── models/                       # Saved models
│   └── model.pt                 # Trained model weights
├── data/                         # Dataset directory
│   ├── raw/                     # Original data
│   └── processed/               # Preprocessed data
├── deployment/                   # Deployment configurations
│   ├── kubernetes/              # K8s manifests
│   │   └── deployment.yaml
│   └── docker-compose/          # Docker Compose setup
│       └── docker-compose.yml
├── scripts/                      # Utility scripts
│   ├── train_model.py           # Training script
│   ├── smoke_test.sh            # Post-deployment tests
│   ├── docker_run.sh            # Docker run helper
│   └── create_dummy_model.py    # Create demo model
├── monitoring/                   # Monitoring configs
│   └── prometheus.yml           # Prometheus configuration
├── .github/workflows/           # CI/CD workflows
│   └── ci-cd.yml               # GitHub Actions pipeline
├── notebooks/                    # Jupyter notebooks
├── Dockerfile                    # Container image definition
├── docker-compose.yml           # Alternative: compose in root
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Test configuration
├── Makefile                     # Common commands
├── .gitignore                   # Git ignore patterns
├── .dockerignore               # Docker ignore patterns
├── .dvc/config                 # DVC configuration
└── README.md                    # This file
```

## Requirements

### System Requirements
- Python 3.10+
- Docker 20.10+
- Git 2.30+
- 4GB RAM minimum (8GB recommended)
- 10GB disk space

### Python Dependencies
See `requirements.txt` for complete list. Key dependencies:
- torch==2.1.0
- torchvision==0.16.0
- fastapi==0.104.1
- mlflow==2.8.1
- dvc==3.30.1
- pytest==7.4.3

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd mlops-cats-dogs-project
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Initialize DVC
```bash
dvc init
```

## Quick Start

### Option 1: Using Make (Recommended)

```bash
# Install dependencies
make install

# Run tests
make test

# Train model (with dummy data)
make train

# Build Docker image
make docker-build

# Run Docker container
make docker-run

# Run smoke tests
make smoke-test

# Stop container
make docker-stop
```

### Option 2: Manual Commands

#### Run Tests
```bash
pytest tests/ -v --cov=src
```

#### Train Model
```bash
python scripts/train_model.py \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001
```

#### Start API Locally
```bash
uvicorn src.inference_api:app --host 0.0.0.0 --port 8000
```

#### Build and Run Docker
```bash
# Build
docker build -t cats-dogs-classifier:latest .

# Run
docker run -d -p 8000:8000 --name cats-dogs-api \
    -v $(pwd)/models:/app/models:ro \
    cats-dogs-classifier:latest
```

## Module Details

### M1: Model Development & Experiment Tracking

#### Data Preprocessing
The `data_preprocessing.py` module handles:
- Image resizing to 224x224
- Data augmentation (flip, rotation, color jitter)
- Normalization using ImageNet statistics
- Train/validation/test splitting

#### Model Architecture
Custom CNN (`SimpleCNN`) with:
- 4 convolutional blocks with batch normalization
- Max pooling layers
- Dropout for regularization (50%)
- 3 fully connected layers
- ~17M parameters

#### Training
```python
from src.train import Trainer
from src.model import get_model

model = get_model(num_classes=2)
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10
)
results = trainer.train()
```

#### MLflow Tracking
```bash
# Start MLflow UI
mlflow ui --port 5000

# View at http://localhost:5000
```

Tracks:
- Hyperparameters (learning rate, batch size, optimizer)
- Metrics per epoch (loss, accuracy, precision, recall, F1)
- Artifacts (confusion matrix, training curves, model)

### M2: Model Packaging & Containerization

#### API Endpoints

**Health Check**
```bash
curl http://localhost:8000/health
```

**Prediction**
```bash
curl -X POST http://localhost:8000/predict \
    -F "file=@cat_image.jpg"
```

Response:
```json
{
  "prediction": "cat",
  "confidence": 0.92,
  "probabilities": {
    "cat": 0.92,
    "dog": 0.08
  },
  "latency_seconds": 0.045
}
```

**Model Info**
```bash
curl http://localhost:8000/model/info
```

**Metrics**
```bash
curl http://localhost:8000/metrics
```

### M3: CI Pipeline

GitHub Actions workflow (`.github/workflows/ci-cd.yml`):

1. **Test Job**
   - Checkout code
   - Setup Python
   - Install dependencies
   - Run pytest with coverage
   - Upload coverage reports

2. **Build Job** (on main branch)
   - Build Docker image
   - Tag with branch and SHA
   - Push to Docker Hub

3. **Deploy Job** (optional)
   - Deploy to Kubernetes
   - Run smoke tests

#### Setup CI/CD

1. Add secrets to GitHub repository:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
   - `KUBECONFIG` (if using K8s)

2. Push to trigger workflow:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### M4: CD Pipeline & Deployment

#### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f deployment/kubernetes/deployment.yaml

# Check status
kubectl get pods
kubectl get svc

# Access service
kubectl port-forward svc/cats-dogs-classifier-service 8000:80
```

Features:
- 2 replicas for high availability
- Horizontal Pod Autoscaler (2-5 replicas)
- Resource limits (1 CPU, 1Gi memory)
- Liveness and readiness probes
- LoadBalancer service

#### Docker Compose Deployment

```bash
# Start services
cd deployment/docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Includes:
- Classifier API (port 8000)
- MLflow server (port 5000)
- Prometheus (port 9090)

#### Smoke Tests

```bash
bash scripts/smoke_test.sh
```

Tests:
- Health endpoint
- Root endpoint
- Model info
- Prediction (with test image)
- Metrics endpoint

### M5: Monitoring & Logging

#### Application Logging

Structured JSON logging for:
- Request/response details
- Prediction results
- Error traces
- Performance metrics

Example:
```json
{
  "timestamp": "2024-02-10T10:30:45",
  "level": "INFO",
  "message": "Prediction: cat (confidence: 0.92)",
  "latency_seconds": 0.045
}
```

#### Prometheus Metrics

Available metrics:
- `prediction_requests_total` - Total prediction requests
- `prediction_latency_seconds` - Request latency histogram
- `predictions_by_class` - Predictions count by class

Access Prometheus:
```bash
# If using Docker Compose
http://localhost:9090

# Query examples
prediction_requests_total
rate(prediction_requests_total[5m])
histogram_quantile(0.95, prediction_latency_seconds)
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Files
```bash
pytest tests/test_preprocessing.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
# View coverage report in htmlcov/index.html
```

### Test Categories
- **Data Preprocessing**: Normalization, transforms, dataset creation
- **Model Architecture**: Forward pass, parameter counts, save/load
- **API Endpoints**: Health checks, predictions, error handling

## Data Preparation

### Download Dataset
```bash
# Download from Kaggle
kaggle datasets download -d salader/dogs-vs-cats

# Or use any cats/dogs dataset
# Structure should be:
# data/
#   raw/
#     cats/
#       cat1.jpg
#       cat2.jpg
#     dogs/
#       dog1.jpg
#       dog2.jpg
```

### Preprocess Data
```python
# See notebooks/ or create a preprocessing script
# - Resize to 224x224
# - Split 80/10/10 (train/val/test)
# - Save processed images
```

### Version Data with DVC
```bash
dvc add data/processed
git add data/processed.dvc
git commit -m "Add processed dataset"
```

## Environment Variables

Copy `.env.template` to `.env` and configure:

```bash
cp .env.template .env
```

Edit `.env`:
```
MODEL_PATH=models/model.pt
DEVICE=cpu
API_HOST=0.0.0.0
API_PORT=8000
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Troubleshooting

### Issue: Model not loading in API
```bash
# Create a dummy model for testing
python scripts/create_dummy_model.py

# Or train a real model
python scripts/train_model.py
```

### Issue: Docker build fails
```bash
# Clear Docker cache
docker builder prune

# Rebuild without cache
docker build --no-cache -t cats-dogs-classifier:latest .
```

### Issue: Tests failing
```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests with verbose output
pytest tests/ -vv
```

### Issue: Port already in use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
uvicorn src.inference_api:app --port 8001
```

## Performance Optimization

### Model Optimization
- Use mixed precision training (FP16)
- Implement model quantization
- Apply pruning techniques
- Use TorchScript for deployment

### API Optimization
- Enable response caching
- Batch inference for multiple images
- Use async I/O
- Implement request queuing

### Infrastructure Optimization
- Use GPU for inference
- Implement horizontal scaling
- Add CDN for static assets
- Use connection pooling

## Security Best Practices

1. **Container Security**
   - Run as non-root user
   - Scan images for vulnerabilities
   - Use minimal base images
   - Keep dependencies updated

2. **API Security**
   - Implement rate limiting
   - Add authentication/authorization
   - Validate input files
   - Sanitize error messages

3. **Secrets Management**
   - Use environment variables
   - Never commit secrets to Git
   - Rotate credentials regularly
   - Use secrets managers (K8s secrets, Vault)

## Contributing

### Development Workflow

1. Create feature branch
```bash
git checkout -b feature/new-feature
```

2. Make changes and test
```bash
# Make changes
# Run tests
make test
```

3. Commit with conventional commits
```bash
git commit -m "feat: add new feature"
# Types: feat, fix, docs, test, refactor, chore
```

4. Push and create PR
```bash
git push origin feature/new-feature
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings
- Add unit tests for new code

## License

This project is created for educational purposes as part of an MLOps assignment.

## Acknowledgments

- PyTorch and torchvision teams
- FastAPI framework
- MLflow for experiment tracking
- Prometheus for monitoring
- Kaggle for datasets

## Contact

For questions or issues, please open an issue in the repository.

---

**Assignment Completion Checklist:**

- [x] M1: Model Development & Experiment Tracking
  - [x] Git versioning
  - [x] DVC for data versioning
  - [x] Baseline CNN model
  - [x] MLflow experiment tracking
  
- [x] M2: Model Packaging & Containerization
  - [x] FastAPI inference service
  - [x] Health and predict endpoints
  - [x] requirements.txt with pinned versions
  - [x] Dockerfile
  
- [x] M3: CI Pipeline
  - [x] Unit tests (preprocessing, model, API)
  - [x] GitHub Actions CI workflow
  - [x] Automated testing
  - [x] Docker image publishing
  
- [x] M4: CD Pipeline & Deployment
  - [x] Kubernetes manifests
  - [x] Docker Compose configuration
  - [x] Smoke tests
  - [x] Health checks
  
- [x] M5: Monitoring & Logging
  - [x] Request/response logging
  - [x] Prometheus metrics
  - [x] Performance tracking
  - [x] Basic monitoring setup
