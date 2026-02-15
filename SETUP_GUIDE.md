# Setup and Verification Guide

## Complete Setup Steps

### Step 1: Initial Setup

```bash
# 1. Clone or extract the project
cd mlops-cats-dogs-project

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Initialize Git (if not already done)
git init
git add .
git commit -m "Initial commit"
```

### Step 2: Data Setup (Important!)

Since this is a demonstration project, you need to either:

**Option A: Use the dummy training approach**
```bash
# This will train with synthetic data (for testing the pipeline)
python scripts/train_model.py --epochs 5 --batch-size 16
```

**Option B: Download real data**
```bash
# 1. Download Cats vs Dogs dataset from Kaggle
# Visit: https://www.kaggle.com/datasets/salader/dogs-vs-cats

# 2. Create data structure:
mkdir -p data/raw/cats data/raw/dogs

# 3. Extract images to respective folders
# data/raw/cats/*.jpg
# data/raw/dogs/*.jpg

# 4. Create preprocessing script (example provided in notebooks/)
```

### Step 3: Create a Model

```bash
# Option 1: Create dummy model (for API testing)
python scripts/create_dummy_model.py

# Option 2: Train actual model (requires data)
python scripts/train_model.py --epochs 10 --batch-size 32 --lr 0.001
```

### Step 4: Test the API Locally

```bash
# Terminal 1: Start the API
uvicorn src.inference_api:app --host 0.0.0.0 --port 8000

# Terminal 2: Test endpoints
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Test prediction (need an image)
curl -X POST http://localhost:8000/predict \
    -F "file=@/path/to/test_image.jpg"
```

### Step 5: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_preprocessing.py -v
```

### Step 6: Docker Setup

```bash
# Build Docker image
docker build -t cats-dogs-classifier:latest .

# Run container
docker run -d -p 8000:8000 --name cats-dogs-api \
    -v $(pwd)/models:/app/models:ro \
    cats-dogs-classifier:latest

# Test the containerized API
curl http://localhost:8000/health

# View logs
docker logs cats-dogs-api

# Stop and remove
docker stop cats-dogs-api
docker rm cats-dogs-api
```

### Step 7: Docker Compose Setup (Optional)

```bash
# Navigate to docker-compose directory
cd deployment/docker-compose

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f classifier

# Access services:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090

# Stop all services
docker-compose down
```

### Step 8: Kubernetes Deployment (Optional)

```bash
# Prerequisites: minikube or kind installed

# Start minikube
minikube start

# Update deployment.yaml with your Docker image
# Edit: deployment/kubernetes/deployment.yaml
# Replace: YOUR_DOCKERHUB_USERNAME/cats-dogs-classifier:latest

# Apply manifests
kubectl apply -f deployment/kubernetes/deployment.yaml

# Check deployment
kubectl get pods
kubectl get svc

# Port forward to access
kubectl port-forward svc/cats-dogs-classifier-service 8000:80

# Test
curl http://localhost:8000/health
```

### Step 9: CI/CD Setup (GitHub Actions)

```bash
# 1. Push to GitHub
git remote add origin <your-github-repo-url>
git push -u origin main

# 2. Add GitHub Secrets (Settings → Secrets and variables → Actions)
# - DOCKERHUB_USERNAME: your Docker Hub username
# - DOCKERHUB_TOKEN: your Docker Hub access token

# 3. The CI/CD pipeline will run automatically on push to main

# 4. View workflow results in the Actions tab
```

### Step 10: Run Smoke Tests

```bash
# Make sure API is running (locally or in Docker)

# Run smoke tests
bash scripts/smoke_test.sh

# Or with custom service URL
SERVICE_URL=http://localhost:8000 bash scripts/smoke_test.sh
```

## Verification Checklist

### Module 1: Model Development & Experiment Tracking
- [ ] Git repository initialized
- [ ] DVC configured (.dvc/config exists)
- [ ] Model code implemented (src/model.py)
- [ ] Training code with MLflow (src/train.py)
- [ ] Can run training: `python scripts/train_model.py`
- [ ] MLflow UI accessible: `mlflow ui` → http://localhost:5000
- [ ] Model saved in models/ directory

### Module 2: Model Packaging & Containerization
- [ ] FastAPI service implemented (src/inference_api.py)
- [ ] Health endpoint works: `/health`
- [ ] Predict endpoint works: `/predict`
- [ ] Model info endpoint works: `/model/info`
- [ ] Metrics endpoint works: `/metrics`
- [ ] requirements.txt has pinned versions
- [ ] Dockerfile builds successfully
- [ ] Docker image runs and serves API

### Module 3: CI Pipeline
- [ ] Unit tests written for preprocessing (tests/test_preprocessing.py)
- [ ] Unit tests written for model (tests/test_model.py)
- [ ] Unit tests written for API (tests/test_api.py)
- [ ] All tests pass: `pytest tests/ -v`
- [ ] GitHub Actions workflow file exists (.github/workflows/ci-cd.yml)
- [ ] CI pipeline configured for test → build → publish

### Module 4: CD Pipeline & Deployment
- [ ] Kubernetes manifests created (deployment/kubernetes/)
- [ ] Docker Compose file created (deployment/docker-compose/)
- [ ] Deployment includes Service and Deployment
- [ ] HPA configured
- [ ] Smoke test script works (scripts/smoke_test.sh)
- [ ] Can deploy to local Kubernetes or Docker Compose
- [ ] Health checks configured (liveness/readiness)

### Module 5: Monitoring & Logging
- [ ] Application logs request/response
- [ ] Prometheus metrics exposed (/metrics)
- [ ] Metrics tracked: request count, latency, predictions by class
- [ ] Prometheus configuration file exists
- [ ] Can view metrics in Prometheus UI (if using Docker Compose)

## Common Issues and Solutions

### Issue 1: Module Not Found Errors
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 2: Tests Fail Due to Missing Dependencies
```bash
# Install test dependencies explicitly
pip install pytest pytest-cov httpx
```

### Issue 3: Docker Build Fails
```bash
# Clear Docker cache
docker builder prune

# Build with no cache
docker build --no-cache -t cats-dogs-classifier:latest .
```

### Issue 4: Model File Not Found
```bash
# Create dummy model for testing
python scripts/create_dummy_model.py

# Verify file exists
ls -lh models/model.pt
```

### Issue 5: Port Already in Use
```bash
# Find what's using the port
lsof -i :8000

# Kill the process or use different port
uvicorn src.inference_api:app --port 8001
```

### Issue 6: Kubernetes Pod Not Starting
```bash
# Check pod logs
kubectl logs <pod-name>

# Describe pod for events
kubectl describe pod <pod-name>

# Common fixes:
# - Update image pull policy
# - Check resource limits
# - Verify image exists in registry
```

## Quick Reference Commands

```bash
# Development
make install          # Install dependencies
make test            # Run tests
make train           # Train model
make clean           # Clean up generated files

# Docker
make docker-build    # Build image
make docker-run      # Run container
make docker-stop     # Stop container

# Testing
pytest tests/ -v                    # All tests
pytest tests/ -k "test_normalize"   # Specific test
pytest tests/ --cov=src            # With coverage

# API Testing
curl http://localhost:8000/health
curl http://localhost:8000/model/info
curl -X POST http://localhost:8000/predict -F "file=@image.jpg"

# MLflow
mlflow ui --port 5000              # Start UI
mlflow models serve -m runs:/<run-id>/model  # Serve model

# Docker Compose
docker-compose up -d               # Start all
docker-compose ps                  # Status
docker-compose logs -f             # Logs
docker-compose down                # Stop all

# Kubernetes
kubectl apply -f deployment/kubernetes/
kubectl get pods
kubectl get svc
kubectl logs <pod-name>
kubectl port-forward svc/cats-dogs-classifier-service 8000:80
```

## Next Steps for Production

1. **Data Pipeline**
   - Implement automated data ingestion
   - Add data validation and quality checks
   - Set up data versioning with DVC remote storage

2. **Model Improvements**
   - Use transfer learning (ResNet, EfficientNet)
   - Implement model ensembling
   - Add model versioning and A/B testing

3. **API Enhancements**
   - Add authentication and authorization
   - Implement rate limiting
   - Add request validation and error handling
   - Enable HTTPS

4. **Monitoring & Observability**
   - Set up Grafana dashboards
   - Implement distributed tracing
   - Add alerting rules
   - Track model performance drift

5. **Security**
   - Scan container images for vulnerabilities
   - Implement secrets management
   - Add network policies
   - Enable RBAC in Kubernetes

6. **Performance**
   - Implement batch inference
   - Add caching layer
   - Use GPU for inference
   - Optimize model (quantization, pruning)
