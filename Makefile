.PHONY: help install test train docker-build docker-run docker-stop clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run unit tests"
	@echo "  make train         - Train the model"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make docker-stop   - Stop Docker container"
	@echo "  make clean         - Clean up generated files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

train:
	python scripts/train_model.py

docker-build:
	docker build -t cats-dogs-classifier:latest .

docker-run:
	docker run -d --name cats-dogs-api -p 8000:8000 \
		-v $$(pwd)/models:/app/models:ro \
		cats-dogs-classifier:latest

docker-stop:
	docker stop cats-dogs-api || true
	docker rm cats-dogs-api || true

docker-compose-up:
	cd deployment/docker-compose && docker-compose up -d

docker-compose-down:
	cd deployment/docker-compose && docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml
	rm -f *.log

lint:
	flake8 src/ tests/ --max-line-length=120

format:
	black src/ tests/

smoke-test:
	bash scripts/smoke_test.sh
