#!/bin/bash
# Build and run Docker container locally

set -e

IMAGE_NAME="cats-dogs-classifier"
CONTAINER_NAME="cats-dogs-api"
PORT=8000

echo "=== Building Docker Image ==="
docker build -t $IMAGE_NAME:latest .

echo ""
echo "=== Stopping existing container (if any) ==="
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo ""
echo "=== Running Docker Container ==="
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8000 \
    -v $(pwd)/models:/app/models:ro \
    $IMAGE_NAME:latest

echo ""
echo "=== Container Status ==="
docker ps | grep $CONTAINER_NAME

echo ""
echo "=== Waiting for service to be ready ==="
sleep 5

echo ""
echo "=== Testing API ==="
curl http://localhost:$PORT/health

echo ""
echo ""
echo "Container is running!"
echo "API available at: http://localhost:$PORT"
echo "Health check: curl http://localhost:$PORT/health"
echo "View logs: docker logs -f $CONTAINER_NAME"
echo "Stop container: docker stop $CONTAINER_NAME"
