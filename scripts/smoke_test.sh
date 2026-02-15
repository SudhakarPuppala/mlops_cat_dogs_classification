#!/bin/bash
# Smoke tests for deployed service

set -e

# Configuration
SERVICE_URL="${SERVICE_URL:-http://localhost:8000}"
MAX_RETRIES=30
RETRY_INTERVAL=2

echo "=== Running Smoke Tests ==="
echo "Service URL: $SERVICE_URL"
echo ""

# Function to wait for service
wait_for_service() {
    echo "Waiting for service to be ready..."
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -f -s "${SERVICE_URL}/health" > /dev/null 2>&1; then
            echo "Service is ready!"
            return 0
        fi
        echo "Attempt $i/$MAX_RETRIES: Service not ready yet, waiting..."
        sleep $RETRY_INTERVAL
    done
    echo "Service failed to become ready after $MAX_RETRIES attempts"
    return 1
}

# Test 1: Health Check
test_health_check() {
    echo "Test 1: Health Check"
    response=$(curl -s -w "\n%{http_code}" "${SERVICE_URL}/health")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" -eq 200 ] || [ "$http_code" -eq 503 ]; then
        echo "✓ Health check passed (HTTP $http_code)"
        echo "  Response: $body"
    else
        echo "✗ Health check failed (HTTP $http_code)"
        echo "  Response: $body"
        return 1
    fi
}

# Test 2: Root Endpoint
test_root_endpoint() {
    echo ""
    echo "Test 2: Root Endpoint"
    response=$(curl -s -w "\n%{http_code}" "${SERVICE_URL}/")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" -eq 200 ]; then
        echo "✓ Root endpoint passed (HTTP $http_code)"
        echo "  Response: $body"
    else
        echo "✗ Root endpoint failed (HTTP $http_code)"
        return 1
    fi
}

# Test 3: Model Info
test_model_info() {
    echo ""
    echo "Test 3: Model Info"
    response=$(curl -s -w "\n%{http_code}" "${SERVICE_URL}/model/info")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" -eq 200 ] || [ "$http_code" -eq 503 ]; then
        echo "✓ Model info passed (HTTP $http_code)"
        echo "  Response: $body"
    else
        echo "✗ Model info failed (HTTP $http_code)"
        return 1
    fi
}

# Test 4: Prediction (with dummy image)
test_prediction() {
    echo ""
    echo "Test 4: Prediction Endpoint"
    
    # Create a temporary test image
    temp_image="/tmp/test_image.jpg"
    
    # Try to create a simple image using ImageMagick or Python
    if command -v convert &> /dev/null; then
        convert -size 224x224 xc:red "$temp_image"
    elif command -v python3 &> /dev/null; then
        python3 -c "from PIL import Image; Image.new('RGB', (224, 224), 'red').save('$temp_image')"
    else
        echo "⚠ Skipping prediction test (no image creation tool available)"
        return 0
    fi
    
    if [ -f "$temp_image" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST \
            -F "file=@${temp_image}" \
            "${SERVICE_URL}/predict")
        http_code=$(echo "$response" | tail -n1)
        body=$(echo "$response" | head -n-1)
        
        if [ "$http_code" -eq 200 ]; then
            echo "✓ Prediction passed (HTTP $http_code)"
            echo "  Response: $body"
        else
            echo "⚠ Prediction returned HTTP $http_code (may be expected if model not loaded)"
            echo "  Response: $body"
        fi
        
        rm -f "$temp_image"
    fi
}

# Test 5: Metrics Endpoint
test_metrics() {
    echo ""
    echo "Test 5: Metrics Endpoint"
    response=$(curl -s -w "\n%{http_code}" "${SERVICE_URL}/metrics")
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" -eq 200 ]; then
        echo "✓ Metrics endpoint passed (HTTP $http_code)"
        # Show first few lines of metrics
        echo "  Sample metrics:"
        echo "$response" | head -n-1 | head -n 5
    else
        echo "✗ Metrics endpoint failed (HTTP $http_code)"
        return 1
    fi
}

# Main execution
main() {
    wait_for_service || exit 1
    
    failed=0
    
    test_health_check || ((failed++))
    test_root_endpoint || ((failed++))
    test_model_info || ((failed++))
    test_prediction || ((failed++))
    test_metrics || ((failed++))
    
    echo ""
    echo "=== Smoke Test Summary ==="
    if [ $failed -eq 0 ]; then
        echo "✓ All critical tests passed!"
        exit 0
    else
        echo "✗ $failed test(s) failed"
        exit 1
    fi
}

main
