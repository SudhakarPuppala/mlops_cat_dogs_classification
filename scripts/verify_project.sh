#!/bin/bash
# Project structure verification script

set -e

echo "========================================"
echo "MLOps Project Verification Script"
echo "========================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success=0
warnings=0
errors=0

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((success++))
    else
        echo -e "${RED}✗${NC} $1 - MISSING"
        ((errors++))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        ((success++))
    else
        echo -e "${YELLOW}⚠${NC} $1/ - MISSING (may be optional)"
        ((warnings++))
    fi
}

echo "Checking Directory Structure..."
echo "--------------------------------"
check_dir "src"
check_dir "tests"
check_dir "models"
check_dir "deployment"
check_dir "scripts"
check_dir ".github/workflows"
check_dir "monitoring"
echo ""

echo "Checking M1: Model Development Files..."
echo "----------------------------------------"
check_file "src/model.py"
check_file "src/train.py"
check_file "src/data_preprocessing.py"
check_file ".dvc/config"
check_file "requirements.txt"
echo ""

echo "Checking M2: Containerization Files..."
echo "---------------------------------------"
check_file "src/inference_api.py"
check_file "Dockerfile"
check_file ".dockerignore"
echo ""

echo "Checking M3: CI Pipeline Files..."
echo "----------------------------------"
check_file "tests/test_preprocessing.py"
check_file "tests/test_model.py"
check_file "tests/test_api.py"
check_file ".github/workflows/ci-cd.yml"
check_file "pytest.ini"
echo ""

echo "Checking M4: CD & Deployment Files..."
echo "--------------------------------------"
check_file "deployment/kubernetes/deployment.yaml"
check_file "deployment/docker-compose/docker-compose.yml"
check_file "scripts/smoke_test.sh"
echo ""

echo "Checking M5: Monitoring Files..."
echo "---------------------------------"
check_file "monitoring/prometheus.yml"
echo ""

echo "Checking Documentation..."
echo "-------------------------"
check_file "README.md"
check_file "SETUP_GUIDE.md"
check_file ".gitignore"
echo ""

echo "Checking Utility Files..."
echo "-------------------------"
check_file "Makefile"
check_file "scripts/train_model.py"
check_file "scripts/docker_run.sh"
check_file "scripts/create_dummy_model.py"
echo ""

echo "========================================"
echo "Verification Summary"
echo "========================================"
echo -e "${GREEN}Success: $success${NC}"
echo -e "${YELLOW}Warnings: $warnings${NC}"
echo -e "${RED}Errors: $errors${NC}"
echo ""

if [ $errors -eq 0 ]; then
    echo -e "${GREEN}✓ Project structure is complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review SETUP_GUIDE.md for setup instructions"
    echo "2. Install dependencies: pip install -r requirements.txt"
    echo "3. Create a model: python scripts/create_dummy_model.py"
    echo "4. Run tests: pytest tests/ -v"
    echo "5. Build Docker: docker build -t cats-dogs-classifier ."
    exit 0
else
    echo -e "${RED}✗ Project has missing files${NC}"
    echo "Please check the errors above"
    exit 1
fi
