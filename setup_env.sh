#!/bin/bash
# =============================================================================
# Setup script — Chest X-Ray Pneumonia Detection Project
# =============================================================================
set -e

echo "=================================================="
echo "  Chest X-Ray Pneumonia Detection — Environment Setup"
echo "=================================================="

# Suppress albumentations version check (avoids SSL warning on macOS)
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment (.venv)..."
python3 -m venv .venv

# Activate
echo "Activating .venv..."
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install project as editable package
echo ""
echo "Installing project in editable mode..."
pip install -e .

# Register Jupyter kernel
echo ""
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=chest-xray --display-name="Python (chest-xray)"

# Create reports directory
mkdir -p reports models_saved

echo ""
echo "=================================================="
echo "  Setup complete!"
echo ""
echo "  To activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  To launch Jupyter:"
echo "    jupyter notebook notebooks/"
echo ""
echo "  To view MLflow experiments:"
echo "    mlflow ui --backend-store-uri ./mlruns --port 5000"
echo "=================================================="
