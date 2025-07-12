#!/bin/bash

# ReCEP Environment Installation Script
set -e

echo "Starting ReCEP environment installation..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'ReCEP' with Python 3.10..."
conda create -n ReCEP python=3.10 -y

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ReCEP

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121_full

# Install PyTorch Geometric dependencies
echo "Installing PyTorch Geometric dependencies..."
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric==2.6.1

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install in development mode
echo "Installing ReCEP in development mode..."
pip install -e .

echo "Installation completed successfully!"
echo "To activate the environment, run: conda activate ReCEP" 