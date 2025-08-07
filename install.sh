#!/bin/bash

# ReCEP Environment Installation Script
set -e

echo "Starting RoBep environment installation..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'RoBep' with Python 3.10..."
conda create -n RoBep python=3.10 -y

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate RoBep

pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121_full

pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

pip install torch-geometric==2.6.1

pip install -r requirements.txt

pip install -e .

echo "Installation completed successfully!"
