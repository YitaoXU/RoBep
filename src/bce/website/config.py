"""
Configuration settings for BCE Prediction Web Server
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parents[3]  # Project root directory
WEBSITE_DIR = Path(__file__).parent
STATIC_DIR = WEBSITE_DIR / "static"
TEMPLATES_DIR = WEBSITE_DIR / "templates"

# Model and data paths
DEFAULT_MODEL_PATH = os.getenv("BCE_MODEL_PATH", str(BASE_DIR / "models" / "ReCEP" / "20250626_110438" / "best_mcc_model.bin"))
DATA_DIR = os.getenv("BCE_DATA_DIR", str(BASE_DIR / "data"))

# Server settings
SERVER_HOST = os.getenv("BCE_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("BCE_PORT", "8000"))
DEBUG_MODE = os.getenv("BCE_DEBUG", "false").lower() == "true"

# Prediction settings - Updated for ESM-C as default, CPU mode by default
DEFAULT_DEVICE_ID = int(os.getenv("BCE_DEVICE_ID", "-1"))
DEFAULT_RADIUS = float(os.getenv("BCE_RADIUS", "19.0"))
DEFAULT_K = int(os.getenv("BCE_K", "7"))
DEFAULT_ENCODER = os.getenv("BCE_ENCODER", "esmc")  # Default to ESM-C
DEFAULT_CHAIN_ID = os.getenv("BCE_CHAIN_ID", "A")

# File upload settings
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".pdb", ".ent"}
TEMP_DIR = "/tmp"

# Task management
TASK_TIMEOUT = 30 * 60  # 30 minutes in seconds
MAX_CONCURRENT_TASKS = int(os.getenv("BCE_MAX_TASKS", "10"))
TASK_CLEANUP_INTERVAL = 60 * 60  # 1 hour in seconds

# Security settings
CORS_ORIGINS = ["*"]  # In production, specify allowed origins
MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100MB

# Logging configuration
LOG_LEVEL = os.getenv("BCE_LOG_LEVEL", "info").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ESM-C configuration
ESM_TOKEN = os.getenv("ESM_TOKEN", "1mzAo8l1uxaU8UfVcGgV7B")
ESM_MODEL_NAME = "esmc-6b-2024-12"
ESM_URL = "https://forge.evolutionaryscale.ai"

# Validation settings
MIN_SEQUENCE_LENGTH = 10
MAX_SEQUENCE_LENGTH = 2046  # ESM model limit

# Cache settings
ENABLE_CACHING = True
CACHE_DIR = Path(DATA_DIR) / "cache"
CACHE_EXPIRY = 24 * 60 * 60  # 24 hours in seconds

# Visualization settings
DEFAULT_VIZ_MODE = "prediction"
DEFAULT_VIZ_STYLE = "cartoon"
VIZ_COLORS = {
    "base": "#e6e6f7",
    "epitope": "#9C6ADE",
    "true_positive": "#a0d293",
    "false_positive": "#ef5331",
    "true_negative": "#f1b54c",
    "regions": ["#FF6B6B", "#96CEB4", "#4ECDC4", "#45B7D1", "#FFEAA7", "#DDA0DD", "#87CEEB"]
}

# Database settings (for future use)
DATABASE_URL = os.getenv("BCE_DATABASE_URL", "sqlite:///./bce_predictions.db")

# Redis settings (for production task queue)
REDIS_URL = os.getenv("BCE_REDIS_URL", "redis://localhost:6379")

# Performance settings
USE_GPU = True
GPU_MEMORY_FRACTION = 0.8
BATCH_SIZE = 1  # Process one protein at a time for web interface

def get_config():
    """
    Get configuration dictionary
    """
    return {
        "base_dir": BASE_DIR,
        "model_path": DEFAULT_MODEL_PATH,
        "data_dir": DATA_DIR,
        "device_id": DEFAULT_DEVICE_ID,
        "radius": DEFAULT_RADIUS,
        "k": DEFAULT_K,
        "encoder": DEFAULT_ENCODER,
        "chain_id": DEFAULT_CHAIN_ID,
        "max_upload_size": MAX_UPLOAD_SIZE,
        "task_timeout": TASK_TIMEOUT,
        "esm_token": ESM_TOKEN,
        "esm_model_name": ESM_MODEL_NAME,
        "esm_url": ESM_URL,
    }

def validate_config():
    """
    Validate configuration settings
    """
    errors = []
    
    # Check if base directories exist
    if not BASE_DIR.exists():
        errors.append(f"Base directory does not exist: {BASE_DIR}")
    
    # Check if model file exists (if specified)
    if DEFAULT_MODEL_PATH and not Path(DEFAULT_MODEL_PATH).exists():
        errors.append(f"Model file does not exist: {DEFAULT_MODEL_PATH}")
    
    # Check if data directory exists
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        try:
            data_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create data directory: {e}")
    
    # Check GPU availability if specified
    if USE_GPU:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("GPU requested but CUDA is not available")
        except ImportError:
            errors.append("PyTorch not installed but GPU usage requested")
    
    # Check ESM-C token if using esmc encoder
    if DEFAULT_ENCODER == "esmc" and not ESM_TOKEN:
        errors.append("ESM token is required for ESM-C encoder. Set ESM_TOKEN environment variable.")
    
    # Test ESM-C SDK availability if using esmc encoder
    if DEFAULT_ENCODER == "esmc":
        try:
            from esm.sdk.api import ESMProtein, LogitsConfig
            from esm.sdk.forge import ESM3ForgeInferenceClient
        except ImportError:
            errors.append("ESM-C SDK not available. Please install with: pip install fair-esm[esmfold]")
    
    return errors

# Environment-specific settings
if DEBUG_MODE:
    # Development settings
    CORS_ORIGINS = ["*"]
    MAX_CONCURRENT_TASKS = 2
    TASK_TIMEOUT = 10 * 60  # 10 minutes for development
else:
    # Production settings
    CORS_ORIGINS = ["https://your-domain.com"]  # Update for production
    MAX_CONCURRENT_TASKS = 10
    TASK_TIMEOUT = 30 * 60  # 30 minutes 