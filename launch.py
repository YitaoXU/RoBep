#!/usr/bin/env python3
"""
Launch script for Hugging Face Spaces deployment
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set environment variables for Hugging Face Spaces
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "7860"

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Try to import and run the main app
try:
    from app import create_interface
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Don't create public links on Spaces
        show_error=True,
        show_tips=True,
        enable_queue=True,
        max_threads=4
    )
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed correctly.")
    sys.exit(1)
except Exception as e:
    print(f"Error launching application: {e}")
    sys.exit(1) 