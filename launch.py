#!/usr/bin/env python3
"""
Launch script for B-cell Epitope Prediction Server
Allows users to choose between Gradio and FastAPI interfaces
"""

import sys
import os
import argparse
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🧬 B-cell Epitope Prediction Server")
    print("   AI-powered epitope prediction using ReCEP model")
    print("=" * 60)
    print()

def launch_gradio():
    """Launch the Gradio interface"""
    print("🚀 Starting Gradio Interface...")
    print("📍 Interface will be available at: http://localhost:7860")
    print("✨ Features: Advanced UI with progress tracking")
    print()
    
    # Import and run the Gradio app
    import app
    interface = app.create_interface()
    
    # Check if running on Hugging Face Spaces
    is_spaces = os.getenv("SPACE_ID") is not None
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=is_spaces,
        show_error=True,
        max_threads=4 if is_spaces else 8
    )

def launch_fastapi():
    """Launch the FastAPI interface"""
    print("🚀 Starting FastAPI Interface...")
    print("📍 Interface will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("✨ Features: RESTful API with interactive web interface")
    print()
    
    # Import and run the FastAPI app
    import uvicorn
    import fastapi_app
    
    uvicorn.run(
        fastapi_app.app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True
    )

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Launch B-cell Epitope Prediction Server")
    parser.add_argument(
        "--interface", 
        choices=["gradio", "fastapi", "interactive"],
        default="interactive",
        help="Interface type to launch (default: interactive)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number (7860 for Gradio, 8000 for FastAPI)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.interface == "gradio":
        launch_gradio()
    elif args.interface == "fastapi":
        launch_fastapi()
    else:
        # Interactive selection
        print("Please choose an interface:")
        print("1. Gradio Interface (Recommended for beginners)")
        print("   - User-friendly web interface")
        print("   - Progress tracking")
        print("   - Built-in file handling")
        print("   - Port: 7860")
        print()
        print("2. FastAPI Interface (Recommended for developers)")
        print("   - RESTful API")
        print("   - Interactive documentation")
        print("   - Programmatic access")
        print("   - Port: 8000")
        print()
        
        while True:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == "1":
                launch_gradio()
                break
            elif choice == "2":
                launch_fastapi()
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 