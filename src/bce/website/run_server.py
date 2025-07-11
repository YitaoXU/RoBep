#!/usr/bin/env python3
"""
Launch script for the BCE Prediction Web Server
"""

import argparse
import uvicorn
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Launch BCE Prediction Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"],
                       help="Log level (default: info)")
    
    args = parser.parse_args()
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    
    print(f"🚀 Starting BCE Prediction Server...")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    print(f"📖 API documentation at: http://{args.host}:{args.port}/docs")
    print(f"🔄 Auto-reload: {'enabled' if args.reload else 'disabled'}")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # reload doesn't work with multiple workers
        log_level=args.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main() 