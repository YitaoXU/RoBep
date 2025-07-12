#!/usr/bin/env python3
"""
Launch script for the BCE Prediction Web Server
"""

import argparse
import uvicorn
import sys
import time
import torch
from pathlib import Path

def preload_model(device_id=-1, model_path=None, verbose=True):
    """
    预加载ReCEP模型以减少首次请求延迟
    """
    try:
        if verbose:
            print("🔄 Preloading ReCEP model...")
        
        # Add project root to path
        script_dir = Path(__file__).parent
        project_root = script_dir.parents[2]
        sys.path.insert(0, str(project_root))
        
        from bce.model.ReCEP import ReCEP
        from bce.utils.constants import BASE_DIR
        
        # 设置设备
        if device_id >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{device_id}")
            if verbose:
                print(f"🎯 Using GPU: {torch.cuda.get_device_name(device_id)}")
        else:
            device = torch.device("cpu")
            if verbose:
                print("🎯 Using CPU")
        
        # 使用默认模型路径如果未指定
        if model_path is None:
            model_path = f"{BASE_DIR}/models/ReCEP/20250626_110438/best_mcc_model.bin"
        
        start_time = time.time()
        
        # 加载模型
        model, threshold = ReCEP.load(model_path, device=device, strict=False, verbose=False)
        model.eval()
        
        # 预热GPU（如果使用GPU）
        if device.type == 'cuda':
            # 创建一个小的测试张量来预热GPU
            dummy_tensor = torch.randn(10, 512).to(device)
            with torch.no_grad():
                _ = dummy_tensor.sum()
            del dummy_tensor
            torch.cuda.synchronize()
        
        load_time = time.time() - start_time
        
        if verbose:
            print(f"✅ Model preloaded successfully in {load_time:.2f}s")
            print(f"📏 Model threshold: {threshold:.4f}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Model preload failed: {str(e)}")
            print("   Server will load model on first request")
        return False

def main():
    parser = argparse.ArgumentParser(description="Launch BCE Prediction Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"],
                       help="Log level (default: info)")
    parser.add_argument("--preload", action="store_true", default=True, help="Preload model on startup (default: True)")
    parser.add_argument("--no-preload", action="store_true", help="Skip model preloading")
    parser.add_argument("--device-id", type=int, default=-1, help="Device ID for model preloading (default: -1 for CPU)")
    parser.add_argument("--model-path", type=str, default=None, help="Custom model path for preloading")
    
    args = parser.parse_args()
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    
    # Change to the website directory to ensure relative imports work
    import os
    original_cwd = os.getcwd()
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {script_dir}")
    print(f"🐍 Python path includes: {script_dir}")
    
    print(f"🚀 Starting BCE Prediction Server...")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    print(f"📖 API documentation at: http://{args.host}:{args.port}/docs")
    print(f"🔄 Auto-reload: {'enabled' if args.reload else 'disabled'}")
    
    # 模型预加载
    should_preload = args.preload and not args.no_preload and not args.reload
    if should_preload:
        success = preload_model(
            device_id=args.device_id,
            model_path=args.model_path,
            verbose=True
        )
        if success:
            print("🎉 Server ready for fast predictions!")
        else:
            print("⚠️  Server starting without preload")
    else:
        if args.reload:
            print("ℹ️  Skipping preload in reload mode")
        else:
            print("ℹ️  Skipping model preload (use --preload to enable)")
    
    print("-" * 50)
    
    # Run the server - Use direct import to avoid module resolution issues
    print("🔧 Starting server with direct import method...")
    
    # Ensure the website directory is first in Python path to avoid conflicts
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    # Remove any conflicting paths that might have main.py
    project_root = script_dir.parents[2]
    if str(project_root) in sys.path:
        sys.path.remove(str(project_root))
    
    print(f"🔍 Python path: {sys.path[:3]}...")  # Show first 3 paths
    
    try:
        import main
        print(f"📄 Imported main from: {main.__file__}")
        app = main.app
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level,
            access_log=True
        )
    except Exception as e:
        print(f"❌ Direct import failed: {str(e)}")
        print("💡 Trying string-based import...")
        
        # Fallback: try string-based import
        try:
            uvicorn.run(
                "main:app",
                host=args.host,
                port=args.port,
                reload=args.reload,
                workers=args.workers if not args.reload else 1,
                log_level=args.log_level,
                access_log=True
            )
        except Exception as e2:
            print(f"❌ String-based import also failed: {str(e2)}")
            print("🔧 Please check your environment and dependencies")

if __name__ == "__main__":
    main() 