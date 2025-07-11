#!/usr/bin/env python3
"""
诊断BCE服务器启动慢的问题
"""

import time
import sys
import torch
from pathlib import Path

def test_import_time():
    """测试各个模块的导入时间"""
    print("🔍 Testing import times...")
    
    tests = [
        ("fastapi", "import fastapi"),
        ("uvicorn", "import uvicorn"),
        ("torch", "import torch"),
        ("biotite", "import biotite.structure as bs"),
        ("esm", "from esm.utils.structure.protein_chain import ProteinChain"),
    ]
    
    for name, import_stmt in tests:
        start_time = time.time()
        try:
            exec(import_stmt)
            import_time = time.time() - start_time
            status = "✅" if import_time < 1.0 else "⚠️" if import_time < 5.0 else "❌"
            print(f"  {status} {name}: {import_time:.2f}s")
        except Exception as e:
            print(f"  ❌ {name}: Failed - {str(e)}")

def test_cuda_initialization():
    """测试CUDA初始化时间"""
    print("\n🎯 Testing CUDA initialization...")
    
    # Test CUDA availability check
    start_time = time.time()
    cuda_available = torch.cuda.is_available()
    check_time = time.time() - start_time
    print(f"  CUDA available check: {check_time:.2f}s")
    
    if cuda_available:
        # Test device creation
        start_time = time.time()
        device = torch.device("cuda:0")
        device_time = time.time() - start_time
        print(f"  Device creation: {device_time:.2f}s")
        
        # Test GPU name query
        start_time = time.time()
        gpu_name = torch.cuda.get_device_name(0)
        name_time = time.time() - start_time
        print(f"  GPU name query: {name_time:.2f}s")
        print(f"  GPU: {gpu_name}")
        
        # Test tensor creation and move to GPU
        start_time = time.time()
        tensor = torch.randn(1000, 1000).to(device)
        torch.cuda.synchronize()
        tensor_time = time.time() - start_time
        print(f"  Tensor to GPU: {tensor_time:.2f}s")
        
        # Test memory info
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("  ❌ CUDA not available")

def test_model_loading():
    """测试模型加载时间"""
    print("\n🤖 Testing model loading...")
    
    try:
        # Add project root to path
        script_dir = Path(__file__).parent
        project_root = script_dir.parents[2]
        sys.path.insert(0, str(project_root))
        
        from bce.model.ReCEP import ReCEP
        from bce.utils.constants import BASE_DIR
        
        model_path = f"{BASE_DIR}/models/ReCEP/20250626_110438/best_mcc_model.bin"
        
        if not Path(model_path).exists():
            print(f"  ❌ Model file not found: {model_path}")
            return
        
        # Test model file size
        file_size = Path(model_path).stat().st_size / 1e6  # MB
        print(f"  Model file size: {file_size:.1f}MB")
        
        # Test model loading time
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        start_time = time.time()
        model, threshold = ReCEP.load(model_path, device=device, strict=False, verbose=False)
        load_time = time.time() - start_time
        
        status = "✅" if load_time < 5.0 else "⚠️" if load_time < 15.0 else "❌"
        print(f"  {status} Model loading: {load_time:.2f}s")
        print(f"  Model threshold: {threshold:.4f}")
        
        # Test model evaluation mode
        start_time = time.time()
        model.eval()
        eval_time = time.time() - start_time
        print(f"  Model eval mode: {eval_time:.2f}s")
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {str(e)}")

def test_esmc_connection():
    """测试ESM-C连接"""
    print("\n🌐 Testing ESM-C connection...")
    
    try:
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.sdk.forge import ESM3ForgeInferenceClient
        
        # Test API client creation
        start_time = time.time()
        client = ESM3ForgeInferenceClient(
            model="esmc-6b-2024-12",
            url="https://forge.evolutionaryscale.ai",
            token="1mzAo8l1uxaU8UfVcGgV7B"
        )
        client_time = time.time() - start_time
        
        status = "✅" if client_time < 2.0 else "⚠️" if client_time < 5.0 else "❌"
        print(f"  {status} API client creation: {client_time:.2f}s")
        
        # Test a small sequence (optional - may take time)
        print("  ℹ️  Skipping actual API call to avoid delays")
        
    except Exception as e:
        print(f"  ❌ ESM-C test failed: {str(e)}")

def test_disk_io():
    """测试磁盘I/O性能"""
    print("\n💾 Testing disk I/O performance...")
    
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parents[2]
        
        # Test creating a temporary file
        temp_file = project_root / "temp_test_file.txt"
        
        start_time = time.time()
        with open(temp_file, "w") as f:
            f.write("test" * 10000)  # 40KB
        write_time = time.time() - start_time
        
        start_time = time.time()
        with open(temp_file, "r") as f:
            content = f.read()
        read_time = time.time() - start_time
        
        # Clean up
        temp_file.unlink()
        
        print(f"  Write speed: {write_time:.3f}s (40KB)")
        print(f"  Read speed: {read_time:.3f}s (40KB)")
        
        if write_time > 0.1 or read_time > 0.1:
            print("  ⚠️  Slow disk I/O detected")
        else:
            print("  ✅ Disk I/O normal")
            
    except Exception as e:
        print(f"  ❌ Disk I/O test failed: {str(e)}")

def test_system_resources():
    """测试系统资源"""
    print("\n📊 System resources...")
    
    try:
        import psutil
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"  CPU cores: {cpu_count}")
        print(f"  CPU usage: {cpu_usage:.1f}%")
        
        # Memory info
        memory = psutil.virtual_memory()
        print(f"  Total RAM: {memory.total / 1e9:.1f}GB")
        print(f"  Available RAM: {memory.available / 1e9:.1f}GB")
        print(f"  Memory usage: {memory.percent:.1f}%")
        
        # Disk info
        disk = psutil.disk_usage('/')
        print(f"  Disk usage: {disk.percent:.1f}%")
        print(f"  Free disk: {disk.free / 1e9:.1f}GB")
        
    except ImportError:
        print("  ℹ️  psutil not available, skipping system resource check")
    except Exception as e:
        print(f"  ❌ System resource check failed: {str(e)}")

def main():
    print("🔍 BCE Server Startup Diagnostics")
    print("=" * 50)
    
    overall_start = time.time()
    
    test_system_resources()
    test_disk_io()
    test_import_time()
    test_cuda_initialization()
    test_model_loading()
    test_esmc_connection()
    
    overall_time = time.time() - overall_start
    
    print("\n" + "=" * 50)
    print(f"🏁 Total diagnostic time: {overall_time:.2f}s")
    
    print("\n💡 Recommendations:")
    print("  • Use --preload flag to preload model on server startup")
    print("  • Consider caching embeddings for frequently used proteins")
    print("  • Use SSD storage for faster model loading")
    print("  • Ensure stable internet connection for ESM-C API")

if __name__ == "__main__":
    main() 