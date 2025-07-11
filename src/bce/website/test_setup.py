#!/usr/bin/env python3
"""
Test script to verify BCE Prediction Web Server setup
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('jinja2', 'Jinja2'),
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('h5py', 'h5py'),
        ('tqdm', 'tqdm'),
        ('biotite', 'Biotite'),
        ('Bio', 'BioPython'),
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed_imports.append(name)
    
    return failed_imports

def test_esm_sdk():
    """Test ESM-C SDK availability"""
    print("\n🔍 Testing ESM-C SDK...")
    
    try:
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.sdk.forge import ESM3ForgeInferenceClient
        print("  ✅ ESM-C SDK import successful")
        
        # Test basic protein creation
        protein = ESMProtein(sequence="ACDEFGHIKLMNPQRSTVWY")
        print("  ✅ ESMProtein creation successful")
        
        return True
    except Exception as e:
        print(f"  ❌ ESM-C SDK test failed: {e}")
        return False

def test_torch_cuda():
    """Test PyTorch and CUDA availability"""
    print("\n🔍 Testing PyTorch and CUDA...")
    
    try:
        import torch
        print(f"  ✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                print(f"    Device {i}: {device_name}")
        else:
            print("  ⚠️  CUDA not available (will use CPU)")
        
        # Test tensor creation
        x = torch.randn(2, 3)
        print(f"  ✅ Tensor creation successful")
        
        return True
    except Exception as e:
        print(f"  ❌ PyTorch test failed: {e}")
        return False

def test_project_structure():
    """Test if project structure is correct"""
    print("\n🔍 Testing project structure...")
    
    current_dir = Path(__file__).parent
    required_files = [
        'main.py',
        'run_server.py',
        'requirements.txt',
        'config.py',
        'templates/index.html',
        'static/css/style.css',
        'static/js/app.js'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    return missing_files

def test_antigen_module():
    """Test if the antigen module can be imported"""
    print("\n🔍 Testing antigen module...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parents[3]
        sys.path.insert(0, str(project_root))
        
        from src.bce.antigen.antigen import AntigenChain
        print("  ✅ AntigenChain import successful")
        
        # Test static methods
        result = AntigenChain.convert_letter_1to3('A')
        if result == 'ALA':
            print("  ✅ Static method test successful")
        else:
            print(f"  ❌ Static method test failed: expected 'ALA', got '{result}'")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Antigen module test failed: {e}")
        print("  ℹ️  This is expected if not in the correct environment")
        return True  # Don't fail the test for this
    
def test_config():
    """Test configuration settings"""
    print("\n🔍 Testing configuration...")
    
    try:
        import config
        
        # Test configuration validation
        errors = config.validate_config()
        if errors:
            print("  ⚠️  Configuration warnings:")
            for error in errors:
                print(f"    - {error}")
        else:
            print("  ✅ Configuration validation passed")
        
        # Test basic config values
        config_dict = config.get_config()
        print(f"  ✅ Base directory: {config_dict['base_dir']}")
        print(f"  ✅ Model path: {config_dict['model_path']}")
        print(f"  ✅ Data directory: {config_dict['data_dir']}")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_fastapi_basic():
    """Test basic FastAPI functionality without importing main"""
    print("\n🔍 Testing FastAPI basic functionality...")
    
    try:
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
        
        # Create a test app
        app = FastAPI(title="Test App")
        print("  ✅ FastAPI app creation successful")
        
        # Test static files and templates
        import config
        if config.STATIC_DIR.exists() and config.TEMPLATES_DIR.exists():
            print("  ✅ Static and template directories exist")
        else:
            print("  ⚠️  Static or template directories missing")
        
        return True
    except Exception as e:
        print(f"  ❌ FastAPI basic test failed: {e}")
        return False

def test_web_dependencies():
    """Test web-specific dependencies"""
    print("\n🔍 Testing web dependencies...")
    
    web_packages = [
        ('multipart', 'python-multipart'),
        ('aiofiles', 'aiofiles'),
    ]
    
    failed_imports = []
    
    for package, name in web_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0

def main():
    """Run all tests"""
    print("🚀 BCE Prediction Web Server Setup Test (ReCEP Environment)")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    failed_imports = test_imports()
    if failed_imports:
        print(f"\n❌ Some imports failed: {', '.join(failed_imports)}")
        all_tests_passed = False
    
    if not test_esm_sdk():
        print("\n⚠️  ESM-C SDK test failed")
        # Don't fail completely for this
    
    if not test_torch_cuda():
        all_tests_passed = False
    
    missing_files = test_project_structure()
    if missing_files:
        print(f"\n❌ Some files are missing: {', '.join(missing_files)}")
        all_tests_passed = False
    
    test_antigen_module()  # Don't fail on this
    
    if not test_config():
        all_tests_passed = False
    
    if not test_fastapi_basic():
        all_tests_passed = False
    
    if not test_web_dependencies():
        print("\n⚠️  Some web dependencies missing")
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 Core tests passed! The setup looks good.")
        print("\n📋 Next steps:")
        print("  1. Make sure you're in the ReCEP environment:")
        print("     conda activate ReCEP")
        print("  2. Install web dependencies:")
        print("     pip install fastapi uvicorn python-multipart jinja2 aiofiles")
        print("  3. Run the server:")
        print("     python run_server.py")
        print("  4. Open: http://localhost:8000")
        print("  5. Try a prediction with PDB ID '1FBI' chain 'A'")
        print("\n⚠️  Note: Ensure you have a valid ESM token for ESM-C predictions")
    else:
        print("⚠️  Some critical tests failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("  - Activate ReCEP environment: conda activate ReCEP")
        print("  - Install missing packages: pip install fastapi uvicorn python-multipart")
        print("  - Check file paths and permissions")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 