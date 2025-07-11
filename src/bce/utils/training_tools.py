import random
import numpy as np

import torch

def parse_range(range_str):
    """Parse range string in format 'start:end:step' into a list of values"""
    try:
        start, end, step = map(float, range_str.split(':'))
        # Add a small epsilon to ensure end value is included
        epsilon = step / 100
        return list(np.arange(start, end + epsilon, step))
    except ValueError:
        # If only one value, return a list containing that value
        try:
            value = float(range_str)
            return [value]
        except ValueError:
            raise ValueError(f"Invalid range format: {range_str}. Use 'start:end:step' or single value.")

def setup_device(device_id=1):
    """Setup and verify CUDA device."""
    if device_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        if not hasattr(setup_device, '_printed'):
            print(f"[INFO] Using device: {device}")
            print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
            setup_device._printed = True
        return device_id
    else:
        if not hasattr(setup_device, '_printed'):
            print(f"[INFO] Using device: cpu")
            setup_device._printed = True
        return -1  # Return -1 to indicate CPU usage

def set_seed(seed, deterministic=False):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed (int): Random seed value
        deterministic (bool): Whether to enable deterministic mode in PyTorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    if deterministic:
        # These settings may impact performance but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[INFO] Deterministic mode enabled (may impact performance)")
    
    print(f"[INFO] Random seed set to {seed} for reproducibility")