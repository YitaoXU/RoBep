import torch.nn as nn

def get_activation(activation: str) -> nn.Module:
    """Get activation function by name."""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")