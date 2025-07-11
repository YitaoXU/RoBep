from torch_scatter import scatter_softmax, scatter_sum
import torch
import torch.nn as nn

from .activation import get_activation

class AttentionPooling(nn.Module):
    """Attention-based pooling layer supporting batched graphs."""
    def __init__(self, input_dim: int, dropout: float = 0.2, activation: str = 'gelu'):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (N, input_dim) Node features from multiple graphs
            batch: (N,) Graph ID per node
        Returns:
            (num_graphs, input_dim) Pooled graph features
        """
        attn_logits = self.attention(x).squeeze(-1)       # (N,)
        attn_weights = scatter_softmax(attn_logits, batch)  # (N,)
        weighted_x = x * attn_weights.unsqueeze(-1)       # (N, D)
        pooled = scatter_sum(weighted_x, batch, dim=0)    # (num_graphs, D)
        return self.dropout(pooled)

class AddPooling(nn.Module):
    """Simple addition-based pooling layer supporting batched graphs."""
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (N, input_dim) Node features from multiple graphs
            batch: (N,) Graph ID per node
        Returns:
            (num_graphs, input_dim) Pooled graph features
        """
        pooled = scatter_sum(x, batch, dim=0)    # (num_graphs, input_dim)
        return self.dropout(pooled)