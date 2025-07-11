from pathlib import Path
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .dihedral import DihedralFeatures
from .EGNN import EGNNLayer
from .activation import get_activation

class EP(nn.Module):
    """
    Epitope Predictor - Residue-based Binary Classification Only.
    
    This model performs only node-level (residue-level) epitope prediction,
    without global protein-level predictions.
    """
    def __init__(
        self,
        in_dim: int = 2560,
        rsa: bool = True,
        dihedral: bool = True,
        node_dims: list = [1024, 512, 256],
        edge_dim: int = 32,
        dropout: float = 0.4,
        activation: str = "gelu",
        residual: bool = True,
        attention: bool = True,
        normalize: bool = True,
        coords_agg: str = 'mean',
        ffn: bool = True,
        batch_norm: bool = True,
        concat: bool = True,
        # Node classifier
        node_norm: bool = True,
        node_layers: int = 2,
        out_dropout: float = 0.2,
        use_egnn: bool = True,
        encoder: str = 'esmc',
    ):
        super().__init__()
        self.use_egnn = use_egnn
        self.in_dim = in_dim
        self.rsa = rsa
        self.dihedral = dihedral
        self.original_node_dims = node_dims.copy()
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.activation = activation
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.ffn = ffn
        self.batch_norm = batch_norm
        self.coords_agg = coords_agg
        self.node_norm = node_norm
        self.node_layers = node_layers
        self.out_dropout = out_dropout
        self.concat = concat
        self.base_node_dim = node_dims[0]
        self.node_dims = node_dims.copy()
        self.node_dims[0] += 1 if rsa else 0
            
        # Modify input dimension based on encoder
        self.encoder = encoder
        if encoder == 'esmc':
            self.in_dim = 2560
        elif encoder == 'esm2':
            self.in_dim = 1280
        else:
            self.in_dim = in_dim
        
        # Calculate final node dimension
        if self.use_egnn:
            self.final_node_dim = self.node_dims[-1]
        else:
            self.final_node_dim = self.node_dims[0]
            self.concat = False  # Disable concat if EGNN is not used
            
        # Adjust final dimension for concat
        if self.concat and self.use_egnn:
            self.final_node_dim += self.node_dims[0]

        # Input projection layer
        self.proj_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.base_node_dim),
            get_activation(activation),
            nn.Dropout(out_dropout),
        )

        # Dihedral features (optional)
        if dihedral:
            try:
                self.dihedral_features = DihedralFeatures(self.base_node_dim)
            except:
                print("Warning: DihedralFeatures not found, skipping dihedral features")
                self.dihedral = False

        # EGNN layers for geometric message passing
        self.egnn_layers = nn.ModuleList()
        if self.use_egnn:
            for i in range(len(self.node_dims) - 1):
                self.egnn_layers.append(
                    EGNNLayer(
                        input_nf=self.node_dims[i],
                        output_nf=self.node_dims[i+1],
                        hidden_nf=self.node_dims[i+1],
                        edges_in_d=edge_dim,
                        act_fn=get_activation(activation),
                        residual=residual,
                        attention=attention,
                        normalize=normalize,
                        coords_agg=coords_agg,
                        dropout=dropout,
                        ffn=ffn,
                        batch_norm=batch_norm
                    )
                )

        # Node classifier for epitope prediction
        self.node_classifier = self._build_node_classifier()

        self._param_printed = False
        self.apply(self._init_weights)

    def _build_node_classifier(self):
        """Build the node classifier for epitope prediction."""
        layers = []
        input_dim = self.final_node_dim
        current_dim = input_dim
        
        for i in range(self.node_layers):
            output_dim = 1 if i == self.node_layers - 1 else max(current_dim // 2, 32)
            layers.append(nn.Linear(current_dim, output_dim))
            
            if self.node_norm and i < self.node_layers - 1:
                layers.append(nn.LayerNorm(output_dim))
            if i < self.node_layers - 1:
                layers.append(get_activation(self.activation))
                layers.append(nn.Dropout(self.out_dropout))
            current_dim = output_dim
            
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0 if module.out_features == 1 else 0.01)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, data: Batch) -> dict:
        """
        Forward pass for epitope prediction.
        
        Args:
            data: Batch of protein graphs
            
        Returns:
            dict: Contains 'node_preds' for residue-level epitope predictions
        """
        if self.training and not self._param_printed:
            print(f"EP total params: {sum(p.numel() for p in self.parameters()):,}")
            self._param_printed = True

        # Select embeddings based on encoder
        if self.encoder == 'esm2':
            x = data.embeddings2
        else:
            x = data.x
    
        coords = data.pos
        batch = data.batch
        e_attr = data.edge_attr
        coords_C = coords[:, 1].clone() if coords is not None else None

        # Project input embeddings
        x = self.proj_layer(x)
        
        # Add dihedral features if available
        if self.dihedral and coords is not None:
            x = x + self.dihedral_features(coords)
            
        # Add RSA features if available
        if self.rsa and hasattr(data, 'rsa') and data.rsa is not None:
            rsa = data.rsa.unsqueeze(-1)
            x = torch.cat([x, rsa], dim=-1)

        h = x
        assert h.shape[1] == self.node_dims[0], f"[EP] Node feature dim mismatch: got {h.shape[1]}, expected {self.node_dims[0]}"

        # Apply EGNN layers for geometric message passing
        if self.use_egnn:
            for layer in self.egnn_layers:
                h, coords_C, _ = layer(h, coords_C, data.edge_index, batch, edge_attr=e_attr)

        # Concatenate input features with final features if concat is enabled
        if self.concat and self.use_egnn:
            h = torch.cat([x, h], dim=-1)

        # Node-level epitope prediction
        node_preds = self.node_classifier(h).squeeze(-1)

        return {"node_preds": node_preds}
    
    def print_param_count(self):
        """Print a summary table of parameter counts"""
        table = PrettyTable()
        table.field_names = ["Layer Name", "Type", "Parameters", "Trainable"]
        total_params = 0
        trainable_params = 0

        for name, module in self.named_modules():
            if not list(module.children()):  # Only leaf nodes
                params = sum(p.numel() for p in module.parameters())
                is_trainable = any(p.requires_grad for p in module.parameters())
                
                if params > 0:
                    total_params += params
                    trainable_params += params if is_trainable else 0
                    
                    table.add_row([
                        name,
                        module.__class__.__name__,
                        f"{params:,}",
                        "✓" if is_trainable else "✗"
                    ])

        table.add_row(["", "", "", ""], divider=True)
        table.add_row([
            "TOTAL", 
            "", 
            f"{total_params:,}",
            f"Trainable: {trainable_params:,}"
        ])

        print("\nEP Model Parameter Summary:")
        print(table)
        print(f"Parameter Density: {trainable_params/total_params:.1%}\n")
    
    def save(self, path, threshold: float = 0.5):
        """Save model with configuration"""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_path = path.with_suffix('.bin')
            
            config = self.get_config()
            # config = {
            #     'in_dim': self.in_dim,
            #     'rsa': self.rsa,
            #     'dihedral': self.dihedral,
            #     'node_dims': self.original_node_dims,  # Use original node_dims
            #     'edge_dim': self.edge_dim,
            #     'dropout': self.dropout,
            #     'activation': self.activation,
            #     'residual': self.residual,
            #     'attention': self.attention,
            #     'normalize': self.normalize,
            #     'coords_agg': self.coords_agg,
            #     'ffn': self.ffn,
            #     'batch_norm': self.batch_norm,
            #     'concat': self.concat,
            #     'node_norm': self.node_norm,
            #     'node_layers': self.node_layers,
            #     'node_gate': self.node_gate,
            #     'out_dropout': self.out_dropout
            # }
            
            torch.save({
                'model_state': self.state_dict(),
                'config': config,
                'model_class': self.__class__.__name__,
                'version': '1.0',
                'threshold': threshold
            }, save_path)
            print(f"EP model saved to {save_path}")
        except Exception as e:
            print(f"Save failed: {str(e)}")
            raise
    
    @classmethod
    def load(cls, path, device='cpu', strict=True, verbose=True):
        """Load model with configuration"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} not found")

        try:
            if isinstance(device, str):
                device = torch.device(device)
            elif isinstance(device, int):
                if device >= 0 and torch.cuda.is_available():
                    device = torch.device(f'cuda:{device}')
                else:
                    device = torch.device('cpu')
            elif not isinstance(device, torch.device):
                raise ValueError(f"Unsupported device type: {type(device)}")
            
            checkpoint = torch.load(
                path,
                map_location=device,
                weights_only=False
            )
        except RuntimeError:
            print("Warning: Using unsafe load due to weights_only restriction")
            checkpoint = torch.load(path, map_location=device)

        # Version compatibility check
        if 'version' not in checkpoint:
            print("Warning: Loading legacy model without version info")
        
        # Rebuild configuration
        config = checkpoint.get('config', {})
        model = cls(**config)
        
        # Load state dict
        model_state = checkpoint['model_state']
        current_state = model.state_dict()
        
        # Auto-match parameters
        matched_state = {}
        for name, param in model_state.items():
            if name in current_state:
                if param.shape == current_state[name].shape:
                    matched_state[name] = param
                else:
                    print(f"Size mismatch: {name} (load {param.shape} vs model {current_state[name].shape})")
            else:
                print(f"Parameter not found: {name}")
        
        current_state.update(matched_state)
        model.load_state_dict(current_state, strict=strict)
        
        if verbose:
            print(f"Successfully loaded {len(matched_state)}/{len(model_state)} parameters")
        
        return model.to(device), checkpoint.get('threshold', 0.5)
    
    def get_config(self):
        """Get model configuration"""
        return {
            'in_dim': self.in_dim,
            'rsa': self.rsa,
            'dihedral': self.dihedral,
            'node_dims': self.original_node_dims,
            'edge_dim': self.edge_dim,
            'dropout': self.dropout,
            'activation': self.activation,
            'residual': self.residual,
            'attention': self.attention,
            'normalize': self.normalize,
            'coords_agg': self.coords_agg,
            'ffn': self.ffn,
            'batch_norm': self.batch_norm,
            'concat': self.concat,
            'node_norm': self.node_norm,
            'node_layers': self.node_layers,
            'out_dropout': self.out_dropout,
            'use_egnn': self.use_egnn,
            'encoder': self.encoder
        }
        