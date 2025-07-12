from pathlib import Path
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_sum
from torch_geometric.data import Data, Batch

from .dihedral import DihedralFeatures
from .EGNN import EGNNLayer
from .pooling import AttentionPooling, AddPooling
from .activation import get_activation
from .baseline import EP

class ReCEP(nn.Module):
    """
    Refined Graph Epitope Predictor with optional EGNN layer skipping for ablation.
    """
    def __init__(
        self,
        in_dim: int = 2560,
        rsa: bool = True,
        dihedral: bool = True,
        node_dims: list = [512, 256, 256],
        edge_dim: int = 32,
        dropout: float = 0.3,
        activation: str = "gelu",
        residual: bool = True,
        attention: bool = True,
        normalize: bool = True,
        coords_agg: str = 'mean',
        ffn: bool = True,
        batch_norm: bool = True,
        concat: bool = False,
        addition: bool = False,
        # Global predictor
        pooling: str = 'attention',
        # Node classifier
        fusion_type: str = 'concat',
        node_gate: bool = False,
        node_norm: bool = False,
        node_layers: int = 2,
        out_dropout: float = 0.2,
        use_egnn: bool = True,  # NEW: toggle for EGNN layer usage
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
        self.concat = concat
        self.addition = addition
        self.fusion_type = fusion_type
        self.node_gate = node_gate
        self.node_norm = node_norm
        self.node_layers = node_layers
        self.out_dropout = out_dropout
        self.pooling = pooling

        self.base_node_dim = node_dims[0]
        self.node_dims = node_dims.copy()
        self.node_dims[0] += 1 if rsa else 0
        
        self.node_dims[-1] = self.node_dims[0] if addition else self.node_dims[-1]
        
        # Modify input dimension based on encoder
        self.encoder = encoder
        if encoder == 'esmc':
            self.in_dim = 2560
        elif encoder == 'esm2':
            self.in_dim = 1280
        else:
            self.in_dim = in_dim
        
        # Calculate actual final node dimension based on whether EGNN is used
        if self.use_egnn:
            self.final_node_dim = self.node_dims[-1]
        else:
            self.final_node_dim = self.node_dims[0]
            self.concat = False
            self.addition = False

        self.proj_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.base_node_dim),
            get_activation(activation),
            nn.Dropout(dropout),
        )

        if dihedral:
            try:
                self.dihedral_features = DihedralFeatures(self.base_node_dim)
            except:
                print("Warning: DihedralFeatures not found, skipping dihedral features")
                self.dihedral = False

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

        if concat and self.use_egnn:
            self.final_node_dim += self.node_dims[0]
            
        if addition and self.use_egnn:
            assert self.node_dims[0] == self.node_dims[-1], "Node dimension mismatch for addition"
            self.final_node_dim = self.node_dims[0]
            
        # Calculate node classifier input dimension based on fusion type
        if fusion_type == 'concat':
            self.node_classifier_input_dim = self.final_node_dim * 2
        elif fusion_type == 'add':
            self.node_classifier_input_dim = self.final_node_dim
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        # Calculate node gate input dimension
        if node_gate:
            if fusion_type == 'concat':
                self.node_gate_input_dim = self.final_node_dim * 2
            elif fusion_type == 'add':
                self.node_gate_input_dim = self.final_node_dim
            else:
                raise ValueError(f"Unsupported fusion type: {fusion_type}")

        if pooling == 'attention':
            self.graph_pool = AttentionPooling(
                input_dim=self.final_node_dim,
                dropout=dropout,
                activation=activation
            )
        elif pooling == 'add':
            self.graph_pool = AddPooling(
                input_dim=self.final_node_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")

        self.global_predictor = nn.Sequential(
            nn.Linear(self.final_node_dim, self.final_node_dim // 2),
            get_activation(activation),
            nn.Dropout(out_dropout),
            nn.Linear(self.final_node_dim // 2, 1)
        )

        if node_gate:
            self.node_gate = nn.Sequential(
                nn.Linear(self.node_gate_input_dim, self.final_node_dim),
                get_activation(activation),
                nn.LayerNorm(self.final_node_dim),
                nn.Linear(self.final_node_dim, self.final_node_dim),
                nn.Sigmoid()
            )

        self.node_classifier = self._build_node_classifier()

        self._param_printed = False
        self.apply(self._init_weights)

    def _build_node_classifier(self):
        layers = []
        input_dim = self.node_classifier_input_dim
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
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0 if module.out_features == 1 else 0.01)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, data: Batch) -> dict:
        if self.training and not self._param_printed:
            print(f"ReCEP total params: {sum(p.numel() for p in self.parameters()):,}")
            self._param_printed = True

        x = data.x
    
        coords = data.pos
        batch = data.batch
        e_attr = data.edge_attr
        coords_C = coords[:, 1].clone()

        x = self.proj_layer(x)
        if self.dihedral and coords is not None:
            x = x + self.dihedral_features(coords)
        if self.rsa and data.rsa is not None:
            rsa = data.rsa.unsqueeze(-1)
            x = torch.cat([x, rsa], dim=-1)

        h = x
        assert h.shape[1] == self.node_dims[0], f"[ReCEP] Node feature dim mismatch: got {h.shape[1]}, expected {self.node_dims[0]}"

        if self.use_egnn:
            for layer in self.egnn_layers:
                h, coords_C, _ = layer(h, coords_C, data.edge_index, batch, edge_attr=e_attr)

        if self.concat and self.use_egnn:
            h = torch.cat([x, h], dim=-1)
        elif self.addition and self.use_egnn:
            h = h + x

        graph_feats = self.graph_pool(h, batch)
        global_pred = self.global_predictor(graph_feats).squeeze(-1)

        context = graph_feats[batch]
        if self.node_gate and hasattr(self, 'node_gate'):
            if self.fusion_type == 'concat':
                gate_input = torch.cat([h, context], dim=-1)
            elif self.fusion_type == 'add':
                gate_input = h + context
            else:
                raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
            gate = self.node_gate(gate_input)
            gated_h = h + gate * h
        else:
            gated_h = h

        if self.fusion_type == 'concat':
            cat = torch.cat([gated_h, context], dim=-1)
        elif self.fusion_type == 'add':
            # Ensure dimensions match for addition
            assert gated_h.shape[-1] == context.shape[-1], f"[ReCEP] Dimension mismatch for add fusion: gated_h {gated_h.shape[-1]} vs context {context.shape[-1]}"
            cat = gated_h + context
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        # Verify input dimension matches node classifier expectation
        expected_dim = self.node_classifier_input_dim
        actual_dim = cat.shape[-1]
        assert actual_dim == expected_dim, f"[ReCEP] Node classifier input dim mismatch: got {actual_dim}, expected {expected_dim}"
        
        node_preds = self.node_classifier(cat).squeeze(-1)

        return {"global_pred": global_pred, "node_preds": node_preds}
    
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

        print("\nReCEP Model Parameter Summary:")
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
            print(f"ReCEP model saved to {save_path}")
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
            'addition': self.addition,
            'pooling': self.pooling,
            'fusion_type': self.fusion_type,
            'node_gate': self.node_gate,
            'node_norm': self.node_norm,
            'node_layers': self.node_layers,
            'out_dropout': self.out_dropout,
            'use_egnn': self.use_egnn,
            'encoder': self.encoder
        }
        
        
model_registry = {
    "ReCEP": ReCEP,
    "EP": EP,
}

def get_model(configs):
    """
    Flexible model loader. Accepts either an argparse.Namespace or a dict.
    Returns an instance of the selected model.
    """
    # Support both argparse.Namespace and dict
    if hasattr(configs, '__dict__'):
        args = vars(configs)
    else:
        args = configs
    
    # Default to ReCEP if no model specified
    model_name = args.get('model', 'ReCEP')
    
    if model_name not in model_registry:
        valid_models = list(model_registry.keys())
        raise ValueError(f"Invalid model type: {model_name}. Must be one of: {valid_models}")
    
    model_class = model_registry[model_name]
    
    # Use inspect to get the model's __init__ parameters
    import inspect
    init_signature = inspect.signature(model_class.__init__)
    parameters = init_signature.parameters
    
    # Build model configuration from args
    model_config = {}
    for param_name, param in parameters.items():
        if param_name == 'self':
            continue
        if param_name in args:
            model_config[param_name] = args[param_name]
        elif param.default is not param.empty:
            model_config[param_name] = param.default
        else:
            print(f"[WARNING] Required parameter '{param_name}' not found in args and has no default value")
    
    # print(f"[INFO] Creating {model_name} model with config: {list(model_config.keys())}")
    model = model_class(**model_config)
    return model