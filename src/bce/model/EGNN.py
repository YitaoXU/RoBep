import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import InstanceNorm

class EGNNLayer(nn.Module):
    """
    EGNN layer with optional feed forward network and batch normalization.
    
    Args:
        input_nf: Number of input node features
        output_nf: Number of output node features
        hidden_nf: Number of hidden features
        edges_in_d: Number of input edge features
        act_fn: Activation function
        residual: Whether to use residual connections
        attention: Whether to use attention mechanism for edge features
        normalize: Whether to normalize coordinates
        coords_agg: Aggregation method for coordinates (mean, sum, max, min)
        tanh: Whether to use tanh activation for coordinate updates
        dropout: Dropout rate
        ffn: Whether to use feed forward network
        batch_norm: Whether to use batch normalization
    """
    def __init__(self, input_nf, output_nf, hidden_nf,
                 edges_in_d=0, act_fn=nn.SiLU(),
                 residual=True, attention=False, normalize=False,
                 coords_agg='mean', tanh=False, dropout=0.0,
                 ffn=False, batch_norm=True):
        super().__init__()
        self.input_nf = input_nf
        self.output_nf = output_nf
        self.hidden_nf = hidden_nf
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.dropout = dropout
        self.ffn = ffn
        self.batch_norm = batch_norm

        # Edge MLP
        in_edge = input_nf*2 + 1 + edges_in_d
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge, hidden_nf),
            act_fn, nn.Dropout(dropout),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn, nn.Dropout(dropout),
        )
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf,1), nn.Sigmoid())

        # Coord MLP
        layer = nn.Linear(hidden_nf,1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_blocks = [nn.Linear(hidden_nf, hidden_nf), act_fn,
                        nn.Dropout(dropout), layer]
        if tanh: coord_blocks.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_blocks)

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn, nn.Dropout(dropout),
            nn.Linear(hidden_nf, output_nf),
        )

        # per-graph normalization
        if batch_norm:
            self.norm_node = InstanceNorm(output_nf, affine=True)
            self.norm_coord = InstanceNorm(3, affine=True)

        # FFN
        if ffn:
            self.ff1 = nn.Linear(output_nf, output_nf*2)
            self.ff2 = nn.Linear(output_nf*2, output_nf)
            self.act_ff = act_fn
            self.drop_ff = nn.Dropout(dropout)
            if batch_norm:
                self.norm_ff1 = InstanceNorm(output_nf, affine=True)
                self.norm_ff2 = InstanceNorm(output_nf, affine=True)

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        diff = coord[row] - coord[col]
        dist2 = (diff**2).sum(dim=-1, keepdim=True)
        
        # Clamp distance to prevent extreme values
        dist2 = torch.clamp(dist2, min=self.epsilon, max=100.0)
        
        if self.normalize:
            norm = (dist2.sqrt().detach() + self.epsilon)
            diff = diff / norm
            # Check for NaN/Inf in normalized diff
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
        return dist2, diff

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.drop_ff(self.act_ff(self.ff1(x)))
        return self.ff2(x)
    
    def forward(self, h, coord, edge_index, batch, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # -- edge features --
        e_in = [h[row], h[col], radial]
        if edge_attr is not None: e_in.append(edge_attr)
        e = torch.cat(e_in, dim=-1)
        e = self.edge_mlp(e)
        if self.attention:
            att = self.att_mlp(e)
            e = e * att

        # -- coordinate update --
        coord_update = self.coord_mlp(e)  # [E,1]
        # Clamp coordinate updates to prevent explosion
        coord_update = torch.clamp(coord_update, -1.0, 1.0)
        trans = coord_diff * coord_update  # [E,3]
        
        # Check for NaN/Inf in coordinate updates
        trans = torch.where(torch.isfinite(trans), trans, torch.zeros_like(trans))
        
        agg_coord = scatter(trans, row, dim=0,
                            dim_size=coord.size(0),
                            reduce=self.coords_agg)
        coord = coord + agg_coord
        
        # Check for NaN/Inf in final coordinates
        coord = torch.where(torch.isfinite(coord), coord, torch.zeros_like(coord))
        
        if self.batch_norm:
            coord = self.norm_coord(coord, batch)

        # -- node update --
        agg_node = scatter(e, row, dim=0,
                           dim_size=h.size(0), reduce='sum')
        x_in = torch.cat([h, agg_node], dim=-1)
        if node_attr is not None:
            x_in = torch.cat([x_in, node_attr], dim=-1)
        h_new = self.node_mlp(x_in)
        if self.batch_norm:
            h_new = self.norm_node(h_new, batch)
        if self.residual and h_new.shape[-1] == h.shape[-1]:
            h_new = h + h_new

        # -- optional FFN --
        if self.ffn:
            if self.batch_norm:
                h_new = self.norm_ff1(h_new, batch)
            h_new = h_new + self._ff_block(h_new)
            if self.batch_norm:
                h_new = self.norm_ff2(h_new, batch)

        return h_new, coord, e

class EGNNLayer2(nn.Module):
    """
    EGNN layer with optional feed forward network and batch normalization.
    
    Args:
        input_nf: Number of input node features
        output_nf: Number of output node features
        hidden_nf: Number of hidden features
        edges_in_d: Number of input edge features
        act_fn: Activation function
        residual: Whether to use residual connections
        attention: Whether to use attention mechanism for edge features
        normalize: Whether to normalize coordinates
        coords_agg: Aggregation method for coordinates (mean, sum, max, min)
        tanh: Whether to use tanh activation for coordinate updates
        dropout: Dropout rate
        ffn: Whether to use feed forward network
        batch_norm: Whether to use batch normalization
    """
    def __init__(self, input_nf, output_nf, hidden_nf,
                 edges_in_d=0, act_fn=nn.SiLU(),
                 residual=True, attention=False, normalize=False,
                 coords_agg='mean', tanh=False, dropout=0.0,
                 ffn=False, batch_norm=True):
        super().__init__()
        self.input_nf = input_nf
        self.output_nf = output_nf
        self.hidden_nf = hidden_nf
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.dropout = dropout
        self.ffn = ffn
        self.batch_norm = batch_norm

        # Edge MLP
        in_edge = input_nf*2 + 1 + edges_in_d
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge, hidden_nf),
            act_fn, nn.Dropout(dropout),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn, nn.Dropout(dropout),
        )
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf,1), nn.Sigmoid())

        # Coord MLP
        layer = nn.Linear(hidden_nf,1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_blocks = [nn.Linear(hidden_nf, hidden_nf), act_fn,
                        nn.Dropout(dropout), layer]
        if tanh: coord_blocks.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_blocks)

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn, nn.Dropout(dropout),
            nn.Linear(hidden_nf, output_nf),
        )

        # per-graph normalization
        if batch_norm:
            self.norm_node = InstanceNorm(output_nf, affine=True)
            self.norm_coord = InstanceNorm(3, affine=True)

        # FFN
        if ffn:
            self.ff1 = nn.Linear(output_nf, output_nf*2)
            self.ff2 = nn.Linear(output_nf*2, output_nf)
            self.act_ff = act_fn
            self.drop_ff = nn.Dropout(dropout)
            if batch_norm:
                self.norm_ff1 = InstanceNorm(output_nf, affine=True)
                self.norm_ff2 = InstanceNorm(output_nf, affine=True)

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        diff = coord[row] - coord[col]
        dist2 = (diff**2).sum(dim=-1, keepdim=True)
        
        # Clamp distance to prevent extreme values
        dist2 = torch.clamp(dist2, min=self.epsilon, max=100.0)
        
        if self.normalize:
            norm = (dist2.sqrt().detach() + self.epsilon)
            diff = diff / norm
            # Check for NaN/Inf in normalized diff
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
        return dist2, diff

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.drop_ff(self.act_ff(self.ff1(x)))
        return self.ff2(x)
    
    def forward(self, h, coord, edge_index, batch, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # -- edge features --
        e_in = [h[row], h[col], radial]
        if edge_attr is not None: e_in.append(edge_attr)
        e = torch.cat(e_in, dim=-1)
        e = self.edge_mlp(e)
        if self.attention:
            att = self.att_mlp(e)
            e = e * att

        # -- coordinate update --
        coord_update = self.coord_mlp(e)  # [E,1]
        # Clamp coordinate updates to prevent explosion
        coord_update = torch.clamp(coord_update, -1.0, 1.0)
        trans = coord_diff * coord_update  # [E,3]
        
        # Check for NaN/Inf in coordinate updates
        trans = torch.where(torch.isfinite(trans), trans, torch.zeros_like(trans))
        
        agg_coord = scatter(trans, row, dim=0,
                            dim_size=coord.size(0),
                            reduce=self.coords_agg)
        coord = coord + agg_coord
        
        # Check for NaN/Inf in final coordinates
        coord = torch.where(torch.isfinite(coord), coord, torch.zeros_like(coord))
        
        if self.batch_norm:
            coord = self.norm_coord(coord, batch)

        # -- node update --
        agg_node = scatter(e, row, dim=0,
                           dim_size=h.size(0), reduce='sum')
        x_in = torch.cat([h, agg_node], dim=-1)
        if node_attr is not None:
            x_in = torch.cat([x_in, node_attr], dim=-1)
        h_new = self.node_mlp(x_in)
        if self.batch_norm:
            h_new = self.norm_node(h_new, batch)
        if self.residual and h_new.shape[-1] == h.shape[-1]:
            h_new = h + h_new

        # -- optional FFN --
        if self.ffn:
            if self.batch_norm:
                h_new = self.norm_ff1(h_new, batch)
            h_new = h_new + self._ff_block(h_new)
            if self.batch_norm:
                h_new = self.norm_ff2(h_new, batch)

        return h_new, coord, e