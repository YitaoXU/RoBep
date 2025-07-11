"""
Simplified graph utilities for SphereGraphDataset.
Contains only the essential functions needed without external dependencies.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

def parse_binding_site_txt(txt_path: Path) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    """
    Parse a fasta-like txt file with 3 lines per entry: >id, sequence, label
    Returns: (rna_ids, rna_seqs, labels)
    """
    rna_ids, rna_seqs, labels = [], [], []
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    for i in range(0, len(lines), 3):
        rna_id = lines[i][1:] if lines[i].startswith('>') else lines[i]
        seq = lines[i+1]
        label_str = lines[i+2]
        label = torch.tensor([int(x) for x in label_str], dtype=torch.float32)
        rna_ids.append(rna_id)
        rna_seqs.append(seq)
        labels.append(label)
    return rna_ids, rna_seqs, labels

def create_graph_data_full(
    embeddings: np.ndarray,
    backbone_atoms: np.ndarray,
    rsa_values: np.ndarray,
    epitope_indices: List[int],
    pdb_id: str,
    chain_id: str,
    num_rbf: int = 16,
    num_posenc: int = 16,
    radius: float = 18.0,
    verbose: bool = True
    ) -> Optional[Data]:
    """
    Create a PyTorch Geometric Data object for a full protein graph.
    
    Args:
        embeddings: Full protein embeddings [seq_len, embed_dim]
        backbone_atoms: Full protein backbone atoms [seq_len, 3, 3] (N, CA, C)
        rsa_values: Full protein RSA values [seq_len]
        epitope_indices: List of epitope residue indices
        pdb_id: PDB ID
        chain_id: Chain ID
        num_rbf: Number of RBF features
        num_posenc: Number of positional encoding features
        radius: Distance threshold for edge creation (default: 18.0 Å)
        verbose: Whether to print debug information
        
    Returns:
        PyTorch Geometric Data object or None if creation fails
    """
    try:
        # Validate input dimensions
        seq_len = len(embeddings)
        if len(backbone_atoms) != seq_len or len(rsa_values) != seq_len:
            if verbose:
                print(f"[WARNING] Dimension mismatch for {pdb_id}_{chain_id}: "
                      f"embeddings={len(embeddings)}, backbone={len(backbone_atoms)}, "
                      f"rsa={len(rsa_values)}")
            return None
        
        if seq_len == 0:
            if verbose:
                print(f"[WARNING] Empty protein {pdb_id}_{chain_id}")
            return None
        
        # Create node labels (binary epitope classification)
        node_labels = np.zeros(seq_len, dtype=np.float32)
        if epitope_indices:
            # Filter epitope_indices to ensure they are within bounds
            valid_epitope_indices = [idx for idx in epitope_indices if 0 <= idx < seq_len]
            if valid_epitope_indices:
                node_labels[valid_epitope_indices] = 1.0
            
            if verbose and len(valid_epitope_indices) != len(epitope_indices):
                print(f"[WARNING] Some epitope indices out of bounds for {pdb_id}_{chain_id}: "
                      f"filtered {len(epitope_indices)} -> {len(valid_epitope_indices)}")
        
        # Extract CA coordinates for distance calculation
        ca_coords = backbone_atoms[:, 1, :]  # CA is the second atom [seq_len, 3]
        
        # Validate CA coordinates
        if ca_coords.shape[0] == 0:
            if verbose:
                print(f"[WARNING] Empty CA coordinates for {pdb_id}_{chain_id}")
            return None
        
        # Check for NaN or infinite values
        if np.any(np.isnan(ca_coords)) or np.any(np.isinf(ca_coords)):
            if verbose:
                print(f"[WARNING] Invalid CA coordinates (NaN/Inf) for {pdb_id}_{chain_id}")
            return None
        
        # Create edges based on distance threshold using radius_graph
        ca_coords_tensor = torch.tensor(ca_coords, dtype=torch.float32)
        
        # Additional safety check for tensor
        if ca_coords_tensor.numel() == 0:
            if verbose:
                print(f"[WARNING] Empty CA coordinates tensor for {pdb_id}_{chain_id}")
            return None
        
        edge_index = radius_graph(ca_coords_tensor, r=radius, loop=False, max_num_neighbors=32)
        
        if edge_index.shape[1] == 0:
            if verbose:
                print(f"[WARNING] No edges found for {pdb_id}_{chain_id} with radius {radius}")
            # Create a minimal graph with self-loops to avoid empty graph
            edge_index = torch.stack([torch.arange(seq_len), torch.arange(seq_len)], dim=0)
        
        # Compute edge features
        edge_features = compute_edge_features(ca_coords, edge_index, num_rbf=num_rbf, num_posenc=num_posenc)
        
        # Convert to tensors
        x = torch.tensor(embeddings, dtype=torch.float32)  # [seq_len, embed_dim]
        pos = torch.tensor(backbone_atoms, dtype=torch.float32)  # [seq_len, 3, 3]
        rsa = torch.tensor(rsa_values, dtype=torch.float32)  # [seq_len]
        
        # Node-level labels
        y_node = torch.tensor(node_labels, dtype=torch.float32)  # [seq_len]
        
        # Additional protein-level statistics
        num_epitopes = int(node_labels.sum())
        epitope_ratio = num_epitopes / seq_len if seq_len > 0 else 0.0
        
        # Create Data object
        data = Data(
            x=x,  # Node embeddings [seq_len, embed_dim]
            pos=pos,  # Backbone coordinates [seq_len, 3, 3]
            rsa=rsa,  # RSA values [seq_len]
            edge_index=edge_index,  # Edge connectivity [2, n_edges]
            edge_attr=edge_features,  # Edge features [n_edges, edge_dim]
            y_node=y_node,  # Node-level labels [seq_len]
            epitope_indices=epitope_indices,  # Original epitope indices
            pdb_id=pdb_id,  # PDB ID
            chain_id=chain_id,  # Chain ID
            num_nodes=seq_len,  # Number of nodes (residues)
            num_epitopes=num_epitopes,  # Number of epitope residues
            epitope_ratio=epitope_ratio,  # Ratio of epitope residues
            radius=radius  # Distance threshold used for edges
        )
        
        if verbose:
            print(f"[INFO] Created full protein graph for {pdb_id}_{chain_id}: "
                  f"{seq_len} nodes, {edge_index.shape[1]} edges, {num_epitopes} epitopes")
        
        return data
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to create full protein graph for {pdb_id}_{chain_id}: {str(e)}")
        return None

def create_graph_data(
    center_idx: int,
    covered_indices: List[int],
    covered_epitope_indices: List[int],
    embeddings: np.ndarray,
    backbone_atoms: np.ndarray,
    rsa_values: np.ndarray,
    epitope_indices: List[int],
    recall: float,
    precision: float,
    pdb_id: str,
    chain_id: str,
    embeddings2: np.ndarray = None,
    num_rbf: int = 16,
    num_posenc: int = 16,
    verbose: bool = True
    ) -> Optional[Data]:
    """
    Create a PyTorch Geometric Data object for a spherical region.
        
    Args:
        center_idx: Index of the center residue
        covered_indices: List of residue indices in the region
        covered_epitope_indices: List of epitope residue indices in the region
        embeddings: Full protein embeddings
        backbone_atoms: Full protein backbone atoms [seq_len, 3, 3]
        rsa_values: Full protein RSA values
        epitope_indices: List of all epitope indices in the protein (if available)
        recall: Region recall value (if available)
        precision: Region precision value (if available)
        pdb_id: PDB ID
        chain_id: Chain ID
            
        Returns:
            PyTorch Geometric Data object or None if creation fails
        """
    try:
        # Validate indices first
        if not covered_indices:
            if verbose:
                print(f"[WARNING] Empty covered_indices for center {center_idx}")
            return None
        
        # Check if indices are within bounds
        max_idx = max(covered_indices)
        if max_idx >= len(embeddings) or max_idx >= len(backbone_atoms) or max_idx >= len(rsa_values):
            if verbose:
                print(f"[WARNING] Index out of bounds: max_idx={max_idx}, "
                      f"embeddings_len={len(embeddings)}, backbone_len={len(backbone_atoms)}, "
                      f"rsa_len={len(rsa_values)}")
            return None
        
        # Extract node features for covered residues
        node_embeddings = embeddings[covered_indices]  # [n_nodes, embed_dim]
        node_backbone = backbone_atoms[covered_indices]  # [n_nodes, 3, 3]  
        node_rsa = rsa_values[covered_indices]  # [n_nodes]
        
        if embeddings2 is not None:
            node_embeddings2 = embeddings2[covered_indices]  # [n_nodes, embed_dim]
        else:
            node_embeddings2 = None
            
        # Create node labels (binary epitope classification)
        node_labels = np.zeros(len(covered_indices), dtype=np.float32)
        # Use the epitope_indices from the loaded data if available
        epitope_mask = np.isin(covered_indices, epitope_indices)
        node_labels[epitope_mask] = 1.0
            
        # Create fully connected edge index (no self-loops)
        n_nodes = len(covered_indices)
        edge_index = get_edges(n_nodes)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # Compute edge features using CA coordinates
        ca_coords = node_backbone[:, 1, :]  # Extract CA coordinates [n_nodes, 3]
        edge_features = compute_edge_features(ca_coords, edge_index, num_rbf=num_rbf, num_posenc=num_posenc)
        
        # Convert to tensors
        x = torch.tensor(node_embeddings, dtype=torch.float32)
        pos = torch.tensor(node_backbone, dtype=torch.float32)  # [n_nodes, 3, 3]
        rsa = torch.tensor(node_rsa, dtype=torch.float32)
        
        # Graph-level label (recall)
        y_graph = torch.tensor([recall], dtype=torch.float32)
        
        # Node-level labels
        y_node = torch.tensor(node_labels, dtype=torch.float32)
        
        # Create Data object
        data = Data(
            x=x,  # Node embeddings [n_nodes, embed_dim]
            pos=pos,  # Backbone coordinates [n_nodes, 3, 3]
            rsa=rsa,  # RSA values [n_nodes]
            edge_index=edge_index,  # Edge connectivity [2, n_edges]
            edge_attr=edge_features,  # Edge features [n_edges, edge_dim]
            y=y_graph,  # Graph-level label (recall)
            y_node=y_node,  # Node-level labels [n_nodes]
            center_idx=center_idx,  # Center residue index
            covered_indices=covered_indices,  # All covered residue indices
            precision=precision,  # Region precision
            pdb_id=pdb_id,  # PDB ID
            chain_id=chain_id,  # Chain ID
            num_nodes=n_nodes,  # Number of nodes
            embeddings2=node_embeddings2,  # other embeddings [n_nodes, embed_dim] - region-specific
        )
        
        return data
        
    except Exception as e:
        if verbose:
            print(f"Error creating graph data for {pdb_id}_{chain_id} center {center_idx}: {str(e)}")
        return None
    
def compute_edge_features(coords: np.ndarray, edge_index: torch.Tensor, num_rbf: int = 16, num_posenc: int = 16) -> torch.Tensor:
    """
    Compute edge features including RBF and positional encoding.
        
    Args:
        coords: Node coordinates [n_nodes, 3]
        edge_index: Edge connectivity [2, n_edges]
        num_rbf: Number of RBF features
        num_posenc: Number of positional encoding features
        
    Returns:
        Edge features [n_edges, edge_dim]
        """
    # Convert to torch tensors
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
        
    # Compute edge vectors and distances
    edge_vectors = coords_tensor[edge_index[0]] - coords_tensor[edge_index[1]]  # [n_edges, 3]
    edge_distances = torch.norm(edge_vectors, dim=-1)  # [n_edges]
        
    # RBF features
    edge_rbf = rbf(edge_distances, D_count=num_rbf)  # [n_edges, num_rbf]
        
    # Positional encoding
    edge_posenc = get_posenc(edge_index, num_posenc=num_posenc)  # [n_edges, num_posenc]
        
    # Concatenate edge features
    edge_features = torch.cat([edge_rbf, edge_posenc], dim=-1)  # [n_edges, num_rbf + num_posenc]
        
    return edge_features


def get_edges(n_nodes):
    """Generate fully connected edge indices (no self-loops)"""
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return [rows, cols]


def get_posenc(edge_index, num_posenc=16):
    """
    Generate positional encoding for edges.
    From https://github.com/jingraham/neurips19-graph-protein-design
    """
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_posenc, 2, dtype=torch.float32, device=d.device)
        * -(np.log(10000.0) / num_posenc)
    )

    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def rbf(D, D_min=0., D_max=20., D_count=16):
    """
    Radial Basis Function (RBF) encoding for distances.
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF 