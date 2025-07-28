import os
import h5py
import torch
import numpy as np
import json
import random
from pathlib import Path
from multiprocessing import Pool

from typing import List, Tuple, Dict, Optional, Union
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle

from ..utils.loading import load_epitopes_csv, load_data_split
from ..utils.constants import BASE_DIR
from .utils import create_graph_data, create_graph_data_full


def apply_undersample(data_list: List, undersample_param: Union[int, float], seed: int = 42, verbose: bool = True):
    """
    Apply undersampling to a data list.
    
    Args:
        data_list: List of data samples
        undersample_param: If int, sample that many samples; if float (0-1), sample that fraction of data
        seed: Random seed for reproducibility
        verbose: Whether to print sampling information
        
    Returns:
        Undersampled data list
    """
    if undersample_param is None:
        return data_list
    
    original_size = len(data_list)
    
    if isinstance(undersample_param, float):
        # Sample a fraction of the data
        if not (0 < undersample_param <= 1.0):
            raise ValueError(f"Float undersample must be between 0 and 1, got {undersample_param}")
        target_size = int(len(data_list) * undersample_param)
    elif isinstance(undersample_param, int):
        # Sample a specific number of samples
        if undersample_param <= 0:
            raise ValueError(f"Int undersample must be positive, got {undersample_param}")
        target_size = min(undersample_param, len(data_list))
    else:
        raise ValueError(f"Undersample must be int, float, or None, got {type(undersample_param)}")
    
    if target_size < len(data_list):
        # Set random seed for reproducibility
        random.seed(seed)
        sampled_data = random.sample(data_list, target_size)
        
        if verbose:
            print(f"Applied undersampling: {original_size} -> {target_size} samples")
        
        return sampled_data
    elif verbose:
        print(f"No undersampling applied: requested {target_size}, available {original_size}")
    
    return data_list

class AntigenDataset(Dataset):
    """
    Dataset for antigen chains.
    Each data point represents a complete protein as a graph, with nodes being residues
    and edges based on spatial distance (< 18 Å).
    """
    def __init__(
        self,
        data_split: str = "train",
        radius: float = 18,
        threshold: float = 0.25,
        num_posenc: int = 16,
        num_rbf: int = 16,
        undersample: Union[int, float, None] = None,
        cache_dir: Optional[str] = None,
        force_rebuild: bool = False,
        verbose: bool = True,
        seed: int = 42,
        encoder: str = "esmc"
    ):
        """
        Initialize the antigen dataset.
        
        Args:
            data_split: Data split name ('train', 'val', 'test')
            radius: Distance threshold for edge creation (Å)
            threshold: SASA threshold for surface residues (not used in full protein)
            num_posenc: Number of positional encoding features
            num_rbf: Number of RBF features
            undersample: Undersample parameter (int for count, float for ratio)
            cache_dir: Directory to cache processed data
            force_rebuild: Whether to force rebuild the dataset
            verbose: Whether to print progress information
            seed: Random seed for reproducibility
            encoder: Encoder type ('esmc' or 'esm2')
        """
        self.data_split = data_split
        self.radius = radius
        self.threshold = threshold
        self.num_posenc = num_posenc
        self.num_rbf = num_rbf
        self.undersample = undersample
        self.verbose = verbose
        self.seed = seed
        self.encoder = encoder
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = Path(f"{BASE_DIR}/data/full_region_cache/antigen_r{radius}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file for this configuration
        self.cache_file = self.cache_dir / f"{data_split}_antigen_dataset.h5"
        
        # Load data splits and epitope information
        self.antigens = load_data_split(data_split, verbose=verbose)
        _, _, self.epitope_dict = load_epitopes_csv()
        
        # Initialize data list
        self.data_list = []
        
        # Load or build dataset
        if self.cache_file.exists() and not force_rebuild:
            if verbose:
                print(f"Loading cached antigen dataset from {self.cache_file}")
            self._load_cache()
        else:
            if verbose:
                print(f"Building antigen dataset for {data_split} split...")
            self._build_dataset()
            self._save_cache()
        
        super().__init__()
    
    def _load_protein_data(self, pdb_id: str, chain_id: str) -> Optional[Dict]:
        """
        Load precomputed protein data from files.
        
        Args:
            pdb_id: PDB ID
            chain_id: Chain ID
            
        Returns:
            Dictionary containing all protein data or None if loading fails
        """
        try:
            protein_key = f"{pdb_id}_{chain_id}"
            
            # Load embeddings
            embedding_file = Path(BASE_DIR) / "data" / "embeddings" / self.encoder / f"{protein_key}.h5"
            if not embedding_file.exists():
                if self.verbose:
                    print(f"Embedding file not found: {embedding_file}")
                return None
            
            with h5py.File(embedding_file, "r") as h5f:
                embeddings = h5f["embedding"][:]
            
            # Load backbone atoms
            coords_file = Path(BASE_DIR) / "data" / "coords" / f"{protein_key}.npy"
            if not coords_file.exists():
                if self.verbose:
                    print(f"Coords file not found: {coords_file}")
                return None
            backbone_atoms = np.load(coords_file)
            
            # Load RSA values
            rsa_file = Path(BASE_DIR) / "data" / "rsa" / f"{protein_key}.npy"
            if not rsa_file.exists():
                if self.verbose:
                    print(f"RSA file not found: {rsa_file}")
                return None
            rsa_values = np.load(rsa_file)
            
            # Load epitope data from epitope_dict
            binary_labels = self.epitope_dict.get(protein_key, [])
            
            # Create epitope indices from binary labels
            epitope_indices = []
            for idx, is_epitope in enumerate(binary_labels):
                if is_epitope == 1:
                    epitope_indices.append(idx)
            
            return {
                'embeddings': embeddings,
                'backbone_atoms': backbone_atoms,
                'rsa_values': rsa_values,
                'epitope_indices': epitope_indices,
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading protein data for {pdb_id}_{chain_id}: {str(e)}")
            return None
    
    def _build_dataset(self):
        """Build the dataset from precomputed data files."""
        failed_proteins = []
        
        for pdb_id, chain_id in tqdm(self.antigens, desc=f"Processing {self.data_split} antigens", 
                                   disable=not self.verbose):
            try:
                # Load precomputed data
                protein_data = self._load_protein_data(pdb_id, chain_id)
                if protein_data is None:
                    failed_proteins.append(f"{pdb_id}_{chain_id}")
                    continue
                
                embeddings = protein_data['embeddings']
                backbone_atoms = protein_data['backbone_atoms']
                rsa_values = protein_data['rsa_values']
                epitope_indices = protein_data['epitope_indices']
                
                # Create graph data for the full protein
                graph_data = create_graph_data_full(
                    embeddings=embeddings,
                    backbone_atoms=backbone_atoms,
                    rsa_values=rsa_values,
                    epitope_indices=epitope_indices,
                    pdb_id=pdb_id,
                    chain_id=chain_id,
                    num_rbf=self.num_rbf,
                    num_posenc=self.num_posenc,
                    radius=self.radius,
                    verbose=self.verbose
                )
                
                if graph_data is not None:
                    self.data_list.append(graph_data)
                else:
                    failed_proteins.append(f"{pdb_id}_{chain_id}")
                        
            except Exception as e:
                failed_proteins.append(f"{pdb_id}_{chain_id}")
                if self.verbose:
                    print(f"Error processing {pdb_id}_{chain_id}: {str(e)}")
        
        if failed_proteins and self.verbose:
            print(f"Failed to process {len(failed_proteins)} proteins: {failed_proteins[:5]}...")
        
        # Apply undersampling if specified
        if self.undersample is not None:
            self.data_list = apply_undersample(
                self.data_list,
                self.undersample,
                seed=self.seed,
                verbose=self.verbose
            )
        
        if self.verbose:
            print(f"Successfully created {len(self.data_list)} protein graphs")
    
    def _save_cache(self):
        """Save processed dataset to cache."""
        try:
            self._save_cache_hdf5()
            if self.verbose:
                print(f"Dataset cached to {self.cache_file}")
        except Exception as e:
            if self.verbose:
                print(f"Failed to save cache: {str(e)}")
    
    def _load_cache(self):
        """Load processed dataset from cache."""
        try:
            self._load_cache_hdf5()
            if self.verbose:
                print(f"Loaded {len(self.data_list)} samples from cache")
        except Exception as e:
            if self.verbose:
                print(f"Failed to load cache: {str(e)}")
            self.data_list = []
    
    def _save_cache_hdf5(self):
        """Save dataset using HDF5 format."""
        with h5py.File(self.cache_file, 'w') as f:
            # Save metadata
            f.attrs['num_samples'] = len(self.data_list)
            f.attrs['radius'] = self.radius
            f.attrs['threshold'] = self.threshold
            f.attrs['data_split'] = self.data_split
            f.attrs['encoder'] = self.encoder
            f.attrs['dataset_type'] = 'antigen_full'
            
            # Save each protein as a separate group
            for i, data in enumerate(tqdm(self.data_list, desc="Saving dataset...", disable=not self.verbose)):
                group = f.create_group(f'protein_{i}')
                
                # Save tensors as datasets with compression
                group.create_dataset('x', data=data.x.numpy(), compression='gzip', compression_opts=6)
                group.create_dataset('pos', data=data.pos.numpy(), compression='gzip', compression_opts=6)
                group.create_dataset('rsa', data=data.rsa.numpy(), compression='gzip', compression_opts=6)
                group.create_dataset('edge_index', data=data.edge_index.numpy(), compression='gzip', compression_opts=6)
                group.create_dataset('edge_attr', data=data.edge_attr.numpy(), compression='gzip', compression_opts=6)
                group.create_dataset('y_node', data=data.y_node.numpy(), compression='gzip', compression_opts=6)
                
                # Save scalar and list attributes
                group.attrs['pdb_id'] = data.pdb_id.encode('utf-8')
                group.attrs['chain_id'] = data.chain_id.encode('utf-8')
                group.attrs['num_nodes'] = data.num_nodes
                group.attrs['num_epitopes'] = data.num_epitopes
                group.attrs['epitope_ratio'] = data.epitope_ratio
                group.attrs['radius'] = data.radius
                
                # Save epitope indices
                group.create_dataset('epitope_indices', data=np.array(data.epitope_indices), compression='gzip', compression_opts=6)
    
    def _load_cache_hdf5(self):
        """Load dataset from HDF5 cache."""
        self.data_list = []
        
        with h5py.File(self.cache_file, 'r') as f:
            total_samples = f.attrs['num_samples']
            
            for i in tqdm(range(total_samples), desc="Loading dataset...", disable=not self.verbose):
                group = f[f'protein_{i}']
                attrs = dict(group.attrs)
                
                # Safe string decoding
                def safe_decode(attr):
                    val = attrs[attr]
                    return val.decode('utf-8') if isinstance(val, bytes) else str(val)
                
                data = Data(
                    x=torch.tensor(group['x'][:]),
                    pos=torch.tensor(group['pos'][:]),
                    rsa=torch.tensor(group['rsa'][:]),
                    edge_index=torch.tensor(group['edge_index'][:]),
                    edge_attr=torch.tensor(group['edge_attr'][:]),
                    y_node=torch.tensor(group['y_node'][:]),
                    epitope_indices=group['epitope_indices'][:].tolist(),
                    pdb_id=safe_decode('pdb_id'),
                    chain_id=safe_decode('chain_id'),
                    num_nodes=int(attrs['num_nodes']),
                    num_epitopes=int(attrs['num_epitopes']),
                    epitope_ratio=float(attrs['epitope_ratio']),
                    radius=float(attrs['radius'])
                )
                self.data_list.append(data)
        
        # Apply undersampling if specified
        if self.undersample is not None:
            self.data_list = apply_undersample(
                self.data_list,
                self.undersample,
                seed=self.seed,
                verbose=self.verbose
            )
    
    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        """Get a sample by index."""
        return self.data_list[idx]
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if not self.data_list:
            return {}
        
        # Collect statistics
        num_nodes_list = [data.num_nodes for data in self.data_list]
        num_edges_list = [data.edge_index.shape[1] for data in self.data_list]
        num_epitopes_list = [data.num_epitopes for data in self.data_list]
        epitope_ratio_list = [data.epitope_ratio for data in self.data_list]
        
        # Overall statistics
        total_nodes = sum(num_nodes_list)
        total_edges = sum(num_edges_list)
        total_epitopes = sum(num_epitopes_list)
        
        stats = {
            'num_proteins': len(self.data_list),
            'avg_nodes_per_protein': np.mean(num_nodes_list),
            'std_nodes_per_protein': np.std(num_nodes_list),
            'min_nodes_per_protein': np.min(num_nodes_list),
            'max_nodes_per_protein': np.max(num_nodes_list),
            'avg_edges_per_protein': np.mean(num_edges_list),
            'std_edges_per_protein': np.std(num_edges_list),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'total_epitopes': total_epitopes,
            'avg_epitopes_per_protein': np.mean(num_epitopes_list),
            'avg_epitope_ratio': np.mean(epitope_ratio_list),
            'overall_epitope_ratio': total_epitopes / total_nodes if total_nodes > 0 else 0,
        }
        
        return stats
    
    def print_stats(self):
        """Print dataset statistics."""
        stats = self.get_stats()
        if not stats:
            print("No statistics available (empty dataset)")
            return
        
        print(f"\n=== {self.data_split.upper()} Antigen Dataset Statistics ===")
        print(f"Number of proteins: {stats['num_proteins']:,}")
        print(f"Average nodes per protein: {stats['avg_nodes_per_protein']:.1f} ± {stats['std_nodes_per_protein']:.1f}")
        print(f"Nodes per protein range: [{stats['min_nodes_per_protein']}, {stats['max_nodes_per_protein']}]")
        print(f"Average edges per protein: {stats['avg_edges_per_protein']:.1f} ± {stats['std_edges_per_protein']:.1f}")
        print(f"Total nodes: {stats['total_nodes']:,}")
        print(f"Total edges: {stats['total_edges']:,}")
        print(f"Total epitope nodes: {stats['total_epitopes']:,}")
        print(f"Average epitopes per protein: {stats['avg_epitopes_per_protein']:.1f}")
        print(f"Average epitope ratio per protein: {stats['avg_epitope_ratio']:.3f}")
        print(f"Overall epitope ratio: {stats['overall_epitope_ratio']:.3f}")
        print("=" * 50)


class SphereGraphDataset(Dataset):
    """
    Optimized graph dataset for training ReGEP model using spherical regions from antigen chains.
    Each graph represents a spherical region centered on a surface residue.
    
    Optimizations:
    - Only uses HDF5 format for caching
    - Builds complete dataset without zero_ratio filtering
    - Applies zero_ratio and undersample during loading
    - Faster caching with optimized HDF5 structure
    """
    
    def __init__(
        self,
        data_split: str = "train",
        radius: int = 18,
        threshold: float = 0.25,
        num_posenc: int = 16,
        num_rbf: int = 16,
        zero_ratio: float = 0.1,
        undersample: Union[int, float, None] = None,
        cache_dir: Optional[str] = None,
        force_rebuild: bool = False,
        verbose: bool = True,
        seed: int = 42,
        use_embeddings2: bool = False
    ):
        """
        Initialize the spherical graph dataset.
        
        Args:
            data_split: Data split name ('train', 'val', 'test')
            radius: Radius for spherical regions
            threshold: SASA threshold for surface residues
            num_posenc: Number of positional encoding features
            num_rbf: Number of RBF features
            zero_ratio: Ratio to downsample graphs with recall=0 (0.3 means keep 30%)
            undersample: Undersample parameter (int for count, float for ratio)
            cache_dir: Directory to cache processed data
            force_rebuild: Whether to force rebuild the dataset
            verbose: Whether to print progress information
            seed: Random seed for reproducibility
        """
        self.data_split = data_split
        self.radius = radius
        self.threshold = threshold
        self.num_posenc = num_posenc
        self.num_rbf = num_rbf
        self.zero_ratio = zero_ratio
        self.undersample = undersample
        self.verbose = verbose
        self.seed = seed
        self.use_embeddings2 = use_embeddings2
        
        # Set cache directory to large disk
        if cache_dir is None:
            cache_dir = Path(f"{BASE_DIR}/data/region_cache/sphere_r{radius}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file for this configuration (only HDF5)
        self.cache_file = self.cache_dir / f"{data_split}_dataset_complete.h5"
        
        # Load data splits
        self.antigens = load_data_split(data_split, verbose=verbose)
        
        # Initialize data list
        self.data_list = []
        
        # Load or build dataset
        if self.cache_file.exists() and not force_rebuild:
            if verbose:
                print(f"Loading cached dataset with radius {self.radius} from {self.cache_file}")
            self._load_cache()
        else:
            if verbose:
                print(f"Building complete dataset with radius {self.radius} for {data_split} split...")
            self._build_dataset()
            self._save_cache()
        
        super().__init__()
    
    def _load_protein_data(self, pdb_id: str, chain_id: str) -> Optional[Dict]:
        """
        Load precomputed protein data from files.
        
        Args:
            pdb_id: PDB ID
            chain_id: Chain ID
            
        Returns:
            Dictionary containing all protein data or None if loading fails
        """
        try:
            protein_key = f"{pdb_id}_{chain_id}"
            
            # Load embeddings
            embedding_file = Path(BASE_DIR) / "data" / "embeddings" / 'esmc' / f"{protein_key}.h5"
            if not embedding_file.exists():
                if self.verbose:
                    print(f"Embedding file not found: {embedding_file}")
                return None
            
            with h5py.File(embedding_file, "r") as h5f:
                embeddings = h5f["embedding"][:]
            
            # Load other embeddings if available
            esm2_file = Path(BASE_DIR) / "data" / "embeddings" / "esm2" / f"{protein_key}.h5"
            if not esm2_file.exists():
                if self.verbose:
                    print(f"ESM2 file not found: {esm2_file}")
                embeddings2 = None
            else:
                with h5py.File(esm2_file, "r") as h5f:
                    embeddings2 = h5f["embedding"][:]
            
            # Load backbone atoms
            coords_file = Path(BASE_DIR) / "data" / "coords" / f"{protein_key}.npy"
            if not coords_file.exists():
                if self.verbose:
                    print(f"Coords file not found: {coords_file}")
                return None
            backbone_atoms = np.load(coords_file)
            
            # Load RSA values
            rsa_file = Path(BASE_DIR) / "data" / "rsa" / f"{protein_key}.npy"
            if not rsa_file.exists():
                if self.verbose:
                    print(f"RSA file not found: {rsa_file}")
                return None
            rsa_values = np.load(rsa_file)
            
            # Load surface coverage data
            sphere_file = Path(BASE_DIR) / "data" / "antigen_sphere" / f"{protein_key}.h5"
            radius_key = f"r{self.radius}"
            
            if not sphere_file.exists():
                if self.verbose:
                    print(f"Sphere file not found: {sphere_file}")
                return None
            
            coverage_dict = {}
            with h5py.File(sphere_file, "r") as h5f:
                if radius_key not in h5f:
                    if self.verbose:
                        print(f"Radius {self.radius} not found in {sphere_file}")
                    return None
                
                radius_group = h5f[radius_key]
                for center_idx_str in radius_group.keys():
                    center_idx = int(center_idx_str)
                    center_group = radius_group[center_idx_str]
                    covered_indices = center_group['covered_indices'][:].tolist()
                    covered_epitope_indices = center_group['covered_epitope_indices'][:].tolist()
                    precision = float(center_group.attrs['precision'])
                    recall = float(center_group.attrs['recall'])
                    coverage_dict[center_idx] = (covered_indices, covered_epitope_indices, precision, recall)
            
            # Load epitope data
            _, _, epitopes = load_epitopes_csv()
            binary_labels = epitopes.get(protein_key, [])
            
            # Create epitope indices
            epitope_indices = []
            for idx, is_epitope in enumerate(binary_labels):
                if is_epitope == 1:
                    epitope_indices.append(idx)
            
            return {
                'embeddings': embeddings,
                'backbone_atoms': backbone_atoms,
                'rsa_values': rsa_values,
                'coverage_dict': coverage_dict,
                'epitope_indices': epitope_indices,
                'embeddings2': embeddings2
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading protein data for {pdb_id}_{chain_id}: {str(e)}")
            return None

    def _build_dataset(self):
        """Build the complete dataset from precomputed data files (no zero_ratio filtering)."""
        failed_proteins = []
        
        for pdb_id, chain_id in tqdm(self.antigens, desc=f"Processing {self.data_split} antigens", 
                                   disable=not self.verbose):
            try:
                # Load precomputed data directly
                protein_data = self._load_protein_data(pdb_id, chain_id)
                if protein_data is None:
                    if self.verbose:
                        print(f"Failed to load data for {pdb_id}_{chain_id}")
                    continue
                
                embeddings = protein_data['embeddings']
                embeddings2 = protein_data['embeddings2']
                backbone_atoms = protein_data['backbone_atoms']
                rsa_values = protein_data['rsa_values']
                coverage_dict = protein_data['coverage_dict']
                epitope_indices = protein_data['epitope_indices']
                
                if not coverage_dict:
                    if self.verbose:
                        print(f"No surface regions found for {pdb_id}_{chain_id}")
                    continue
                
                # Process each spherical region (no zero_ratio filtering here)
                for center_idx, (covered_indices, covered_epitope_indices, precision, recall) in coverage_dict.items():
                    if len(covered_indices) < 2:  # Skip regions with too few residues
                        continue
                    
                    # Create graph data for this region (include all data)
                    graph_data = create_graph_data(
                        center_idx=center_idx,
                        covered_indices=covered_indices,
                        covered_epitope_indices=covered_epitope_indices,
                        embeddings=embeddings,
                        embeddings2=embeddings2,
                        backbone_atoms=backbone_atoms,
                        rsa_values=rsa_values,
                        epitope_indices=epitope_indices,
                        recall=recall,
                        precision=precision,
                        pdb_id=pdb_id,
                        chain_id=chain_id,
                        num_rbf=self.num_rbf,
                        num_posenc=self.num_posenc,
                        verbose=self.verbose
                    )
                    
                    if graph_data is not None:
                        self.data_list.append(graph_data)
                        
            except Exception as e:
                failed_proteins.append(f"{pdb_id}_{chain_id}")
                if self.verbose:
                    print(f"Error processing {pdb_id}_{chain_id}: {str(e)}")
        
        if failed_proteins and self.verbose:
            print(f"Failed to process {len(failed_proteins)} proteins: {failed_proteins[:5]}...")
        
        if self.verbose:
            print(f"Successfully created {len(self.data_list)} graph samples (complete dataset)")
    
    def _save_cache(self):
        """Save processed dataset to cache."""
        try:
            self._save_cache_hdf5()
            if self.verbose:
                print(f"Dataset cached to {self.cache_file}")
        except Exception as e:
            if self.verbose:
                print(f"Failed to save cache: {str(e)}")
    
    def _load_cache(self):
        """Load processed dataset from cache."""
        try:
            self._load_cache_hdf5()
            if self.verbose:
                print(f"Loaded {len(self.data_list)} samples from cache")
        except Exception as e:
            if self.verbose:
                print(f"Failed to load cache: {str(e)}")
            self.data_list = []
    
    def _save_cache_hdf5(self):
        """Save dataset using optimized HDF5 format for faster loading."""
        with h5py.File(self.cache_file, 'w') as f:
            # Save metadata
            f.attrs['num_samples'] = len(self.data_list)
            f.attrs['radius'] = self.radius
            f.attrs['threshold'] = self.threshold
            f.attrs['data_split'] = self.data_split
            f.attrs['complete_dataset'] = True  # Mark as complete dataset
            
            # Pre-allocate arrays for better performance
            num_samples = len(self.data_list)
            if num_samples == 0:
                return
            
            # Collect all data first to determine max dimensions
            all_x = []
            all_pos = []
            all_rsa = []
            all_edge_index = []
            all_edge_attr = []
            all_y = []
            all_y_node = []
            all_center_idx = []
            all_precision = []
            all_pdb_ids = []
            all_chain_ids = []
            all_num_nodes = []
            all_covered_indices = []
            all_embeddings2 = []
            
            max_nodes = 0
            max_edges = 0
            
            for data in self.data_list:
                all_x.append(data.x.numpy())
                all_pos.append(data.pos.numpy())
                all_rsa.append(data.rsa.numpy())
                all_edge_index.append(data.edge_index.numpy())
                all_edge_attr.append(data.edge_attr.numpy())
                all_y.append(data.y.numpy())
                all_y_node.append(data.y_node.numpy())
                all_center_idx.append(data.center_idx)
                all_precision.append(data.precision)
                all_pdb_ids.append(data.pdb_id.encode('utf-8'))
                all_chain_ids.append(data.chain_id.encode('utf-8'))
                all_num_nodes.append(data.num_nodes)
                all_covered_indices.append(data.covered_indices)
                
                # Handle embeddings2 safely - it could be None or numpy array
                if hasattr(data, 'embeddings2') and data.embeddings2 is not None:
                    if isinstance(data.embeddings2, np.ndarray):
                        all_embeddings2.append(data.embeddings2)
                    else:
                        # It's a torch tensor
                        all_embeddings2.append(data.embeddings2.numpy())
                else:
                    # No embeddings2 available, use zeros as placeholder
                    all_embeddings2.append(np.zeros((data.num_nodes, 1280), dtype=np.float32))  # ESM2 dim
                
                max_nodes = max(max_nodes, data.num_nodes)
                max_edges = max(max_edges, data.edge_index.shape[1])
            
            # Save each graph as a separate group with compression
            progress_bar = tqdm(enumerate(self.data_list), total=num_samples, desc="Saving dataset...", disable=not self.verbose)
            
            for i, data in progress_bar:
                group = f.create_group(f'graph_{i}')
                
                # Save tensors as datasets with compression
                group.create_dataset('x', data=all_x[i], compression='gzip', compression_opts=6)
                group.create_dataset('pos', data=all_pos[i], compression='gzip', compression_opts=6)
                group.create_dataset('rsa', data=all_rsa[i], compression='gzip', compression_opts=6)
                group.create_dataset('edge_index', data=all_edge_index[i], compression='gzip', compression_opts=6)
                group.create_dataset('edge_attr', data=all_edge_attr[i], compression='gzip', compression_opts=6)
                group.create_dataset('y', data=all_y[i], compression='gzip', compression_opts=6)
                group.create_dataset('y_node', data=all_y_node[i], compression='gzip', compression_opts=6)
                group.create_dataset('embeddings2', data=all_embeddings2[i], compression='gzip', compression_opts=6)
                
                # Save scalar attributes
                group.attrs['center_idx'] = all_center_idx[i]
                group.attrs['precision'] = all_precision[i]
                group.attrs['pdb_id'] = all_pdb_ids[i]
                group.attrs['chain_id'] = all_chain_ids[i]
                group.attrs['num_nodes'] = all_num_nodes[i]
                
                # Save list attributes as datasets with compression
                group.create_dataset('covered_indices', data=np.array(all_covered_indices[i]), compression='gzip', compression_opts=6)
                
    def _load_cache_hdf5(self):
        """Optimized cache loader with robust string handling."""
        self.data_list = []
        
        with h5py.File(self.cache_file, 'r') as f:
            # PHASE 1: Rapid metadata scan
            zero_recall_indices = []
            non_zero_recall_indices = []
            total_samples = f.attrs['num_samples']
            
            if self.verbose:
                print(f"Scanning {total_samples} samples for recall values...")
            
            for i in range(total_samples):
                recall = f[f'graph_{i}/y'][0].item()
                if recall == 0.0:
                    zero_recall_indices.append(i)
                else:
                    non_zero_recall_indices.append(i)
            
            # PHASE 2: Apply zero_ratio filtering
            selected_indices = non_zero_recall_indices.copy()
            
            if isinstance(self.zero_ratio, (int, float)) and 0 <= self.zero_ratio <= 1:
                if self.zero_ratio < 1.0 and zero_recall_indices:
                    random.seed(self.seed)
                    target_count = int(len(zero_recall_indices) * self.zero_ratio)
                    selected_zero_indices = random.sample(zero_recall_indices, target_count)
                    selected_indices.extend(selected_zero_indices)
                    
                    if self.verbose:
                        kept = len(selected_zero_indices)
                        total = len(zero_recall_indices)
                        print(f"Zero-recall filtering: kept {kept}/{total} samples (ratio={self.zero_ratio})")
                else:
                    selected_indices.extend(zero_recall_indices)
            
            # PHASE 3: Selective data loading with safe string handling
            if self.verbose:
                print(f"Loading {len(selected_indices)} selected samples...")
            
            for idx in tqdm(selected_indices, disable=not self.verbose):
                group = f[f'graph_{idx}']
                attrs = dict(group.attrs)
                
                # Safe string decoding
                def safe_decode(attr):
                    val = attrs[attr]
                    return val.decode('utf-8') if isinstance(val, bytes) else str(val)
                
                # Load embeddings2 if available and use_embeddings2 is True
                if 'embeddings2' in group and self.use_embeddings2:
                    if group['embeddings2'] is not None:
                        emb = torch.tensor(group['embeddings2'][:])
                    else:
                        emb = torch.tensor(group['x'][:])
                else:
                    emb = torch.tensor(group['x'][:])
                
                data = Data(
                    x=emb,
                    pos=torch.tensor(group['pos'][:]),
                    rsa=torch.tensor(group['rsa'][:]),
                    edge_index=torch.tensor(group['edge_index'][:]),
                    edge_attr=torch.tensor(group['edge_attr'][:]),
                    y=torch.tensor(group['y'][:]),
                    y_node=torch.tensor(group['y_node'][:]),
                    center_idx=int(attrs['center_idx']),
                    covered_indices=group['covered_indices'][:].tolist(),
                    precision=float(attrs['precision']),
                    pdb_id=safe_decode('pdb_id'),
                    chain_id=safe_decode('chain_id'),
                    num_nodes=int(attrs['num_nodes'])
                )
                
                self.data_list.append(data)
        
        # PHASE 4: Apply undersampling
        if self.undersample is not None:
            self.data_list = apply_undersample(
                self.data_list,
                self.undersample,
                seed=self.seed,
                verbose=self.verbose
            )
        
        if self.verbose:
            print(f"Loaded {len(self.data_list)} samples (optimized loader)")
    
    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        """Get a sample by index."""
        return self.data_list[idx]
    
    def apply_filters(self, zero_ratio: Optional[float] = None, undersample: Union[int, float, None] = None, seed: int = None):
        """
        Apply filtering to the already loaded dataset (for compatibility).
        Note: It's more efficient to set these parameters during initialization.
        
        Args:
            zero_ratio: Ratio to downsample graphs with recall=0
            undersample: Undersample parameter
            seed: Random seed for reproducibility
        """
        if seed is None:
            seed = self.seed
            
        # Update instance parameters and re-filter
        if zero_ratio is not None:
            self.zero_ratio = zero_ratio
        if undersample is not None:
            self.undersample = undersample
        if seed is not None:
            self.seed = seed
            
        # Reload from cache with new parameters
        if self.cache_file.exists():
            if self.verbose:
                print("Re-applying filters to cached dataset...")
            self._load_cache_hdf5()
        else:
            if self.verbose:
                print("Warning: No cache file found, filters cannot be applied")
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if not self.data_list:
            return {}
        
        # Collect statistics
        num_nodes_list = [data.num_nodes for data in self.data_list]
        recall_list = [data.y.item() for data in self.data_list]
        precision_list = [data.precision for data in self.data_list]
        
        # Node-level statistics
        total_nodes = sum(num_nodes_list)
        total_epitopes = sum([data.y_node.sum().item() for data in self.data_list])
        num_zero_recall = sum([1 for data in self.data_list if data.y.item() == 0])
        
        stats = {
            'num_graphs': len(self.data_list),
            'avg_nodes_per_graph': np.mean(num_nodes_list),
            'std_nodes_per_graph': np.std(num_nodes_list),
            'min_nodes_per_graph': np.min(num_nodes_list),
            'max_nodes_per_graph': np.max(num_nodes_list),
            'total_nodes': total_nodes,
            'total_epitopes': total_epitopes,
            'epitope_ratio': total_epitopes / total_nodes if total_nodes > 0 else 0,
            'avg_recall': np.mean(recall_list),
            'std_recall': np.std(recall_list),
            'avg_precision': np.mean(precision_list),
            'std_precision': np.std(precision_list),
            'num_zero_recall': num_zero_recall,
        }
        
        return stats
    
    def print_stats(self):
        """Print dataset statistics."""
        stats = self.get_stats()
        if not stats:
            print("No statistics available (empty dataset)")
            return
        
        print(f"\n=== {self.data_split.upper()} Dataset Statistics ===")
        print(f"Number of graphs: {stats['num_graphs']:,}")
        print(f"Average nodes per graph: {stats['avg_nodes_per_graph']:.1f} ± {stats['std_nodes_per_graph']:.1f}")
        print(f"Nodes per graph range: [{stats['min_nodes_per_graph']}, {stats['max_nodes_per_graph']}]")
        print(f"Total nodes: {stats['total_nodes']:,}")
        print(f"Total epitope nodes: {stats['total_epitopes']:,}")
        print(f"Epitope ratio: {stats['epitope_ratio']:.3f}")
        print(f"Average recall: {stats['avg_recall']:.3f} ± {stats['std_recall']:.3f}")
        print(f"Average precision: {stats['avg_precision']:.3f} ± {stats['std_precision']:.3f}")
        print(f"Number of graphs with zero recall: {stats['num_zero_recall']:,}")
        print("=" * 40)


class MultiRadiusGraphDataset(Dataset):
    """
    Dataset that combines multiple radius datasets for multi-scale training.
    """
    
    def __init__(
        self,
        data_split: str = "train",
        radii: List[int] = [16, 18, 20],
        threshold: float = 0.25,
        num_posenc: int = 16,
        num_rbf: int = 16,
        zero_ratio: float = 0.1,
        undersample: Union[int, float, None] = None,
        cache_dir: Optional[str] = None,
        force_rebuild: bool = False,
        verbose: bool = True,
        use_embeddings2: bool = False
    ):
        """
        Initialize multi-radius dataset.
        
        Args:
            data_split: Data split name
            radii: List of radii to use
            threshold: SASA threshold for surface residues
            num_posenc: Number of positional encoding features
            num_rbf: Number of RBF features
            zero_ratio: Ratio to downsample graphs with recall=0
            undersample: Undersample parameter (int for count, float for ratio)
            cache_dir: Directory to cache processed data
            force_rebuild: Whether to force rebuild the dataset
            verbose: Whether to print progress information
        """
        self.data_split = data_split
        self.radii = radii
        self.verbose = verbose
        
        # Create individual datasets
        self.datasets = []
        for radius in radii:
            dataset = SphereGraphDataset(
                data_split=data_split,
                radius=radius,
                threshold=threshold,
                num_posenc=num_posenc,
                num_rbf=num_rbf,
                zero_ratio=zero_ratio,
                undersample=undersample,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
                verbose=verbose,
                use_embeddings2=use_embeddings2
            )
            self.datasets.append(dataset)
        
        # Combine all data
        self.data_list = []
        for dataset in self.datasets:
            self.data_list.extend(dataset.data_list)
        
        if verbose:
            print(f"Combined {len(self.datasets)} datasets with {len(self.data_list)} total samples")
        
        super().__init__()
    
    def len(self) -> int:
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        return self.data_list[idx]
    
    def apply_filters(self, undersample: Union[int, float, None] = None, seed: int = 42):
        """
        Apply filtering to the loaded multi-radius dataset.
        
        Args:
            undersample: Undersample parameter (int for count, float for ratio)
            seed: Random seed for reproducibility
        """
        if undersample is not None:
            original_size = len(self.data_list)
            self.data_list = apply_undersample(self.data_list, undersample, seed=seed, verbose=True)
    
    def get_stats(self) -> Dict:
        """Get combined dataset statistics."""
        if not self.data_list:
            return {}
        
        # Collect statistics
        num_nodes_list = [data.num_nodes for data in self.data_list]
        recall_list = [data.y.item() for data in self.data_list]
        
        # Node-level statistics
        total_nodes = sum(num_nodes_list)
        total_epitopes = sum([data.y_node.sum().item() for data in self.data_list])
        
        stats = {
            'num_graphs': len(self.data_list),
            'num_radii': len(self.radii),
            'radii': self.radii,
            'avg_nodes_per_graph': np.mean(num_nodes_list),
            'std_nodes_per_graph': np.std(num_nodes_list),
            'min_nodes_per_graph': np.min(num_nodes_list),
            'max_nodes_per_graph': np.max(num_nodes_list),
            'total_nodes': total_nodes,
            'total_epitopes': total_epitopes,
            'epitope_ratio': total_epitopes / total_nodes if total_nodes > 0 else 0,
            'avg_recall': np.mean(recall_list),
        }
        
        return stats

    def print_stats(self):
        """Print dataset statistics."""
        stats = self.get_stats()
        if not stats:
            print("No statistics available (empty dataset)")
            return
        
        print(f"\n=== {self.data_split.upper()} Dataset Statistics ===")
        print(f"Number of graphs: {stats['num_graphs']:,}")
        print(f"Average nodes per graph: {stats['avg_nodes_per_graph']:.1f} ± {stats['std_nodes_per_graph']:.1f}")
        print(f"Nodes per graph range: [{stats['min_nodes_per_graph']}, {stats['max_nodes_per_graph']}]")
        print(f"Total nodes: {stats['total_nodes']:,}")
        print(f"Total epitope nodes: {stats['total_epitopes']:,}")
        print(f"Epitope ratio: {stats['epitope_ratio']:.3f}")
        print(f"Average recall: {stats['avg_recall']:.3f} ± {stats['std_recall']:.3f}")
        print(f"Average precision: {stats['avg_precision']:.3f} ± {stats['std_precision']:.3f}")
        print("=" * 40)



# Utility functions for dataset creation and management
def create_datasets(
    radii: List[int] = [16, 18, 20],
    splits: List[str] = ["train", "test"],
    threshold: float = 0.25,
    zero_ratio: float = None,
    undersample: Union[int, float, None] = None,
    cache_dir: Optional[str] = None,
    force_rebuild: bool = False,
    verbose: bool = False,
    seed: int = 42,
    use_embeddings2: bool = False,
) -> Dict[str, SphereGraphDataset]:
    """
    Create optimized datasets for all splits and radii.
    
    Args:
        radii: List of radii to use
        splits: List of data splits to create
        threshold: SASA threshold for surface residues
        zero_ratio: Ratio to downsample graphs with recall=0
        undersample: Undersample parameter (int for count, float for ratio)
        cache_dir: Directory to cache processed data
        force_rebuild: Whether to force rebuild datasets
        verbose: Whether to print progress information
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping split names to datasets
    """
    datasets = {}
    
    for split in splits:
        if len(radii) == 1:
            # Single radius dataset
            dataset = SphereGraphDataset(
                data_split=split,
                radius=radii[0],
                threshold=threshold,
                zero_ratio=zero_ratio,
                undersample=undersample,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
                verbose=verbose,
                seed=seed,
                use_embeddings2=use_embeddings2
            )
            if verbose:
                dataset.print_stats()
        else:
            # Multi-radius dataset
            dataset = MultiRadiusGraphDataset(
                data_split=split,
                radii=radii,
                threshold=threshold,
                zero_ratio=zero_ratio,
                undersample=undersample,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
                verbose=verbose,
                use_embeddings2=use_embeddings2
            )
        
        datasets[split] = dataset
    
    return datasets


def custom_collate_fn(batch):
    """
    Custom collate function for ReGEP model.
    Converts PyG Data objects to the format expected by ReGEP.
    """
    # Use PyG's default batching
    batched_data = Batch.from_data_list(batch)
    
    # ReGEP expects the input features to be concatenated
    # x: [N_total, embed_dim], rsa: [N_total], ss: [N_total, 2]
    # The model will concatenate them internally: [x, rsa, ss] -> [N_total, embed_dim + 3]
    
    return batched_data


class ReGEPDataLoader(DataLoader):
    """
    Custom DataLoader for ReGEP model that handles the specific input format.
    Supports undersampling at the DataLoader level.
    """
    
    def __init__(self, dataset, batch_size=32, shuffle=True, **kwargs):
        """
        Initialize ReGEP DataLoader with optional undersampling.
        
        Args:
            dataset: The dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            **kwargs: Additional arguments for DataLoader
        """
        # Set default collate_fn if not provided
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = custom_collate_fn
            
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

def create_data_loader(
    radii=[16, 18, 20],
    batch_size=32,
    zero_ratio=0.1,
    undersample=0.5,
    seed=42,
    verbose=False,
    use_embeddings2=False,
    val=False,
    **kwargs
):
    """
    Create train, validation (optional), and test data loaders.
    
    Args:
        radii (list): List of radii for data processing
        batch_size (int): Batch size for training
        zero_ratio (float): Ratio of zero samples for training
        undersample (float): Undersampling ratio for training
        seed (int): Random seed
        verbose (bool): Whether to print verbose information
        use_embeddings2 (bool): Whether to use embeddings2
        val (bool): Whether to split train data into train/val (8:2)
        **kwargs: Additional arguments for data loader
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) if val=True
               (train_loader, test_loader) if val=False
    """
    if val:
        # Create full train dataset first
        full_train_dataset = create_datasets(
            radii=radii,
            splits=["train"],
            threshold=0.25,
            undersample=None,  # Don't undersample the full dataset yet
            zero_ratio=zero_ratio,
            cache_dir=None,
            seed=seed,
            verbose=verbose,
            use_embeddings2=use_embeddings2
        )["train"]
        
        # Split at protein level to avoid data leakage (8:2)
        # First, extract all unique protein_keys from the dataset
        protein_keys = set()
        protein_to_indices = {}
        
        for idx, data in enumerate(full_train_dataset):
            protein_key = f"{data.pdb_id}_{data.chain_id}"
            protein_keys.add(protein_key)
            if protein_key not in protein_to_indices:
                protein_to_indices[protein_key] = []
            protein_to_indices[protein_key].append(idx)
        
        # Convert to list and split at protein level
        protein_keys_list = list(protein_keys)
        total_proteins = len(protein_keys_list)
        train_proteins_count = int(0.8 * total_proteins)
        
        # Set random seed for reproducible split
        torch.manual_seed(seed)
        random.seed(seed)
        random.shuffle(protein_keys_list)
        
        train_proteins = set(protein_keys_list[:train_proteins_count])
        val_proteins = set(protein_keys_list[train_proteins_count:])
        
        # Collect indices for train and val based on protein split
        train_indices = []
        val_indices = []
        
        for protein_key, indices in protein_to_indices.items():
            if protein_key in train_proteins:
                train_indices.extend(indices)
            else:
                val_indices.extend(indices)
        
        total_size = len(full_train_dataset)
        train_size = len(train_indices)
        val_size = len(val_indices)
        
        if verbose:
            print(f"[INFO] Splitting at protein level:")
            print(f"  Total proteins: {total_proteins}")
            print(f"  Train proteins: {len(train_proteins)} ({len(train_proteins)/total_proteins:.1%})")
            print(f"  Val proteins: {len(val_proteins)} ({len(val_proteins)/total_proteins:.1%})")
            print(f"  Total samples: {total_size} -> train: {train_size}, val: {val_size}")
            
            # Show some example proteins for verification
            print(f"  Sample train proteins: {list(train_proteins)[:3]}...")
            print(f"  Sample val proteins: {list(val_proteins)[:3]}...")
        
        # Create train and val datasets using subset
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
        
        # Apply undersampling to train dataset if specified
        if undersample is not None:
            original_train_size = len(train_dataset)
            # Create a wrapper to apply undersampling
            train_data_list = [full_train_dataset[i] for i in train_indices]
            train_data_list = apply_undersample(
                train_data_list,
                undersample,
                seed=seed,
                verbose=verbose
            )
            # Create new dataset from undersampled data
            class ListDataset(torch.utils.data.Dataset):
                def __init__(self, data_list):
                    self.data_list = data_list
                def __len__(self):
                    return len(self.data_list)
                def __getitem__(self, idx):
                    return self.data_list[idx]
            train_dataset = ListDataset(train_data_list)
        
        # Create data loaders
        train_loader = ReGEPDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            **kwargs
        )
        
        val_loader = ReGEPDataLoader(
            val_dataset,
            batch_size=batch_size*2,
            shuffle=False,
            collate_fn=custom_collate_fn,
            **kwargs
        )
        
        if verbose:
            print(f"[INFO] Train samples after undersampling: {len(train_dataset)}")
            print(f"[INFO] Val samples: {len(val_dataset)}")
            
            # Verify no protein overlap between train and val
            _verify_protein_split(train_dataset, val_dataset, full_train_dataset, verbose=True)
            
    else:
        # Original behavior: no validation split
        train_dataset = create_datasets(
                radii=radii,
                splits=["train"],
                threshold=0.25,
                undersample=undersample,
                zero_ratio=zero_ratio,
                cache_dir=None,
                seed=seed,
                verbose=verbose,
                use_embeddings2=use_embeddings2
            )["train"]
        
        train_loader = ReGEPDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            **kwargs
        )
        
        val_loader = None
        
        if verbose:
            print(f"[INFO] Train samples: {len(train_dataset)}")
    
    # Create test dataset (same for both cases)
    test_dataset = create_datasets(
            radii=radii,
            splits=["test"],
            threshold=0.25,
            undersample=None,
            zero_ratio=None,
            cache_dir=None,
            verbose=verbose,
            use_embeddings2=use_embeddings2
        )["test"]
    
    test_loader = ReGEPDataLoader(
        test_dataset,
        batch_size=batch_size*4,
        shuffle=False,
        **kwargs
    )
    
    if verbose:
        print(f"[INFO] Test samples: {len(test_dataset)}")
    
    if val:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader


def extract_antigens_from_dataset(dataset):
    """
    Extract unique (pdb_id, chain_id) pairs from a dataset.
    
    Args:
        dataset: PyTorch dataset containing graph data with pdb_id and chain_id attributes
        
    Returns:
        List of unique (pdb_id, chain_id) tuples
    """
    antigens = []
    seen = set()
    
    for data in dataset:
        pdb_id = data.pdb_id
        chain_id = data.chain_id
        antigen_key = (pdb_id, chain_id)
        
        if antigen_key not in seen:
            antigens.append(antigen_key)
            seen.add(antigen_key)
    
    return antigens


def _verify_protein_split(train_dataset, val_dataset, full_dataset, verbose=True):
    """
    Verify that train and validation datasets have no overlapping proteins.
    
    Args:
        train_dataset: Training dataset subset
        val_dataset: Validation dataset subset
        full_dataset: Original full dataset
        verbose: Whether to print verification results
    """
    def extract_proteins_from_subset(dataset):
        """Extract protein keys from a dataset (handles both Subset and regular datasets)"""
        proteins = set()
        for i in range(len(dataset)):
            try:
                data = dataset[i]
                protein_key = f"{data.pdb_id}_{data.chain_id}"
                proteins.add(protein_key)
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Could not extract protein info from index {i}: {e}")
                continue
        return proteins
    
    # Extract protein keys from both datasets
    train_proteins = extract_proteins_from_subset(train_dataset)
    val_proteins = extract_proteins_from_subset(val_dataset)
    
    # Check for overlap
    overlap = train_proteins.intersection(val_proteins)
    
    if verbose:
        print(f"[INFO] Protein split verification:")
        print(f"  Train proteins: {len(train_proteins)}")
        print(f"  Val proteins: {len(val_proteins)}")
        print(f"  Overlap: {len(overlap)}")
        
        if len(overlap) > 0:
            print(f"  [WARNING] Found {len(overlap)} overlapping proteins!")
            print(f"  First few overlapping proteins: {list(overlap)[:5]}")
            return False
        else:
            print(f"  [SUCCESS] No protein overlap detected - proper data split!")
            return True
    
    return len(overlap) == 0
    