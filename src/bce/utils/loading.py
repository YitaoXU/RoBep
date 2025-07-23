import os
import h5py 
from pathlib import Path
import yaml
import pickle

import concurrent.futures
from tqdm import tqdm
import json

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch

from bce.utils.constants import *

# load combined epitopes csv
def load_epitopes_csv(csv_name: str = "epitopes.csv") -> pd.DataFrame:
    epitopes_csv = Path(BASE_DIR) / "data" / 'epitopes' / csv_name
    if not epitopes_csv.exists():
        raise FileNotFoundError(f"[Error] Epitopes CSV not found at {epitopes_csv}")
    df = pd.read_csv(epitopes_csv)
    if df.empty:
        print(f"[Warning] The CSV {epitopes_csv} is empty.")
        return None
    
    unique_protein_chains = set()
    epitope_dict = {}

    for _, row in df.iterrows():
        antigen_name = row.get("antigen_name", "N/A")
        binary_label = row.get("binary_label", "")

        # Split antigen name into pdb and chain
        if "_" in antigen_name:
            pdb, chain = antigen_name.split("_", 1)
        else:
            pdb, chain = antigen_name, "A"  # Default to chain A if no underscore

        unique_protein_chains.add((pdb, chain))

        # Convert binary_label string to list of integers
        if isinstance(binary_label, str):
            binary_list = [int(char) for char in binary_label if char in ['0', '1']]
        else:
            binary_list = []
        
        epitope_dict[antigen_name] = binary_list

    return df, list(unique_protein_chains), epitope_dict

def load_epitopes_csv_single(
    csv_path: str = None
) -> pd.DataFrame:
    if csv_path is None:
        epitopes_csv = Path(BASE_DIR)/ "data" / "epitopes" / "epitopes_13.csv"
    else:
        epitopes_csv = Path(csv_path)
    if not epitopes_csv.exists():
        raise FileNotFoundError(f"[Error] Epitopes CSV not found at {epitopes_csv}")
    df = pd.read_csv(epitopes_csv)
    if df.empty:
        print(f"[Warning] The CSV {epitopes_csv} is empty.")
        return None
    
    unique_protein_chains = set()
    epitope_dict = {}

    for _, row in df.iterrows():
        antigen = row.get("antigen_chain", "N/A")
        epitopes = row.get("epitopes", "")

        pdb, chain = antigen.split("_")

        unique_protein_chains.add((pdb, chain))

        # Parse out epitope residue numbers
        epitope_nums = []
        if isinstance(epitopes, str):
            for e in epitopes.split(","):
                e = e.strip()
                if "_" in e:
                    parts = e.split("_", 1)
                    try:
                        ep_num = int(parts[0])
                        epitope_nums.append(ep_num)
                    except ValueError:
                        pass
        
        epitope_dict[antigen] = epitope_nums

    return df, list(unique_protein_chains), epitope_dict

def load_species(species_path: str = f"{BASE_DIR}/data/species.json") -> Dict[str, Dict[str, str]]:
    with open(species_path, "r") as f:
        species = json.load(f)
    return species

def load_data_split(data_split: str, verbose: bool = True) -> List[Tuple[str, str]]:
    """
    Load the antigens for the specified data split.
    
    Args:
        data_split: Data split name ('train', 'val', 'test')
        
    Returns:
        List of (pdb_id, chain_id) tuples for the split
    """
    splits_file = Path(BASE_DIR) / "data" / "epitopes" / "data_splits.json"
    
    if not splits_file.exists():
        if verbose:
            print(f"Data splits file not found: {splits_file}")
            print("Using all antigens from load_epitopes_csv()")
        _, antigens, _ = load_epitopes_csv()
        return antigens
    
    try:
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        if data_split not in splits:
            raise KeyError(f"Split '{data_split}' not found in splits file")
        
        antigens = [(item[0], item[1]) for item in splits[data_split]]
        
        if verbose:
            print(f"Loaded {len(antigens)} antigens for {data_split} split")
        
        return antigens
        
    except Exception as e:
        if verbose:
            print(f"Error loading data splits: {str(e)}")
            print("Falling back to load_epitopes_csv()")
        _, antigens, _ = load_epitopes_csv()
        return antigens

def load_split_antigens(base_dir=DISK_DIR / "data", split="train"):
    """
    Load the antigen list for a specific data split.
    
    Args:
        base_dir (str): Base directory where split files are stored
        split (str): One of "train", "val", or "test"
        
    Returns:
        list: List of (pdb_id, chain_id) tuples for the specified split
    """
    
    base_dir = Path(base_dir)
    pickle_path = base_dir / f"{split}_antigens.pkl"
    
    if not pickle_path.exists():
        raise FileNotFoundError(f"Split file not found: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        antigens = pickle.load(f)
    
    print(f"[INFO] Loaded {len(antigens)} antigens for {split} split")
    return antigens

# load protein embedding extracted by ESM-C
def load_protein_embedding(
    pdb_chain: Tuple[str, str],
    embedding_dir: Optional[str] = None,
    mode: str = "full"
) -> np.ndarray:
    """
    Retrieve either full or mean embeddings for a given (PDB ID, Chain ID) pair.

    Args:
        pdb_chain (Tuple[str, str]): Protein identifier (pdb_id, chain_id).
        embedding_dir (Optional[str]): Directory where embeddings are stored.
        mode (str): "mean" to retrieve mean embedding (from .npy) or
                    "full" to retrieve full embedding (from .h5).

    Returns:
        np.ndarray: The requested embedding (mean: (embed_dim,), full: (seq_len, embed_dim)).
    """
    pdb_id, chain_id = pdb_chain
    
    # Handle default embedding directory selection
    if embedding_dir is None:
        embedding_dir = EMBEDDING_DIR if mode == "mean" else FULL_EMBEDDING_DIR
        embedding_dir = Path(embedding_dir) / "esmc-6b"

    embedding_dir = Path(embedding_dir)  # Ensure it's a Path object

    if mode == "mean":
        mean_file = embedding_dir / f"{pdb_id}_{chain_id}.npy"
        if not mean_file.exists():
            raise FileNotFoundError(f"Mean embedding file not found: {mean_file}")
        return np.load(mean_file)  # Shape: (embed_dim,)

    elif mode == "full":
        full_file = embedding_dir / f"{pdb_id}_{chain_id}.h5"
        if not full_file.exists():
            raise FileNotFoundError(f"Full embedding file not found: {full_file}")
        with h5py.File(full_file, "r") as h5f:
            return h5f["embedding"][:]  # Shape: (seq_len, embed_dim)

    else:
        raise ValueError("Invalid mode. Use 'mean' or 'full'.")
