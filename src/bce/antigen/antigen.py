from __future__ import annotations

import json
import h5py
import traceback
import numpy as np
import traceback
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, BinaryIO, TextIO
from dataclasses import dataclass
from scipy.spatial.distance import cdist

import torch

# ESM
from esm.utils import residue_constants as RC
from esm.utils.structure.protein_chain import ProteinChain

# Biotite
import biotite.structure as bs
from biotite.database import rcsb
from biotite.structure.io.pdb import PDBFile
from biotite.structure import annotate_sse

from cloudpathlib import CloudPath
from Bio.Data import PDBData  # Ensure BioPython is imported.

import py3Dmol

# RoBep Packages
from ..utils.constants import BASE_DIR
from ..utils.loading import load_epitopes_csv, load_epitopes_csv_single, load_species
from .pc import AMINO_ACID_1TO3, AMINO_ACID_3TO1, MAX_ASA
from ..model.RoBep import RoBep
from ..data.utils import create_graph_data


PathOrBuffer = Union[str, Path, BinaryIO, TextIO]

@dataclass
class AntigenChain(ProteinChain):
    """
    Extended ProteinChain class that adds additional functionalities,
    such as computing surface residues based on SASA and maxASA constants.
    """
    def __post_init__(self, token: Optional[str] = "1mzAo8l1uxaU8UfVcGgV7B"):
        super().__post_init__()  # Ensure parent class initialization
        
        # Map residue number to index
        self.resnum_to_index = {int(rnum): i for i, rnum in enumerate(self.residue_index)}
        
        # Get epitopes as boolean array
        self.epitopes = self.get_epitopes()  # Automatically get epitopes on initialization
        
        # Set token from parameter or environment variable
        self.token = token
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def convert_letter_1to3(letter: str) -> str:
        """
        Convert a one-letter amino acid code to its corresponding three-letter code.
        
        Args:
            letter (str): A single-character amino acid code (e.g., "A").
            
        Returns:
            str: The corresponding three-letter code (e.g., "ALA").
                 Returns "UNK" if the code is not recognized.
        """
        return AMINO_ACID_1TO3.get(letter.upper(), "UNK")

    @staticmethod
    def convert_letter_3to1(three_letter: str) -> str:
        """
        Convert a three-letter amino acid code to its corresponding one-letter code.
        
        Args:
            three_letter (str): A three-letter amino acid code (e.g., "ALA").
            
        Returns:
            str: The corresponding one-letter code (e.g., "A").
                 Returns "X" if the code is not recognized.
        """
        return AMINO_ACID_3TO1.get(three_letter.upper(), "X")
    
    def get_species(self) -> str:
        """
        Get the species of the antigen.
        """
        from ..utils.tools import get_chain_organism
        
        species_dict = load_species()
        if self.id in species_dict:
            species = species_dict[self.id]['classification']
        else:
            try:
                species = get_chain_organism(self.id, self.chain_id)
                species_dict[self.id] = {'classification': species}
                
                # Create directory if it doesn't exist
                species_file_path = Path(f"{BASE_DIR}/data/species.json")
                species_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(species_file_path, "w") as f:
                    json.dump(species_dict, f, indent=2)
            except Exception as e:
                print(f"[ERROR] Failed to get species for {self.id}_{self.chain_id}: {str(e)}")
                species = "Unknown"
        return species
    
    def get_backbone_atoms(self) -> np.ndarray:
        """
        Get backbone atom coordinates in the order: CA, C, N.

        Returns:
            np.ndarray: [L, 3, 3] array where [:, 0] is CA, [:, 1] is C, [:, 2] is N.
        """
        file = Path(f"{BASE_DIR}/data/coords/{self.id}_{self.chain_id}.npy")
        
        if file.exists():
            return np.load(file)
        else:
            idx_CA = RC.atom_order["CA"]
            idx_C = RC.atom_order["C"]
            idx_N = RC.atom_order["N"]

            backbone_atoms = self.atom37_positions[:, [idx_N, idx_CA, idx_C], :]  # shape: [L, 3, 3]
            
            # Create directory if it doesn't exist
            file.parent.mkdir(parents=True, exist_ok=True)
            np.save(file, backbone_atoms)
            return backbone_atoms
        
    def get_secondary_structure(self) -> np.ndarray:
        """
        Get secondary structure information using numpy operations.
        """
        try:
            ss3_arr = annotate_sse(self.atom_array)
            biotite_ss3_str = "".join(ss3_arr)
            
            if len(biotite_ss3_str) != len(self.sequence):
                print(f"[WARNING] Secondary structure prediction length ({len(biotite_ss3_str)}) "
                    f"doesn't match sequence length ({len(self.sequence)}) "
                    f"for protein {self.id}_{self.chain_id}")
                return None
                
            translation_table = str.maketrans({
                "a": "H",  # alpha helix
                "b": "E",  # beta sheet
                "c": "C",  # coil
            })
            return biotite_ss3_str.translate(translation_table)
            
        except Exception as e:
            print(f"[ERROR] Failed to predict secondary structure for "
                f"{self.id}_{self.chain_id}: {str(e)}")
            return None
    
    def get_ss_onehot(self) -> np.ndarray:
        """
        Get one-hot encoded secondary structure information using numpy operations.
        Only encode H (helix) and E (sheet), as C (coil) can be inferred.
        
        Returns:
            np.ndarray: One-hot encoded secondary structure array of shape (seq_len, 2)
                       where 2 represents [H, E] (Helix, Sheet)
        """
        self.secondary_structure = self.get_secondary_structure()
        seq_len = len(self.secondary_structure)
        ss_onehot = np.zeros((seq_len, 2), dtype=np.float32)
        
        # Use boolean indexing for helix and sheet only
        ss_array = np.array(list(self.secondary_structure))
        ss_onehot[:, 0] = (ss_array == 'H')
        ss_onehot[:, 1] = (ss_array == 'E')
        
        return ss_onehot

    def get_rsa(self) -> np.ndarray:
        """
        Calculate relative solvent accessibility (RSA) for all residues.
        RSA is the ratio of SASA to maximum ASA for each residue.
        
        Returns:
            np.ndarray: An array of RSA values for each residue in the sequence.
        """
        
        cache_file = Path(BASE_DIR) / "data" / "rsa" / f"{self.id}_{self.chain_id}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        
        sasa_values = self.sasa()  # Get SASA values for all residues
        rsa_values = np.zeros(len(self.sequence), dtype=np.float32)
        
        # Calculate RSA for each residue
        for i, (letter, sasa) in enumerate(zip(self.sequence, sasa_values)):
            three_letter = self.convert_letter_1to3(letter)
            max_asa = MAX_ASA.get(three_letter)
            if max_asa is not None and max_asa != 0:
                rsa_values[i] = sasa / max_asa
        
        # Create directory if it doesn't exist
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, rsa_values)
            
        return rsa_values

    def get_surface_residues(self, threshold: float = 0.25) -> list:
        """
        Identify surface-exposed residues using RSA values.
        
        A residue is considered surface-exposed if its RSA value
        is at least `threshold`.
        
        Args:
            threshold (float): The minimum RSA value required to consider 
                             the residue as surface-exposed.
        
        Returns:
            tuple: A tuple of two lists, where the first list contains residue numbers (from the PDB) that are surface-exposed,
                   and the second list contains the indices of the surface residues in the sequence.
        """
        rsa_values = self.get_rsa()
        surface_residue_numbers = []
        surface_residue_indices = []
        
        # Identify surface residues based on RSA threshold
        for idx, rsa in enumerate(rsa_values):
            if rsa >= threshold:
                surface_residue_numbers.append(int(self.residue_index[idx]))
                surface_residue_indices.append(idx)
                
        return surface_residue_numbers, surface_residue_indices
    
    def get_epitopes(self, threshold: float = 0.25, csv_name: str = None) -> np.ndarray:
        """
        Retrieve epitopes for this chain as a boolean array.
        
        Args:
            threshold (float): SASA threshold for determining surface residues.
        
        Returns:
            np.ndarray: A boolean array of length L (sequence length) where True indicates 
                       epitope positions and False indicates non-epitope positions.
                       Only surface-exposed residues can be True.
        """
        _, _, epitopes = load_epitopes_csv(csv_name=csv_name)

        if f'{self.id}_{self.chain_id}' in epitopes:
            binary_labels = epitopes.get(f'{self.id}_{self.chain_id}', [0] * len(self.sequence)) # default to 0 if not found
        else:
            print(f"[WARNING] Epitopes not found for {self.id}_{self.chain_id}. Use single epitopes.")
            binary_labels = self.get_epitopes_single()
            
        # Initialize epitope array with False values
        epitope_array = np.zeros(len(self.sequence), dtype=bool)
        
        # Check if we have binary labels - handle both list and numpy array cases
        if binary_labels is not None and len(binary_labels) > 0:
            # Ensure the binary labels match the sequence length
            if len(binary_labels) == len(self.sequence):
                epitope_array = np.array(binary_labels, dtype=bool)
            else:
                print(f"[WARNING] Binary labels length ({len(binary_labels)}) doesn't match "
                      f"sequence length ({len(self.sequence)}) for {self.id}_{self.chain_id}")
                return epitope_array
            
        if threshold == 0.0:
            return epitope_array
        
        # Filter to ensure only surface residues can be epitopes
        _, surface_indices = self.get_surface_residues(threshold=threshold)
        
        # Create surface mask: True for surface residues, False for buried residues
        surface_mask = np.zeros(len(self.sequence), dtype=bool)
        for res_idx in surface_indices:
            if 0 <= res_idx < len(self.sequence):
                surface_mask[res_idx] = True
        
        # Apply surface filter: epitopes can only be surface residues
        epitope_array = epitope_array & surface_mask
        
        return epitope_array

    def get_epitopes_single(self) -> np.ndarray:
        """
        Retrieve epitopes for this chain as a boolean array.
        """
        _, _, epitopes = load_epitopes_csv_single()
        
        # Try different key formats to find epitopes
        possible_keys = [
            f'{self.id.upper()}_{self.chain_id}',
            f'{self.id}_{self.chain_id}',
            f'{self.id.lower()}_{self.chain_id}'
        ]
        
        epitopes_resnums = None
        for key in possible_keys:
            if key in epitopes:
                epitopes_resnums = epitopes.get(key)
                break
        
        if epitopes_resnums is not None:
            epitope_array = np.zeros(len(self.sequence), dtype=int)
            for resnum in epitopes_resnums:
                if resnum in self.resnum_to_index:
                    epitope_array[self.resnum_to_index[resnum]] = 1
            return epitope_array
        else:
            print(f"[WARNING] Single Epitopes not found for {self.id}_{self.chain_id}. Use no epitopes.")
            epitope_array = np.zeros(len(self.sequence), dtype=int)
        
        return epitope_array
    
    def annotate_epitopes(self, antibody_h, antibody_l = None, cutoff: float = 4.0, 
                          threshold: float = 0.25) -> np.ndarray:
        """
        Annotate epitopes for this chain based on distance to antibody chains.
        
        An epitope residue is defined as one having at least one heavy atom within 
        the distance cutoff to at least one heavy atom from the antibody heavy 
        chain or light chain (if provided). Only surface-exposed residues can be
        annotated as epitopes.
        
        Args:
            antibody_h: Heavy chain of antibody (AntigenChain object)
            antibody_l: Light chain of antibody (AntigenChain object, optional)  
            cutoff: Distance threshold in Angstroms for epitope definition (default: 4.0)
            threshold: SASA threshold for determining surface residues (default: 0.25)
            
        Returns:
            np.ndarray: Boolean array of length L (sequence length) where True indicates 
                       epitope positions and False indicates non-epitope positions.
                       Only surface-exposed residues can be True.
        """
        def extract_heavy_atoms(protein_chain):
            """Extract heavy atom coordinates and their residue indices."""
            heavy_atoms = []
            residue_indices = []
            
            for res_idx in range(len(protein_chain.sequence)):
                mask = protein_chain.atom37_mask[res_idx]
                coords = protein_chain.atom37_positions[res_idx][mask]
                
                # Get atom names for this residue
                atom_names = [RC.atom_types[i] for i, exists in enumerate(mask) if exists]
                
                # Filter heavy atoms (non-hydrogen atoms)
                for atom_name, coord in zip(atom_names, coords):
                    # Skip hydrogen atoms (typically start with 'H')
                    if not atom_name.startswith('H') and not np.any(np.isnan(coord)):
                        heavy_atoms.append(coord)
                        residue_indices.append(res_idx)
            
            return np.array(heavy_atoms) if heavy_atoms else np.empty((0, 3)), residue_indices
        
        # Extract heavy atoms from antigen
        antigen_heavy_atoms, antigen_residue_indices = extract_heavy_atoms(self)
        
        if len(antigen_heavy_atoms) == 0:
            print(f"[WARNING] No heavy atoms found in antigen {self.id}_{self.chain_id}")
            return np.zeros(len(self.sequence), dtype=bool)
        
        # Extract heavy atoms from antibodies
        all_antibody_atoms = []
        
        for antibody in [antibody_h, antibody_l]:
            if antibody is None:
                continue
                
            ab_heavy_atoms, _ = extract_heavy_atoms(antibody)
            if len(ab_heavy_atoms) > 0:
                all_antibody_atoms.append(ab_heavy_atoms)
        
        if not all_antibody_atoms:
            print(f"[WARNING] No heavy atoms found in antibody chains")
            return np.zeros(len(self.sequence), dtype=bool)
        
        # Combine all antibody heavy atoms
        antibody_heavy_atoms = np.vstack(all_antibody_atoms)
        
        # Compute distance matrix using cdist
        try:
            distances = cdist(antigen_heavy_atoms, antibody_heavy_atoms)
        except Exception as e:
            print(f"[ERROR] Failed to compute distance matrix: {str(e)}")
            return np.zeros(len(self.sequence), dtype=bool)
        
        # For each antigen residue, check if any of its heavy atoms are within cutoff distance
        epitopes = np.zeros(len(self.sequence), dtype=bool)
        
        for res_idx in range(len(self.sequence)):
            # Get indices of heavy atoms belonging to this residue
            atom_indices = [i for i, r_idx in enumerate(antigen_residue_indices) if r_idx == res_idx]
            
            if atom_indices:
                # Check if any atom of this residue is within cutoff distance
                min_distance = np.min(distances[atom_indices, :])
                if min_distance <= cutoff:
                    epitopes[res_idx] = True
        
        # Apply surface filter: epitopes can only be surface residues
        if threshold > 0.0:
            _, surface_indices = self.get_surface_residues(threshold=threshold)
            
            # Create surface mask: True for surface residues, False for buried residues
            surface_mask = np.zeros(len(self.sequence), dtype=bool)
            for res_idx in surface_indices:
                if 0 <= res_idx < len(self.sequence):
                    surface_mask[res_idx] = True
            
            # Apply surface filter: epitopes can only be surface residues
            epitopes_before_filter = np.sum(epitopes)
            epitopes = epitopes & surface_mask
            epitopes_after_filter = np.sum(epitopes)
            
            if epitopes_before_filter > epitopes_after_filter:
                print(f"[INFO] Surface filter removed {epitopes_before_filter - epitopes_after_filter} "
                      f"buried residues from epitope annotation")
        
        if np.sum(epitopes) == 0:
            print(f"[WARNING] No epitope residues found for {self.id}_{self.chain_id} at cutoff {cutoff}Å "
                  f"and surface threshold {threshold}")
        else:
            print(f"[INFO] Found {np.sum(epitopes)} epitope residues for {self.id}_{self.chain_id} "
                  f"at cutoff {cutoff}Å and surface threshold {threshold}")
        
        self.epitopes = epitopes
        
        return epitopes
    
    def get_epitope_residue_numbers(self) -> list:
        """
        Get epitope residue numbers from the boolean epitope array.
        
        Returns:
            list: List of residue numbers that are epitopes.
        """
        epitope_indices = np.where(self.epitopes)[0]
        epitope_residue_numbers = [int(self.residue_index[idx]) for idx in epitope_indices]
        return epitope_residue_numbers

    def get_embeddings(self, override: bool = False, encoder: str = "esmc") -> np.ndarray:
        """
        Retrieve or compute per-residue (full) ESM-C embeddings.

        Returns:
            np.ndarray: Array of shape (seq_len, embed_dim), dtype float32.
        """
        full_file = Path(BASE_DIR) / "data" / "embeddings" / f"{encoder}" / f"{self.id}_{self.chain_id}.h5"

        if full_file.exists() and not override:
            with h5py.File(full_file, "r") as h5f:
                full_embedding = h5f["embedding"][:]
        else:
            if encoder == "esmc":
                if self.token is None:
                    raise ValueError("ESM token is not set. Please go to https://forge.evolutionaryscale.ai/ to get a token.")
                
                else:
                    print(f"[INFO] Generating with ESM-C...")
                    
                    from esm.sdk.api import ESMProtein, LogitsConfig
                    from esm.sdk.forge import ESM3ForgeInferenceClient

                    token = self.token
                    model = ESM3ForgeInferenceClient(
                        model="esmc-6b-2024-12",
                        url="https://forge.evolutionaryscale.ai",
                        token=token
                    )
                    config = LogitsConfig(sequence=True, return_embeddings=True)

                    sequence = self.sequence[:2046]  # truncate if too long
                    protein = ESMProtein(sequence)
                    protein_tensor = model.encode(protein)
                    output = model.logits(protein_tensor, config)
                    full_embedding = output.embeddings.squeeze(0)[1:-1, :].to(torch.float32).cpu().numpy()

                    full_file.parent.mkdir(parents=True, exist_ok=True)
                    with h5py.File(full_file, "w") as h5f:
                        h5f.create_dataset("embedding", data=full_embedding, compression="gzip")
            
            elif encoder == "esm2":
                model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
                batch_converter = alphabet.get_batch_converter()
                model.eval()
                data = [
                        ("antigen", self.sequence[:2046])
                    ]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                model.to(self.device)
                batch_tokens = batch_tokens.to(self.device)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                    token_representations = results["representations"][33]
                    full_embedding = token_representations.squeeze(0)[1:-1, :].to(torch.float32).cpu().numpy()
                        
                    full_file.parent.mkdir(parents=True, exist_ok=True)
                    with h5py.File(full_file, "w") as h5f:
                        h5f.create_dataset("embedding", data=full_embedding, compression="gzip")

        return full_embedding
        
    def _scan_surface_residues(self, radius: float, threshold: float = 0.25) -> tuple:
        """
        Helper function to compute the surface coverage for each surface residue.
        For each surface residue, using its C_alpha coordinate as the center of a sphere with
        radius `radius`, determine which surface residues are covered.
        
        Args:
            radius (float): The radius of the sphere (in Ångstroms)
            threshold (float): Fraction of maximum ASA to define a residue as surface-exposed
        
        Returns:
            tuple:
                - coverage (dict): Mapping from center residue index to:
                    (list[int]): List of covered residue indices
                    (list[int]): List of covered epitope residue indices
                    (float): Precision
                    (float): Recall
                - max_recall_res (int): Center residue index with highest recall
                - max_precision_res (int): Center residue index with highest precision
        """
        # Input validation
        if radius <= 0:
            raise ValueError("Radius must be positive")
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        # Get surface residues number and indices
        surface_res_nums, surface_indices = self.get_surface_residues(threshold=threshold)
        
        # Ensure indices are valid
        valid_surface_indices = [
            idx for idx in surface_indices 
            if 0 <= idx < len(self.sequence)
        ]
        valid_surface_res_nums = [
            surface_res_nums[surface_indices.index(idx)] 
            for idx in valid_surface_indices
        ]
        
        if not valid_surface_indices:
            return {}, None, None

        # Collect all atoms and their residue indices from surface residues
        all_atoms = []
        all_res_indices = []
        for idx in valid_surface_indices:
            mask = self.atom37_mask[idx]
            coords = self.atom37_positions[idx][mask]
            if len(coords) > 0:  # Ensure there are atoms
                all_atoms.append(coords)
                all_res_indices.extend([idx] * len(coords))
        
        if not all_atoms:  # No atoms to process
            return {idx: ([], [], 0.0, 0.0) for idx in valid_surface_indices}, None, None
        
        all_atoms = np.vstack(all_atoms).astype(np.float32)  # shape: (total_atoms, 3)
        all_res_indices = np.array(all_res_indices)
        
        # Collect C-alpha coordinates of surface residues
        surface_ca = []
        valid_center_indices = []
        ca_idx = RC.atom_order["CA"]  # Get CA atom index from atom order
        
        for idx in valid_surface_indices:
            # Get CA coordinates from atom37_positions
            ca_coord = self.atom37_positions[idx, ca_idx, :]
            if not np.any(np.isnan(ca_coord)) and self.atom37_mask[idx, ca_idx]:  # Ensure CA atom coordinates are valid and atom exists
                surface_ca.append(ca_coord)
                valid_center_indices.append(idx)
        
        if not surface_ca:  # No valid CA atoms
            return {}, None, None
        
        surface_ca = np.array(surface_ca, dtype=np.float32)
        surface_ca = surface_ca.reshape(-1, 3)  # Ensure shape is (n_residues, 3)
        
        # Compute distance matrix between each C-alpha and all atoms
        try:
            dist_matrix = cdist(surface_ca, all_atoms)
        except ValueError as e:
            print(f"Error in distance calculation: {e}")
            print(f"surface_ca shape: {surface_ca.shape}")
            print(f"all_atoms shape: {all_atoms.shape}")
            return {}, None, None
        
        max_recall = -1
        max_recall_res = None
        max_precision = -1
        max_precision_res = None

        coverage = {}
        epitope_indices = np.where(self.epitopes)[0]  # Get epitope indices directly
        if len(epitope_indices) == 0:
            print(f"No epitopes records for protein {self.id}_{self.chain_id}")
        
        for i, center_idx in enumerate(valid_center_indices):
            within_radius = dist_matrix[i] < radius
            covered_indices = np.unique(all_res_indices[within_radius])
            covered_indices_list = covered_indices.tolist()
            
            # Find intersection with epitopes (using indices)
            covered_epitope_indices = list(set(covered_indices_list).intersection(set(epitope_indices)))

            # Calculate precision and recall
            precision = len(covered_epitope_indices) / len(covered_indices_list) if covered_indices_list else 0.0
            recall = len(covered_epitope_indices) / len(epitope_indices) if len(epitope_indices) > 0 else 0.0

            if recall > max_recall:
                max_recall = recall
                max_recall_res = center_idx
            if precision > max_precision:
                max_precision = precision
                max_precision_res = center_idx
            
            # Convert to native Python types for JSON compatibility
            coverage[int(center_idx)] = (
                [int(idx) for idx in covered_indices_list],
                [int(idx) for idx in covered_epitope_indices], 
                float(precision),
                float(recall)
            )

        return coverage, max_recall_res, max_precision_res
    
    def get_surface_coverage(self, radius: float = 18, 
                             threshold: float = 0.25, 
                             index: bool = True,
                             override: bool = False) -> tuple: 
        """
        Retrieve (or compute and cache) the coverage mapping for surface residues.
        For each surface residue, using its C_alpha as the sphere center (with radius `radius`),
        determine which surface residues are covered (i.e. if any atom falls within that sphere).
        The result is cached to an HDF5 file for faster subsequent retrieval.
        
        The cache file is saved in BASE_DIR / "data/antigen_sphere", with the file name
        "{self.id}_{self.chain_id}.h5", and radius as the first-level key.
        
        Args:
            radius (float): The radius of the sphere (in Ångstroms).
            threshold (float): Fraction of maximum ASA to define a residue as surface-exposed.
            index (bool): If True, return indices instead of residue numbers for easier embeddings/coords access.
            override (bool): If True, recompute even if cache exists.
        
        Returns:
            tuple:
                - coverage (dict): A dictionary mapping each surface residue to a tuple of:
                    If index=True: center_index -> (list[int]): List of covered residue indices
                                                 (list[int]): List of covered epitope residue indices  
                                                 (float): Precision
                                                 (float): Recall
                    If index=False: center_residue_num -> (list[int]): List of covered residue numbers
                                                         (list[int]): List of covered epitope residue numbers  
                                                         (float): Precision
                                                         (float): Recall
                - max_recall_res (int): The surface residue number with the highest recall.
                - max_precision_res (int): The surface residue number with the highest precision.
        """
        # Define the cache directory and file
        cache_dir = BASE_DIR / "data" / "antigen_sphere"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_filename = f"{self.id}_{self.chain_id}.h5"
        cache_path = cache_dir / cache_filename
        radius_key = f"r{radius}"
        
        # If the cache file exists and the radius key exists, load and return the cached result.
        if cache_path.exists() and not override:
            try:
                with h5py.File(cache_path, "r") as h5f:
                    if radius_key in h5f:
                        # Load cached data for this radius
                        radius_group = h5f[radius_key]
                        
                        if index:
                            # Cache stores indices, so directly use them
                            coverage = {}
                            for center_idx_str in radius_group.keys():
                                center_idx = int(center_idx_str)
                                center_group = radius_group[center_idx_str]
                                covered_indices = center_group['covered_indices'][:].tolist()
                                covered_epitope_indices = center_group['covered_epitope_indices'][:].tolist()
                                precision = float(center_group.attrs['precision'])
                                recall = float(center_group.attrs['recall'])
                                coverage[center_idx] = (covered_indices, covered_epitope_indices, precision, recall)
                            return coverage, None, None 
                        else:
                            # Convert indices to residue numbers
                            coverage = {}
                            max_recall = -1
                            max_recall_res = None
                            max_precision = -1
                            max_precision_res = None
                            
                            for center_idx_str in radius_group.keys():
                                center_idx = int(center_idx_str)
                                center_res_num = int(self.residue_index[center_idx])
                                center_group = radius_group[center_idx_str]
                                
                                covered_indices = center_group['covered_indices'][:].tolist()
                                covered_epitope_indices = center_group['covered_epitope_indices'][:].tolist()
                                precision = float(center_group.attrs['precision'])
                                recall = float(center_group.attrs['recall'])
                                
                                # Convert covered indices to residue numbers
                                covered_res_nums = [int(self.residue_index[idx]) for idx in covered_indices if 0 <= idx < len(self.residue_index)]
                                covered_epitope_res_nums = [int(self.residue_index[idx]) for idx in covered_epitope_indices if 0 <= idx < len(self.residue_index)]
                                
                                coverage[center_res_num] = (covered_res_nums, covered_epitope_res_nums, precision, recall)
                                
                                if recall > max_recall:
                                    max_recall = recall
                                    max_recall_res = center_res_num
                                if precision > max_precision:
                                    max_precision = precision
                                    max_precision_res = center_res_num

                            return coverage, max_recall_res, max_precision_res
            except (OSError, KeyError, ValueError) as e:
                print(f"[WARNING] Error reading cache file {cache_path}: {e}")
                print(f"[INFO] Recomputing surface coverage...")
        
        # Otherwise, compute the coverage mapping (returns index-based results)
        coverage, max_recall_res, max_precision_res = self._scan_surface_residues(radius, threshold)
        
        # Save the result to HDF5 file
        # Create or open the HDF5 file and save data under the radius key
        with h5py.File(cache_path, "a") as h5f:  # "a" mode: read/write if exists, create otherwise
            # Create or overwrite the radius group
            if radius_key in h5f:
                del h5f[radius_key]  # Remove existing group if override or recompute
            
            radius_group = h5f.create_group(radius_key)
            
            # Save each center residue's data
            for center_idx, (covered_indices, covered_epitope_indices, precision, recall) in coverage.items():
                center_group = radius_group.create_group(str(center_idx))
                center_group.create_dataset('covered_indices', data=np.array(covered_indices, dtype=np.int32), compression='gzip')
                center_group.create_dataset('covered_epitope_indices', data=np.array(covered_epitope_indices, dtype=np.int32), compression='gzip')
                center_group.attrs['precision'] = precision
                center_group.attrs['recall'] = recall
        
        # Convert to residue numbers if index=False is requested
        if not index:
            coverage_resnums = {}
            max_recall_res_num = None
            max_precision_res_num = None
            
            if max_recall_res is not None:
                max_recall_res_num = int(self.residue_index[max_recall_res])
            if max_precision_res is not None:
                max_precision_res_num = int(self.residue_index[max_precision_res])
            
            for center_idx, (covered_indices, covered_epitope_indices, precision, recall) in coverage.items():
                center_res_num = int(self.residue_index[center_idx])
                # Convert covered indices to residue numbers
                covered_res_nums = [int(self.residue_index[idx]) for idx in covered_indices if 0 <= idx < len(self.residue_index)]
                covered_epitope_res_nums = [int(self.residue_index[idx]) for idx in covered_epitope_indices if 0 <= idx < len(self.residue_index)]
                coverage_resnums[center_res_num] = (covered_res_nums, covered_epitope_res_nums, precision, recall)
            
            return coverage_resnums, max_recall_res_num, max_precision_res_num
        
        return coverage, max_recall_res, max_precision_res
        
    def data_preparation(self, radius: float = None, encoder: str = "esmc", override: bool = False):
        """
        Retrieve or compute region embeddings for surface residues using spherical regions.

        Args:
            radius (float): Radius to define the neighborhood of each center residue.
            threshold (float): Threshold to determine surface residues.
            cover (bool): Whether to recompute and overwrite cached data.
            verbose (bool): Whether to print progress information.

        Returns:
            tuple:
                - embeddings (np.ndarray): Array of embeddings mean of the region. (num_regions, embedding_dim)
                - center_residues (np.ndarray): Array of center residue numbers. (num_regions,)
                - precisions (np.ndarray): Array of precision values for each center residue. (num_regions,)
                - recalls (np.ndarray): Array of recall values for each center residue. (num_regions,)
        """
        embeddings = self.get_embeddings(encoder=encoder)
        backbone_atoms = self.get_backbone_atoms()
        rsa = self.get_rsa()
        if radius is None:
            # Used for creating data
            for i in range(16,21,2):
                _, _, _ = self.get_surface_coverage(radius=i, override=override)
            return embeddings, backbone_atoms, rsa, None
        else:
            coverage_dict, _, _ = self.get_surface_coverage(radius=radius, override=override)
            return embeddings, backbone_atoms, rsa, coverage_dict
    
    def evaluate(self, model_path: str = None, device_id: int = 1, radius: float = 19.0, k: int = 7, 
                threshold: float = None, verbose: bool = True, encoder: str = "esmc", use_gpu: bool = True,
                find_optimal_threshold: bool = False, include_curves: bool = False):
        """
        Evaluate epitopes using RoBep model with spherical regions.
        
        Args:
            model_path (str): Path to the trained RoBep model
            device_id (int): GPU device ID to use
            radius (float): Radius for spherical regions
            k (int): Number of top regions to select
            threshold (float): Threshold for node-level epitope prediction
            verbose (bool): Whether to print progress information
            encoder (str): Encoder type for embeddings
            use_gpu (bool): Whether to use GPU for prediction
            find_optimal_threshold (bool): Whether to find optimal threshold using F1 score
            include_curves (bool): Whether to include PR and ROC curves in results
            
        Returns:
            dict: Dictionary containing comprehensive evaluation metrics:
                - Prediction results: 'predicted_epitopes', 'voted_epitopes', 'true_epitopes'
                - Legacy metrics: 'predicted_precision', 'predicted_recall', 'predicted_f1'
                - Comprehensive metrics: 'precision', 'recall', 'f1', 'mcc', 'agiou', 'accuracy'
                - Ranking metrics: 'auroc', 'auroc0-1', 'auprc'
                - Confusion matrix: 'true_positives', 'false_positives', 'true_negatives', 'false_negatives'
                - Model info: 'top_k_regions', 'residue_votes', 'predictions'
                - Full metrics dict: 'node_metrics' (contains all metrics from calculate_node_metrics)
        """
        # Set device
        if use_gpu and torch.cuda.is_available() and device_id >= 0:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        if verbose:
            print(f"[INFO] Using device: {device}")
            
        if sum(self.epitopes) == 0:
            if verbose:
                print("[WARNING] No epitopes recorded, please run annotate_epitopes() first")
        
        # Load RoBep model
        try:
            if model_path is None:
                model_path = f"{BASE_DIR}/models/RoBep/20250626_110438/best_mcc_model.bin"
                
            if threshold is None:
                model, threshold = RoBep.load(model_path, device=device, strict=False, verbose=False)
            else:
                model, _ = RoBep.load(model_path, device=device, strict=False, verbose=False)
                
            model.eval()
            if verbose:
                print(f"[INFO] Loaded RoBep model from {model_path}")
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to load model: {str(e)}")
            return {}
        
        # Get protein data using data_preparation
        try:
            embeddings, backbone_atoms, rsa, coverage_dict = self.data_preparation(radius=radius, encoder=encoder)
            if verbose:
                print(f"[INFO] Retrieved protein data for {len(coverage_dict)} surface regions")
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to prepare data: {str(e)}")
                traceback.print_exc()
            return {}
        
        if not coverage_dict:
            if verbose:
                print("[WARNING] No surface regions found")
            return {}
        
        # Get epitope indices
        epitope_indices = np.where(self.epitopes)[0].tolist()
        
        # Phase 1: Predict graph-level values for all regions
        region_predictions = []
        
        with torch.no_grad():
            for center_idx, (covered_indices, covered_epitope_indices, precision, recall) in tqdm(
                coverage_dict.items(), desc="Predicting region values", disable=not verbose):
                
                if len(covered_indices) < 2:  # Skip regions with too few residues
                    continue
                
                try:
                    # Create graph data for this region
                    graph_data = create_graph_data(
                        center_idx=center_idx,
                        covered_indices=covered_indices,
                        covered_epitope_indices=covered_epitope_indices,
                        embeddings=embeddings,
                        backbone_atoms=backbone_atoms,
                        rsa_values=rsa,
                        epitope_indices=epitope_indices,
                        recall=recall,
                        precision=precision,
                        pdb_id=self.id,
                        chain_id=self.chain_id,
                        verbose=True  # Enable verbose to see errors
                    )
                    
                    if graph_data is None:
                        if verbose:
                            print(f"[WARNING] Failed to create graph data for region {center_idx}")
                        continue
                    
                    # Move data to device
                    graph_data = graph_data.to(device)
                    
                    # Create batch tensor for single graph - this is crucial!
                    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
                    
                    # Predict using RoBep model (following trainer.py pattern)
                    outputs = model(graph_data)
                    
                    # Get graph-level prediction
                    if 'global_pred' in outputs:
                        graph_pred = torch.sigmoid(outputs['global_pred']).cpu().item()
                    else:
                        # Fallback: use mean of node predictions as graph prediction
                        node_preds = torch.sigmoid(outputs['node_preds']).cpu().numpy()
                        graph_pred = float(np.mean(node_preds))
                    
                    region_predictions.append({
                        'center_idx': center_idx,
                        'covered_indices': covered_indices,
                        'covered_epitope_indices': covered_epitope_indices,
                        'graph_pred': graph_pred,
                        'true_recall': recall,
                        'graph_data': graph_data
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"[WARNING] Error processing region {center_idx}: {str(e)}")          
                        traceback.print_exc()
                    continue
        
        if not region_predictions:
            if verbose:
                print("[WARNING] No valid region predictions")
            return {}
        
        # Phase 2: Select top-k regions based on graph predictions
        region_predictions.sort(key=lambda x: x['graph_pred'], reverse=True)
        top_k_regions = region_predictions[:k]
        
        if verbose:
            print(f"[INFO] Selected top {len(top_k_regions)} regions:")
            for i, region in enumerate(top_k_regions):
                print(f"  Region {i+1}: center={region['center_idx']}, "
                      f"predicted_value={region['graph_pred']:.3f}, "
                      f"true_recall={region['true_recall']:.3f}")
        
        # Phase 3: Predict node-level epitopes for selected regions
        residue_votes = {}  # residue_idx -> [list of binary predictions]
        residue_probs = {}  # residue_idx -> [list of probabilities]
        
        with torch.no_grad():
            for region in tqdm(top_k_regions, desc="Predicting node values", disable=not verbose):
                try:
                    graph_data = region['graph_data']
                    
                    # Ensure graph data has batch information - this is crucial!
                    if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                        graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
                    
                    # Predict using RoBep model (following trainer.py pattern)
                    outputs = model(graph_data)
                    
                    # Get node-level predictions
                    node_preds = torch.sigmoid(outputs['node_preds']).cpu().numpy()
                    
                    # Store votes and probabilities for each residue
                    for local_idx, residue_idx in enumerate(region['covered_indices']):
                        if residue_idx not in residue_votes:
                            residue_votes[residue_idx] = []
                            residue_probs[residue_idx] = []
                        
                        # Store probability and binary vote
                        prob = float(node_preds[local_idx])
                        residue_probs[residue_idx].append(prob)
                        
                        # Binary vote based on threshold
                        vote = 1 if prob >= threshold else 0
                        residue_votes[residue_idx].append(vote)
                        
                except Exception as e:
                    if verbose:
                        print(f"[WARNING] Error in node prediction for region {region['center_idx']}: {str(e)}")
                        traceback.print_exc()
                    continue
        
        # Create predictions dictionary for all residues
        all_residue_predictions = {}
        for idx in range(len(self.residue_index)):
            residue_num = int(self.residue_index[idx])
            if idx in residue_probs:
                # Calculate mean probability for residues in top-k regions
                all_residue_predictions[residue_num] = float(np.mean(residue_probs[idx]))
            else:
                # Set probability to 1e-5 for residues not in any top-k region
                all_residue_predictions[residue_num] = 1e-2
        
        # Phase 4a: Apply voting mechanism for voted_epitopes
        voted_epitope_indices = []
        for residue_idx, votes in residue_votes.items():
            # If >= half of the votes are positive, predict as epitope
            if sum(votes) >= len(votes) / 2:
                voted_epitope_indices.append(residue_idx)
        
        # Convert indices to residue numbers for voted epitopes
        voted_epitope_resnums = [int(self.residue_index[idx]) for idx in voted_epitope_indices 
                               if 0 <= idx < len(self.residue_index)]
        
        # Phase 4b: Apply probability threshold for predicted_epitopes
        predicted_epitope_resnums = []
        for residue_num, prob in all_residue_predictions.items():
            if prob >= threshold:
                predicted_epitope_resnums.append(residue_num)
        
        # Get true epitopes
        true_epitope_resnums = set(self.get_epitope_residue_numbers())
        
        # Calculate metrics for both prediction methods
        # Metrics for voted epitopes
        voted_tp = len(set(voted_epitope_resnums) & true_epitope_resnums)
        voted_precision = voted_tp / len(voted_epitope_resnums) if voted_epitope_resnums else 0
        voted_recall = voted_tp / len(true_epitope_resnums) if true_epitope_resnums else 0
        
        # Calculate comprehensive metrics using metrics.py
        from ..utils.metrics import calculate_node_metrics
        
        # Prepare data for metrics calculation
        # Create arrays for all residues in the sequence
        all_predictions = np.zeros(len(self.sequence))
        all_true_labels = np.zeros(len(self.sequence))
        
        # Fill in the predictions and true labels
        for idx in range(len(self.residue_index)):
            residue_num = int(self.residue_index[idx])
            all_predictions[idx] = all_residue_predictions.get(residue_num, 1e-2)
            all_true_labels[idx] = 1 if residue_num in true_epitope_resnums else 0
        
        # Calculate comprehensive node-level metrics
        node_metrics = calculate_node_metrics(
            preds=all_predictions,
            labels=all_true_labels,
            find_threshold=find_optimal_threshold,
            include_curves=include_curves,
            threshold_metric='f1',  # Use F1 score for threshold optimization
            threshold=threshold
        )
        
        # Legacy metrics for backward compatibility  
        predicted_tp = len(set(predicted_epitope_resnums) & true_epitope_resnums)
        predicted_precision = predicted_tp / len(predicted_epitope_resnums) if predicted_epitope_resnums else 0
        predicted_recall = predicted_tp / len(true_epitope_resnums) if true_epitope_resnums else 0
        predicted_f1 = 2 * predicted_precision * predicted_recall / (predicted_precision + predicted_recall + 1e-10)
        
        if verbose:
            print(f"\n[INFO] Final Results:")
            print(f"  True epitopes: {len(true_epitope_resnums)}")
            print(f"  Residues in top-k regions: {len(residue_probs)}/{len(self.residue_index)}")
            print(f"\n  Voting-based prediction:")
            print(f"    Voted epitopes: {len(voted_epitope_resnums)}")
            print(f"    Voted precision: {voted_precision:.3f}")
            print(f"    Voted recall: {voted_recall:.3f}")
            threshold_info = f"threshold={node_metrics['threshold_used']:.3f}"
            if find_optimal_threshold:
                threshold_info += f" (optimal F1: {node_metrics.get('best_f1', 0):.3f})"
            print(f"\n  Comprehensive Node-Level Metrics ({threshold_info}):")
            print(f"    Predicted epitopes: {len(predicted_epitope_resnums)}")
            print(f"    Precision: {node_metrics['precision']:.3f}")
            print(f"    Recall: {node_metrics['recall']:.3f}")
            print(f"    F1-score: {node_metrics['f1']:.3f}")
            print(f"    MCC: {node_metrics['mcc']:.3f}")
            print(f"    AgIoU: {node_metrics['agiou']:.3f}")
            print(f"    Accuracy: {node_metrics['accuracy']:.3f}")
            print(f"    AUROC: {node_metrics['auroc']:.3f}")
            print(f"    AUROC 0-1: {node_metrics['auroc0-1']:.3f}")
            print(f"    AUPRC: {node_metrics['auprc']:.3f}")
            print(f"    TP: {node_metrics['true_positives']}, FP: {node_metrics['false_positives']}")
            print(f"    TN: {node_metrics['true_negatives']}, FN: {node_metrics['false_negatives']}")
            if find_optimal_threshold and 'best_threshold' in node_metrics:
                print(f"    Optimal threshold: {node_metrics['best_threshold']:.3f}")
        return {
            # Prediction results
            'predicted_epitopes': predicted_epitope_resnums,  # Based on probability threshold
            'voted_epitopes': voted_epitope_resnums,          # Based on voting mechanism
            'true_epitopes': true_epitope_resnums,
            'predictions': all_residue_predictions,           # All residue probabilities
            
            # Legacy metrics (backward compatibility)
            'predicted_precision': predicted_precision,       # Precision for probability-based
            'predicted_recall': predicted_recall,             # Recall for probability-based
            'predicted_f1': predicted_f1,                     # F1-score for probability-based
            'voted_precision': voted_precision,               # Precision for voting-based
            'voted_recall': voted_recall,                     # Recall for voting-based
            
            # Comprehensive node-level metrics from metrics.py
            'node_metrics': node_metrics,                     # Complete metrics dictionary
            'precision': node_metrics['precision'],           # Main precision metric
            'recall': node_metrics['recall'],                 # Main recall metric
            'f1': node_metrics['f1'],                         # Main F1 score
            'mcc': node_metrics['mcc'],                       # Matthews Correlation Coefficient
            'agiou': node_metrics['agiou'],                   # Antigen Intersection over Union
            'accuracy': node_metrics['accuracy'],             # Accuracy
            'auroc': node_metrics['auroc'],                   # Area under ROC curve
            'auroc0-1': node_metrics['auroc0-1'],             # AUROC in low FPR range [0, 0.1]
            'auprc': node_metrics['auprc'],                   # Area under Precision-Recall curve
            'threshold_used': node_metrics['threshold_used'], # Threshold used for metrics
            
            # Confusion matrix components
            'true_positives': node_metrics['true_positives'],
            'false_positives': node_metrics['false_positives'],
            'true_negatives': node_metrics['true_negatives'],
            'false_negatives': node_metrics['false_negatives'],
            
            # Optional curves data (only if include_curves=True)
            'pr_curve': node_metrics.get('pr_curve'),         # Precision-Recall curve
            'roc_curve': node_metrics.get('roc_curve'),       # ROC curve
            
            # Model information
            'top_k_regions': [
                {
                    'center_residue': int(self.residue_index[region['center_idx']]),
                    'center_idx': region['center_idx'],
                    'predicted_value': region['graph_pred'],
                    'true_recall': region['true_recall'],
                    'covered_residues': [int(self.residue_index[idx]) for idx in region['covered_indices']],
                    'radius': radius
                }
                for region in top_k_regions
            ],
            'residue_votes': {
                int(self.residue_index[idx]): votes 
                for idx, votes in residue_votes.items() 
                if 0 <= idx < len(self.residue_index)
            }
        }
        
    def predict(self, model_path: str = None, device_id: int = 1, radius: float = 19.0, k: int = 7, 
                threshold: float = None, verbose: bool = True, encoder: str = "esmc", use_gpu: bool = True):
        """
        Predict epitopes using RoBep model with spherical regions (for unknown true epitopes).
        
        Args:
            model_path (str): Path to the trained RoBep model
            device_id (int): GPU device ID to use
            radius (float): Radius for spherical regions
            k (int): Number of top regions to select
            threshold (float): Threshold for node-level epitope prediction
            verbose (bool): Whether to print progress information
            encoder (str): Encoder type for embeddings
            
        Returns:
            dict: Dictionary containing:
                - 'predicted_epitopes': List of predicted epitope residue numbers
                - 'predictions': Dictionary of all residue probabilities {resnum: probability}
                - 'top_k_centers': List of top-k center residue numbers
                - 'top_k_region_residues': List of all residues covered by top-k regions (union)
                - 'top_k_regions': Detailed information about selected regions
        """
        # Set device
        if use_gpu and torch.cuda.is_available() and device_id >= 0:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        if verbose:
            print(f"[INFO] Using device: {device}")
        
        # Load RoBep model
        try:
            if model_path is None:
                model_path = f"{BASE_DIR}/models/RoBep/20250626_110438/best_mcc_model.bin"
                
            if threshold is None:
                model, threshold = RoBep.load(model_path, device=device, strict=False, verbose=False)
            else:
                model, _ = RoBep.load(model_path, device=device, strict=False, verbose=False)
                
            model.eval()
            if verbose:
                print(f"[INFO] Loaded RoBep model from {model_path}")
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to load model: {str(e)}")
            return {}
        
        # Get protein data using data_preparation
        try:
            embeddings, backbone_atoms, rsa, coverage_dict = self.data_preparation(radius=radius, encoder=encoder)
            if verbose:
                print(f"[INFO] Retrieved protein data for {len(coverage_dict)} surface regions")
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to prepare data: {str(e)}")
                traceback.print_exc()
            return {}
        
        if not coverage_dict:
            if verbose:
                print("[WARNING] No surface regions found")
            return {}
        
        # Phase 1: Predict graph-level values for all regions
        region_predictions = []
        
        with torch.no_grad():
            for center_idx, (covered_indices, covered_epitope_indices, precision, recall) in tqdm(
                coverage_dict.items(), desc="Predicting region values", disable=not verbose):
                
                if len(covered_indices) < 2:  # Skip regions with too few residues
                    continue
                
                try:
                    # Create graph data for this region (without epitope information)
                    graph_data = create_graph_data(
                        center_idx=center_idx,
                        covered_indices=covered_indices,
                        covered_epitope_indices=[],  # No epitope information for prediction
                        embeddings=embeddings,
                        backbone_atoms=backbone_atoms,
                        rsa_values=rsa,
                        epitope_indices=[],  # No epitope information for prediction
                        recall=0.0,  # No recall information
                        precision=0.0,  # No precision information
                        pdb_id=self.id,
                        chain_id=self.chain_id,
                        verbose=False
                    )
                    
                    if graph_data is None:
                        if verbose:
                            print(f"[WARNING] Failed to create graph data for region {center_idx}")
                        continue
                    
                    # Move data to device
                    graph_data = graph_data.to(device)
                    
                    # Create batch tensor for single graph
                    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
                    
                    # Predict using RoBep model
                    outputs = model(graph_data)
                    
                    # Get graph-level prediction
                    if 'global_pred' in outputs:
                        graph_pred = torch.sigmoid(outputs['global_pred']).cpu().item()
                    else:
                        # Fallback: use mean of node predictions as graph prediction
                        node_preds = torch.sigmoid(outputs['node_preds']).cpu().numpy()
                        graph_pred = float(np.mean(node_preds))
                    
                    region_predictions.append({
                        'center_idx': center_idx,
                        'covered_indices': covered_indices,
                        'graph_pred': graph_pred,
                        'graph_data': graph_data
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"[WARNING] Error processing region {center_idx}: {str(e)}")          
                        traceback.print_exc()
                    continue
        
        if not region_predictions:
            if verbose:
                print("[WARNING] No valid region predictions")
            return {}
        
        # Phase 2: Select top-k regions based on graph predictions
        region_predictions.sort(key=lambda x: x['graph_pred'], reverse=True)
        top_k_regions = region_predictions[:k]
        
        if verbose:
            print(f"[INFO] Selected top {len(top_k_regions)} regions:")
            for i, region in enumerate(top_k_regions):
                print(f"  Region {i+1}: center={region['center_idx']}, "
                      f"predicted_value={region['graph_pred']:.3f}")
        
        # Phase 3: Predict node-level epitopes for selected regions
        residue_probs = {}  # residue_idx -> [list of probabilities]
        
        with torch.no_grad():
            for region in tqdm(top_k_regions, desc="Predicting node values", disable=not verbose):
                try:
                    graph_data = region['graph_data']
                    
                    # Ensure graph data has batch information
                    if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                        graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
                    
                    # Predict using RoBep model
                    outputs = model(graph_data)
                    
                    # Get node-level predictions
                    node_preds = torch.sigmoid(outputs['node_preds']).cpu().numpy()
                    
                    # Store probabilities for each residue
                    for local_idx, residue_idx in enumerate(region['covered_indices']):
                        if residue_idx not in residue_probs:
                            residue_probs[residue_idx] = []
                        
                        # Store probability
                        prob = float(node_preds[local_idx])
                        residue_probs[residue_idx].append(prob)
                        
                except Exception as e:
                    if verbose:
                        print(f"[WARNING] Error in node prediction for region {region['center_idx']}: {str(e)}")
                        traceback.print_exc()
                    continue
        
        # Create predictions dictionary for all residues
        all_residue_predictions = {}
        for idx in range(len(self.residue_index)):
            residue_num = int(self.residue_index[idx])
            if idx in residue_probs:
                # Calculate mean probability for residues in top-k regions
                all_residue_predictions[residue_num] = float(np.mean(residue_probs[idx]))
            else:
                # Set probability to 0 for residues not in any top-k region
                all_residue_predictions[residue_num] = 0.0
        
        # Apply probability threshold for predicted epitopes
        predicted_epitope_resnums = []
        node_mean = 0.0
        for residue_num, prob in all_residue_predictions.items():
            node_mean += prob
            if prob >= threshold:
                predicted_epitope_resnums.append(residue_num)
        node_mean /= len(all_residue_predictions) if all_residue_predictions else 1
        
        # Get top-k center residue numbers
        top_k_centers = [int(self.residue_index[region['center_idx']]) for region in top_k_regions]
        
        # Get union of all residues covered by top-k regions and mean graph predicted value
        graph_mean = 0.0
        all_covered_indices = set()
        for region in top_k_regions:
            all_covered_indices.update(region['covered_indices'])
            graph_mean += region['graph_pred']
        graph_mean /= len(top_k_regions)
        
        top_k_region_residues = [int(self.residue_index[idx]) for idx in all_covered_indices 
                               if 0 <= idx < len(self.residue_index)]
        
        if verbose:
            print(f"\n[INFO] Prediction Results:")
            print(f"  Predicted epitopes: {len(predicted_epitope_resnums)}")
            print(f"  Top-k centers: {top_k_centers}")
            print(f"  Total residues in top-k regions: {len(top_k_region_residues)}")
        
        return {
            'predicted_epitopes': predicted_epitope_resnums,
            'predictions': all_residue_predictions,
            'top_k_centers': top_k_centers,
            'top_k_region_residues': top_k_region_residues,
            'top_k_regions': [
                {
                    'center_residue': int(self.residue_index[region['center_idx']]),
                    'center_idx': region['center_idx'],
                    'predicted_value': region['graph_pred'],
                    'covered_residues': [int(self.residue_index[idx]) for idx in region['covered_indices']],
                    'radius': radius
                }
                for region in top_k_regions
            ],
            'antigen_rate': graph_mean,
            'epitope_rate': node_mean
        }
    
    def visualize(self, 
                  mode: str = 'normal',
                  style: str = 'cartoon',
                  predicted_epitopes: list = None,
                  predict_results: dict = None,
                  prediction_mode: str = 'residue',  # 'residue' or 'region'
                  center_res: int = None,
                  radius: float = None,
                  region_index: int = None,  # Index of specific region to show (0-based), deprecated
                  region_indices: list = None,  # List of region indices to show (0-based), e.g. [0, 2, 3]
                  width: int = 800,
                  height: int = 600,
                  base_color: str = '#e6e6f7',
                  true_epitope_color: str = '#f1b54c',  # True epitopes (deeper blue)
                  false_positive_color: str = '#ef5331',  # False positives (deeper red)
                  true_positive_color: str = '#a0d293',  # True positives (deeper green)
                  coverage_color: str = '#9C6ADE',  # Coverage regions (purple)
                  prediction_color: str = '#9C6ADE',  # Prediction color (purple)
                  center_color: str = '#2C3E50',  # Center residue (dark gray)
                  probability_colormap: str = 'RdYlBu_r',  # Colormap for probability visualization
                  show_surface: bool = True,
                  show_shape: bool = True,
                  show_center: bool = True,
                  center_radius: float = 0.7,
                  n_points: int = 50,
                  shape_opacity: float = 0.3,
                  surface_opacity: float = 1.0,
                  wireframe: bool = True,
                  show_epitope: bool = True,
                  show_coverage: bool = True,
                  show_top_regions: bool = True,
                  max_spheres: int = None,  # Maximum number of spheres to show
                  prob_threshold: float = 0.5):
        """
        Visualize the protein chain with various modes and integration with predict results.
        
        Args:
            mode (str): Visualization mode. Options:
                - 'normal': Basic protein structure
                - 'epitope': Show predicted epitopes vs true epitopes
                - 'coverage': Show spherical coverage region
                - 'evaluation': Show evaluation results from evaluate() function (supports region_indices filtering)
                - 'prediction': Show prediction results from predict() function
                - 'probability': Show residue probabilities as color gradient
                - 'top_regions': Show top-k regions from prediction
                - 'comparison': Compare voted vs predicted epitopes
            prediction_mode (str): Sub-mode for prediction visualization ('residue' or 'region')
                - 'residue': Color predicted epitopes by probability (gradient purple)
                - 'region': Color all residues in top-k regions uniformly
            style (str): Protein representation style ('cartoon', 'stick', 'sphere', 'surface')
            predicted_epitopes (list): List of predicted epitope residue numbers
            predict_results (dict): Results dictionary from predict() function
            center_res (int): Center residue number for coverage visualization
            radius (float): Radius for spherical coverage
            region_index (int): Index of specific region to show (0-based), deprecated
                              Use region_indices instead for better flexibility  
            region_indices (list): List of region indices to show (0-based), e.g. [0, 2, 3]
                                  Supported in 'prediction' and 'evaluation' modes
                                  Each region uses a distinct color for shape visualization
            probability_colormap (str): Colormap name for probability visualization
            prob_threshold (float): Threshold for probability-based coloring
            ... (other parameters as before)
            
        Returns:
            py3Dmol.view: The molecular visualization view object
        """
        # Create view object and add basic structure
        view = self._create_base_view(width, height)
        
        # Set basic style
        style_dict = {
            'cartoon': {'cartoon': {}},
            'stick': {'stick': {}},
            'sphere': {'sphere': {}},
            'surface': {'surface': {}}
        }
        base_style = style_dict.get(style, {'cartoon': {}})
        
        # Visualization based on mode
        if mode == 'epitope' and predicted_epitopes is not None:
            self._add_epitope_visualization(
                view, style, predicted_epitopes,
                base_color, true_epitope_color, false_positive_color, 
                true_positive_color, coverage_color,
                show_surface, surface_opacity, show_coverage,
                center_res, radius
            )
            
            # Add shape visualization if needed
            if show_shape and center_res is not None and radius is not None:
                self._add_shape_visualization(
                    view, center_res, radius,
                    coverage_color, center_color,
                    show_center, center_radius,
                    shape_opacity, wireframe
                )
                
        elif mode == 'coverage' and center_res is not None and radius is not None:
            self._add_coverage_visualization(
                view, style, center_res, radius,
                base_color, coverage_color, true_positive_color, true_epitope_color,
                show_surface, show_shape, show_center,
                surface_opacity, shape_opacity, center_radius,
                n_points, center_color, wireframe, show_epitope
            )
            
        elif mode == 'evaluation' and predict_results is not None:
            # Handle backward compatibility: convert single region_index to list
            target_region_indices = None
            if region_indices is not None:
                target_region_indices = region_indices
            elif region_index is not None:
                target_region_indices = [region_index]
                
            self._add_evaluation_visualization(
                view, style, predict_results,
                base_color, true_epitope_color, false_positive_color, 
                true_positive_color, coverage_color,
                show_surface, surface_opacity, show_shape, radius, max_spheres,
                target_region_indices, shape_opacity, show_center, center_radius, 
                wireframe, center_color
            )
            
        elif mode == 'prediction' and predict_results is not None:
            # Handle backward compatibility: convert single region_index to list
            target_region_indices = None
            if region_indices is not None:
                target_region_indices = region_indices
            elif region_index is not None:
                target_region_indices = [region_index]
            
            self._add_probability_visualization(
                view, style, predict_results,
                base_color, probability_colormap, show_surface, surface_opacity, 
                prob_threshold, target_region_indices, radius, show_shape, shape_opacity, 
                show_center, center_radius, wireframe, coverage_color, center_color
            )
            
        elif mode == 'top_regions' and predict_results is not None:
            self._add_top_regions_visualization(
                view, style, predict_results,
                base_color, coverage_color, center_color,
                show_surface, show_shape, show_center,
                surface_opacity, shape_opacity, center_radius,
                wireframe, radius, max_spheres
            )
            
        elif mode == 'comparison' and predict_results is not None:
            self._add_comparison_visualization(
                view, style, predict_results,
                base_color, true_epitope_color, false_positive_color, 
                true_positive_color, coverage_color, show_surface, surface_opacity
            )
            
        else:
            # Default mode: just show the basic structure
            view.setStyle({'chain': self.chain_id}, base_style)
        
        # Adjust view
        view.zoomTo()
        return view
    
    def _add_prediction_visualization(self, view, style, predict_results, prediction_mode,
                                  base_color, prediction_color, show_surface, surface_opacity,
                                  show_shape, shape_opacity, show_center, center_radius,
                                  wireframe, radius, max_spheres):
        """Add visualization for prediction results"""
        if prediction_mode == 'residue':
            self._add_prediction_residue_mode(
                view, style, predict_results, base_color, prediction_color, 
                show_surface, surface_opacity
            )
        elif prediction_mode == 'region':
            self._add_prediction_region_mode(
                view, style, predict_results, base_color, prediction_color,
                show_surface, surface_opacity, show_shape, shape_opacity,
                show_center, center_radius, wireframe, radius, max_spheres
            )
    
    def _add_prediction_residue_mode(self, view, style, predict_results, base_color, prediction_color, 
                                 show_surface, surface_opacity):
        """Add visualization for prediction results in residue mode"""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Get predictions dictionary
        predictions = predict_results.get('predictions', {})
        predicted_epitopes = predict_results.get('predicted_epitopes', [])
        
        # Get style configuration
        style_dict = {
            'cartoon': {'cartoon': {}},
            'stick': {'stick': {}},
            'sphere': {'sphere': {}},
            'surface': {'surface': {}}
        }
        base_style = style_dict.get(style, {'cartoon': {}})
        
        if not predictions:
            # Fallback to basic visualization
            view.setStyle({'chain': self.chain_id}, {**base_style, 
                          list(base_style.keys())[0]: {**list(base_style.values())[0], 'color': base_color}})
            if show_surface:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Softer opacity for fallback
                    'color': base_color
                }, {'chain': self.chain_id})
            return
        
        # Filter predictions to only include predicted epitopes
        epitope_predictions = {res: prob for res, prob in predictions.items() 
                             if res in predicted_epitopes}
        
        if not epitope_predictions:
            # No predicted epitopes, show base style
            view.setStyle({'chain': self.chain_id}, {**base_style, list(base_style.keys())[0]: {**list(base_style.values())[0], 'color': base_color}})
            if show_surface:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Softer opacity for fallback
                    'color': base_color
                }, {'chain': self.chain_id})
            return
        
        # Get probability range for predicted epitopes only
        probs = list(epitope_predictions.values())
        min_prob, max_prob = min(probs), max(probs)
        
        # Improved color scheme - use very light purple gradient for better contrast with gray
        # This avoids confusion with gray background when probability is low
        epitope_colors = [
            '#F8F8FF',  # Ghost white
            '#F0F0FF',  # Very light lavender
            '#E6E6FA',  # Lavender
            '#DDD0F0',  # Light purple
            '#D8BFD8',  # Thistle
            '#C8A2C8',  # Light orchid
            '#BA90D3'   # Medium light orchid
        ]
        n_colors = len(epitope_colors)
        
        # Set base style for entire protein
        view.setStyle({'chain': self.chain_id}, {**base_style, list(base_style.keys())[0]: {**list(base_style.values())[0], 'color': base_color}})
        
        # Color predicted epitopes based on probability with purple gradient
        for residue_num, prob in epitope_predictions.items():
            # Normalize probability to [0, 1] within the epitope range
            if max_prob > min_prob:
                norm_prob = (prob - min_prob) / (max_prob - min_prob)
            else:
                norm_prob = 0.5
            
            # Map to color index
            color_idx = int(norm_prob * (n_colors - 1))
            color_idx = max(0, min(color_idx, n_colors - 1))
            color = epitope_colors[color_idx]
            
            # Add style for this residue with vivid color
            style_name = list(base_style.keys())[0]
            colored_style = {style_name: {'color': color}}
            view.addStyle(
                {'chain': self.chain_id, 'resi': residue_num},
                colored_style
            )
        
        # Add surface overlay if requested
        if show_surface:
            # Add base surface for non-epitope regions
            all_residues = set(int(res) for res in self.residue_index)
            non_epitope_residues = all_residues - set(predicted_epitopes)
            
            if non_epitope_residues:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Softer opacity for fallback
                    'color': base_color
                }, {'chain': self.chain_id, 'resi': list(non_epitope_residues)})
            
            # Add colored surfaces for predicted epitopes
            for residue_num, prob in epitope_predictions.items():
                # Normalize probability to [0, 1] within the epitope range
                if max_prob > min_prob:
                    norm_prob = (prob - min_prob) / (max_prob - min_prob)
                else:
                    norm_prob = 0.5
                
                # Map to color index
                color_idx = int(norm_prob * (n_colors - 1))
                color_idx = max(0, min(color_idx, n_colors - 1))
                color = epitope_colors[color_idx]
                
                # Add surface for this residue
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity,
                    'color': color
                }, {'chain': self.chain_id, 'resi': residue_num})
    
    def _add_prediction_region_mode(self, view, style, predict_results, base_color, prediction_color,
                                show_surface, surface_opacity, show_shape, shape_opacity,
                                show_center, center_radius, wireframe, radius, max_spheres):
        """Add visualization for prediction results in region mode"""
        # Get top-k regions
        top_k_regions = predict_results.get('top_k_regions', [])
        top_k_region_residues = predict_results.get('top_k_region_residues', [])
        
        # Get style configuration
        style_dict = {
            'cartoon': {'cartoon': {}},
            'stick': {'stick': {}},
            'sphere': {'sphere': {}},
            'surface': {'surface': {}}
        }
        base_style = style_dict.get(style, {'cartoon': {}})
        
        if not top_k_region_residues:
            # No regions, show base style
            view.setStyle({'chain': self.chain_id}, {**base_style, list(base_style.keys())[0]: {**list(base_style.values())[0], 'color': base_color}})
            if show_surface:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Softer opacity for fallback
                    'color': base_color
                }, {'chain': self.chain_id})
            return
        
        # Set base style for entire protein
        view.setStyle({'chain': self.chain_id}, {**base_style, list(base_style.keys())[0]: {**list(base_style.values())[0], 'color': base_color}})
        
        # Color all residues in top-k regions with uniform purple
        if top_k_region_residues:
            style_name = list(base_style.keys())[0]
            colored_style = {style_name: {'color': prediction_color}}
            view.addStyle(
                {'chain': self.chain_id, 'resi': top_k_region_residues},
                colored_style
            )
        
        # Add surface overlay if requested
        if show_surface:
            # Add base surface for non-region residues
            all_residues = set(int(res) for res in self.residue_index)
            non_region_residues = all_residues - set(top_k_region_residues)
            
            if non_region_residues:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Softer opacity for fallback
                    'color': base_color
                }, {'chain': self.chain_id, 'resi': list(non_region_residues)})
            
            # Color all residues in top-k regions with uniform purple surface
            if top_k_region_residues:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity,
                    'color': prediction_color
                }, {'chain': self.chain_id, 'resi': top_k_region_residues})
        
        # Add spherical regions if requested
        if show_shape and top_k_regions:
            self._add_multi_shape_visualization(
                view, top_k_regions, radius, max_spheres,
                show_center, center_radius, shape_opacity, wireframe
            )
    
    def _add_evaluation_visualization(self, view, style, predict_results,
                                  base_color, true_epitope_color, false_positive_color, 
                                  true_positive_color, coverage_color,
                                  show_surface, surface_opacity, show_shape, radius, max_spheres,
                                  target_region_indices=None, shape_opacity=0.3, show_center=True,
                                  center_radius=0.7, wireframe=True, center_color='#2C3E50'):
        """Add visualization for evaluation results with optional region filtering"""
        # Get prediction results
        predicted_epitopes = set(predict_results.get('predicted_epitopes', []))
        true_epitopes = set(predict_results.get('true_epitopes', []))
        
        # Calculate different categories
        true_positives = predicted_epitopes & true_epitopes
        false_positives = predicted_epitopes - true_epitopes
        false_negatives = true_epitopes - predicted_epitopes
        
        # Get style configuration
        style_dict = {
            'cartoon': {'cartoon': {}},
            'stick': {'stick': {}},
            'sphere': {'sphere': {}},
            'surface': {'surface': {}}
        }
        base_style = style_dict.get(style, {'cartoon': {}})
        
        # Set base style for entire protein
        view.setStyle({'chain': self.chain_id}, {**base_style, list(base_style.keys())[0]: {**list(base_style.values())[0], 'color': base_color}})
        
        # Add colored styles for specific categories with vivid colors
        for residues, color in [
            (true_positives, true_positive_color),
            (false_positives, false_positive_color),
            (false_negatives, true_epitope_color)
        ]:
            if residues:
                # Create style with the specified color
                style_name = list(base_style.keys())[0]
                colored_style = {style_name: {'color': color}}
                view.addStyle(
                    {'chain': self.chain_id, 'resi': list(residues)},
                    colored_style
                )
        
        # Add surface overlay if requested (works with any base style)
        if show_surface:
            # Get all colored residues
            all_colored_residues = true_positives | false_positives | false_negatives
            
            # Only add base surface for non-colored regions to avoid covering colored surfaces
            if all_colored_residues:
                all_residues = set(int(res) for res in self.residue_index)
                non_colored_residues = all_residues - all_colored_residues
                
                if non_colored_residues:
                    view.addSurface(py3Dmol.VDW, {
                        'opacity': surface_opacity * 0.9,  # Softer opacity for fallback
                        'color': base_color
                    }, {'chain': self.chain_id, 'resi': list(non_colored_residues)})
            else:
                # If no colored residues, show entire surface in base color
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Softer opacity for fallback
                    'color': base_color
                }, {'chain': self.chain_id})
            
            # Add colored surfaces for specific categories
            for residues, color in [
                (true_positives, true_positive_color),
                (false_positives, false_positive_color),
                (false_negatives, true_epitope_color)
            ]:
                if residues:
                    view.addSurface(py3Dmol.VDW, {
                        'opacity': surface_opacity,  # Full opacity for clear colors
                        'color': color
                    }, {'chain': self.chain_id, 'resi': list(residues)})
        
        # Show spherical regions if requested
        if show_shape and 'top_k_regions' in predict_results:
            top_k_regions = predict_results['top_k_regions']
            selected_regions = []
            
            if target_region_indices is not None and len(target_region_indices) > 0:
                # Show only the selected regions, keep track of original indices
                for region_idx in target_region_indices:
                    if 0 <= region_idx < len(top_k_regions):
                        selected_regions.append((region_idx, top_k_regions[region_idx]))
                
                # Use radius from prediction results or provided radius
                sphere_radius = radius or 19.0
                
                # Define distinct colors for different regions
                region_colors = [
                    '#FF6B6B',  # Soft red
                    '#4ECDC4',  # Teal
                    '#FFD93D',  # Bright yellow
                    '#6BCF7F',  # Green
                    '#A8E6CF',  # Mint green
                    '#FFB3BA',  # Light pink
                    '#BFBFFF',  # Light blue
                    '#FFDAB9',  # Peach
                    '#D8BFD8',  # Thistle
                    '#98D8C8',  # Light teal
                    '#F7DC6F',  # Light yellow
                    '#B19CD9'   # Medium light purple
                ]
                
                # Add spheres for selected regions
                for original_idx, region in selected_regions:
                    center_res = region['center_residue']
                    
                    # Select color based on original region index
                    shape_color = region_colors[original_idx % len(region_colors)]
                    
                    # Add sphere for the selected region with region-specific color
                    self._add_shape_visualization(
                        view, center_res, sphere_radius,
                        shape_color, center_color,
                        show_center, center_radius,
                        shape_opacity * 0.6,  # Reduced shape opacity for evaluation mode
                        wireframe
                    )
                    
                    # Highlight center residue with softer color matching the region
                    style_dict = {
                        'cartoon': {'cartoon': {}},
                        'stick': {'stick': {}},
                        'sphere': {'sphere': {}},
                        'surface': {'surface': {}}
                    }
                    base_style = style_dict.get(style, {'cartoon': {}})
                    style_name = list(base_style.keys())[0]
                    view.addStyle(
                        {'chain': self.chain_id, 'resi': center_res},
                        {style_name: {'color': shape_color}}
                    )
            else:
                # Show all regions (original behavior)
                top_regions = top_k_regions[:max_spheres] if max_spheres else top_k_regions
                self._add_multi_shape_visualization(
                    view, top_regions, radius, max_spheres,
                    show_center, center_radius, shape_opacity * 0.6, wireframe
                )
    
    def _add_probability_visualization(self, view, style, predict_results,
                                  base_color, colormap, show_surface, surface_opacity, threshold,
                                  region_indices, radius, show_shape, shape_opacity, 
                                  show_center, center_radius, wireframe, coverage_color, center_color):
        """
        Add visualization based on prediction probabilities with enhanced support for 
        specific region selection and surface rendering.
        
        Args:
            view: py3Dmol view object
            style (str): Protein representation style
            predict_results (dict): Results from predict() function
            base_color (str): Base color for non-highlighted residues
            colormap (str): Colormap name for probability visualization
            show_surface (bool): Whether to show surface
            surface_opacity (float): Surface opacity
            threshold (float): Probability threshold for coloring
            region_indices (list): List of region indices to show (0-based), None for all
                                   Each region uses a distinct color for shape visualization
            radius (float): Radius for spherical regions
            show_shape (bool): Whether to show spherical shapes
            shape_opacity (float): Shape opacity
            show_center (bool): Whether to show center points
            center_radius (float): Center point radius
            wireframe (bool): Whether to show wireframe spheres
            coverage_color (str): Color for coverage regions (not used when region_indices is specified)
            center_color (str): Color for center points
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Get probability predictions and top-k regions
        predictions = predict_results.get('predictions', {})
        top_k_regions = predict_results.get('top_k_regions', [])
        
        # Get style configuration
        style_dict = {
            'cartoon': {'cartoon': {}},
            'stick': {'stick': {}},
            'sphere': {'sphere': {}},
            'surface': {'surface': {}}
        }
        base_style = style_dict.get(style, {'cartoon': {}})
        
        if not predictions:
            # Fallback to basic visualization
            view.setStyle({'chain': self.chain_id}, {**base_style, 
                          list(base_style.keys())[0]: {**list(base_style.values())[0], 'color': base_color}})
            if show_surface:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Softer opacity for fallback
                    'color': base_color
                }, {'chain': self.chain_id})
            return
        
        # Set base style for entire protein
        view.setStyle({'chain': self.chain_id}, {**base_style, 
                      list(base_style.keys())[0]: {**list(base_style.values())[0], 'color': base_color}})
        
        # Determine which residues to color based on region_indices
        target_residues = {}  # residue_num -> probability
        selected_regions = []  # List of (original_index, region_data) tuples
        
        if region_indices is not None and len(region_indices) > 0:
            # Show only the selected regions, keep track of original indices
            for region_idx in region_indices:
                if 0 <= region_idx < len(top_k_regions):
                    selected_regions.append((region_idx, top_k_regions[region_idx]))
            
            # Get probabilities for residues in all selected regions
            for original_idx, region in selected_regions:
                covered_residues = region.get('covered_residues', [])
                for res_num in covered_residues:
                    if res_num in predictions:
                        target_residues[res_num] = predictions[res_num]
        else:
            # Show all residues with probabilities above threshold
            target_residues = {res: prob for res, prob in predictions.items() 
                             if prob >= threshold}
        
        if not target_residues:
            # No residues to color, show base style with surface
            if show_surface:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Softer opacity for fallback
                    'color': base_color
                }, {'chain': self.chain_id})
            return
        
        # Normalize probabilities for selected residues
        probs = list(target_residues.values())
        min_prob, max_prob = min(probs), max(probs)
        
        # Enhanced color scheme for better visibility on surface - use purple gradient like prediction mode
        if colormap in ['RdYlBu_r', 'coolwarm', 'RdBu_r']:
            # Use predefined purple color scheme matching prediction mode
            probability_colors = [
                '#F8F8FF',  # Ghost white (low probability)
                '#F0F0FF',  # Very light lavender
                '#E6E6FA',  # Lavender
                '#DDD0F0',  # Light purple
                '#D8BFD8',  # Thistle
                '#C8A2C8',  # Light orchid
                '#BA90D3',  # Medium light orchid
                '#B19CD9',  # Light medium orchid
                '#A594D1',  # Medium light purple
                '#9B8BC9'   # Soft medium purple (high probability)
            ]
            n_colors = len(probability_colors)
        else:
            # Use purple gradient matching prediction mode for consistency
            probability_colors = [
                '#F8F8FF',  # Ghost white
                '#F0F0FF',  # Very light lavender
                '#E6E6FA',  # Lavender
                '#DDD0F0',  # Light purple
                '#D8BFD8',  # Thistle
                '#C8A2C8',  # Light orchid
                '#BA90D3',  # Medium light orchid
                '#B19CD9',  # Light medium orchid
                '#A594D1',  # Medium light purple
                '#9B8BC9'   # Soft medium purple
            ]
            n_colors = len(probability_colors)
        
        # Color residues based on normalized probability
        colored_residues = []
        for residue_num, prob in target_residues.items():
            # Normalize probability to [0, 1] within the selected range
            if max_prob > min_prob:
                norm_prob = (prob - min_prob) / (max_prob - min_prob)
            else:
                norm_prob = 0.5
            
            # Map to color index
            color_idx = int(norm_prob * (n_colors - 1))
            color_idx = max(0, min(color_idx, n_colors - 1))
            color = probability_colors[color_idx]
            
            # Add style for this residue with vivid color
            style_name = list(base_style.keys())[0]
            colored_style = {style_name: {'color': color}}
            view.addStyle(
                {'chain': self.chain_id, 'resi': residue_num},
                colored_style
            )
            colored_residues.append(residue_num)
        
        # Add surface rendering with improved visibility
        if show_surface:
            # Add base surface for non-colored regions
            all_residues = set(int(res) for res in self.residue_index)
            non_colored_residues = all_residues - set(colored_residues)
            
            if non_colored_residues:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Further reduced opacity for softer appearance
                    'color': base_color
                }, {'chain': self.chain_id, 'resi': list(non_colored_residues)})
            
            # Add colored surfaces for probability residues with enhanced visibility
            for residue_num, prob in target_residues.items():
                # Normalize probability to [0, 1] within the selected range
                if max_prob > min_prob:
                    norm_prob = (prob - min_prob) / (max_prob - min_prob)
                else:
                    norm_prob = 0.5
                
                # Map to color index
                color_idx = int(norm_prob * (n_colors - 1))
                color_idx = max(0, min(color_idx, n_colors - 1))
                color = probability_colors[color_idx]
                
                # Add surface for this residue with softer opacity for gentler visualization
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Slightly reduced opacity for softer colors
                    'color': color
                }, {'chain': self.chain_id, 'resi': residue_num})
        
        # Add spherical region visualization if specific regions are selected
        if selected_regions and show_shape:
            # Use radius from prediction results or provided radius
            sphere_radius = radius or 19.0
            
            # Define distinct colors for different regions - matching prediction mode
            region_colors = [
                '#FF6B6B',  # Soft red
                '#4ECDC4',  # Teal
                '#FFD93D',  # Bright yellow
                '#6BCF7F',  # Green
                '#A8E6CF',  # Mint green
                '#FFB3BA',  # Light pink
                '#BFBFFF',  # Light blue
                '#FFDAB9',  # Peach
                '#D8BFD8',  # Thistle
                '#98D8C8',  # Light teal
                '#F7DC6F',  # Light yellow
                '#B19CD9'   # Medium light purple
            ]
            
            # Add spheres for all selected regions
            for original_idx, region in selected_regions:
                center_res = region['center_residue']
                
                # Select color based on original region index, not loop index
                shape_color = region_colors[original_idx % len(region_colors)]
                
                # Add sphere for the selected region with softer appearance and region-specific color
                self._add_shape_visualization(
                    view, center_res, sphere_radius,
                    shape_color, center_color,  # Use region-specific color for shape
                    show_center, center_radius,
                    shape_opacity * 0.6,  # Reduced shape opacity for softer appearance
                    wireframe
                )
                
                # Highlight center residue with softer color matching the region
                view.addStyle(
                    {'chain': self.chain_id, 'resi': center_res},
                    {list(base_style.keys())[0]: {'color': shape_color}}  # Use region-specific color
                )
    
    def _add_top_regions_visualization(self, view, style, predict_results,
                                  base_color, coverage_color, center_color,
                                  show_surface, show_shape, show_center,
                                  surface_opacity, shape_opacity, center_radius,
                                  wireframe, radius, max_spheres):
        """Add visualization for top-k regions"""
        # Set base style
        view.setStyle({'chain': self.chain_id}, {style: {'color': base_color}})
        
        # Get top regions
        top_regions = predict_results.get('top_k_regions', [])
        
        # Limit number of regions if max_spheres is specified
        if max_spheres is not None:
            top_regions = top_regions[:max_spheres]
        
        # Enhanced color scheme for different regions - matching prediction mode
        region_colors = [
            '#FF6B6B',  # Soft red
            '#4ECDC4',  # Teal
            '#FFD93D',  # Bright yellow
            '#6BCF7F',  # Green
            '#A8E6CF',  # Mint green
            '#FFB3BA',  # Light pink
            '#BFBFFF',  # Light blue
            '#FFDAB9',  # Peach
            '#D8BFD8',  # Thistle
            '#98D8C8',  # Light teal
            '#F7DC6F',  # Light yellow
            '#B19CD9'   # Medium light purple
        ]
        
        for i, region in enumerate(top_regions):
            center_res = region['center_residue']
            covered_residues = region.get('covered_residues', [])
            region_color = region_colors[i % len(region_colors)]
            
            # Color covered residues
            if covered_residues:
                view.addStyle(
                    {'chain': self.chain_id, 'resi': covered_residues},
                    {style: {'color': region_color}}
                )
            
            # Add spherical region
            if show_shape:
                self._add_shape_visualization(
                    view, center_res, radius or 18.0,
                    region_color, center_color,
                    show_center, center_radius * 0.8,
                    shape_opacity, wireframe
                )
        
        # Add surface with balanced visibility
        if show_surface:
            # Base surface with good visibility
            view.addSurface(py3Dmol.VDW, {
                'opacity': surface_opacity * 0.9,  # Keep base surface visible
                'color': base_color
            })
            
            # Colored surface for covered residues
            for i, region in enumerate(top_regions):
                covered_residues = region.get('covered_residues', [])
                region_color = region_colors[i % len(region_colors)]
                
                if covered_residues:
                    view.addSurface(py3Dmol.VDW, {
                        'opacity': surface_opacity,  # Full opacity for clear coloring
                        'color': region_color
                    }, {'resi': covered_residues})
    
    def _add_comparison_visualization(self, view, style, predict_results,
                                 base_color, true_epitope_color, false_positive_color,
                                 true_positive_color, coverage_color, show_surface, surface_opacity):
        """Add visualization comparing voted vs predicted epitopes"""
        # Set base style
        view.setStyle({'chain': self.chain_id}, {style: {'color': base_color}})
        
        # Get different prediction sets
        predicted_epitopes = set(predict_results.get('predicted_epitopes', []))
        voted_epitopes = set(predict_results.get('voted_epitopes', []))
        true_epitopes = set(predict_results.get('true_epitopes', []))
        
        # Calculate overlaps
        both_methods = predicted_epitopes & voted_epitopes  # Agreed by both methods
        only_predicted = predicted_epitopes - voted_epitopes  # Only by probability
        only_voted = voted_epitopes - predicted_epitopes  # Only by voting
        
        # Further categorize by true epitopes
        both_correct = both_methods & true_epitopes
        both_incorrect = both_methods - true_epitopes
        only_pred_correct = only_predicted & true_epitopes
        only_pred_incorrect = only_predicted - true_epitopes
        only_vote_correct = only_voted & true_epitopes
        only_vote_incorrect = only_voted - true_epitopes
        
        # Assign colors and styles
        color_assignments = [
            (both_correct, '#00FF00'),      # Bright green: both correct
            (both_incorrect, '#FF0000'),    # Red: both wrong
            (only_pred_correct, '#90EE90'), # Light green: only predicted correct
            (only_pred_incorrect, '#FFB6C1'), # Light red: only predicted wrong
            (only_vote_correct, '#87CEEB'),  # Sky blue: only voted correct
            (only_vote_incorrect, '#DDA0DD') # Plum: only voted wrong
        ]
        
        for residues, color in color_assignments:
            if residues:
                view.addStyle(
                    {'chain': self.chain_id, 'resi': list(residues)},
                    {style: {'color': color}}
                )
        
        # Add surface
        if show_surface:
            view.addSurface(py3Dmol.VDW, {
                'opacity': surface_opacity,
                'color': base_color
            })
            
            for residues, color in color_assignments:
                if residues:
                    view.addSurface(py3Dmol.VDW, {
                        'opacity': surface_opacity,
                        'color': color
                    }, {'resi': list(residues)})

    def _create_base_view(self, width: int, height: int) -> py3Dmol.view:
        """创建基本的py3Dmol视图并添加蛋白质结构"""
        view = py3Dmol.view(width=width, height=height)
        
        # 构建PDB字符串
        pdb_str = "MODEL        1\n"
        atom_num = 1
        for res_idx in range(len(self.sequence)):
            one_letter = self.sequence[res_idx]
            resname = self.convert_letter_1to3(one_letter)
            resnum = self.residue_index[res_idx]
            
            mask = self.atom37_mask[res_idx]
            coords = self.atom37_positions[res_idx][mask]
            atoms = [name for name, exists in zip(RC.atom_types, mask) if exists]
            
            for atom_name, coord in zip(atoms, coords):
                x, y, z = coord
                pdb_str += (f"ATOM  {atom_num:5d}  {atom_name:<3s} {resname:>3s} {self.chain_id:1s}{resnum:4d}"
                        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
                atom_num += 1
        
        pdb_str += "ENDMDL\n"
        view.addModel(pdb_str, "pdb")
        return view

    def _add_epitope_visualization(self, view, style, predicted_epitopes,
                             base_color, true_epitope_color, false_positive_color, true_positive_color, coverage_color,
                             show_surface, surface_opacity, show_coverage,
                             center_res=None, radius=None):
        """添加表位可视化"""
        # 设置基础颜色
        view.setStyle({'chain': self.chain_id}, {style: {'color': base_color}})
        
        true_epitopes = set(self.get_epitope_residue_numbers())
        true_positives = set(predicted_epitopes) & true_epitopes
        false_positives = set(predicted_epitopes) - true_epitopes
        false_negatives = true_epitopes - set(predicted_epitopes)
        
        # 如果提供了center_res和radius，获取覆盖区域
        covered_residues = []
        if center_res is not None and radius is not None:
            coverage_dict, _, _ = self.get_surface_coverage(
                radius=radius, threshold=0.25, index=False  # Use residue numbers for visualization
            )
            covered_res_list = coverage_dict.get(center_res, [[], [], 0, 0])[0]
            covered_residues = covered_res_list
            
            # 计算覆盖区域内的True Negative (不是表位也没被预测为表位)
            if covered_residues:
                true_negatives = [res for res in covered_residues 
                                 if res not in true_epitopes and res not in predicted_epitopes]
                
                # 为True Negative添加特殊颜色 (使用更明显的灰色)
                true_negative_color = '#888888'  # 更深的灰色
                
                if true_negatives:
                    view.addStyle(
                        {'chain': self.chain_id, 'resi': true_negatives},
                        {style: {'color': true_negative_color}}
                    )
        
        # 添加样式 - 增加颜色的饱和度
        for residues, color in [
            (true_positives, true_positive_color),
            (false_positives, false_positive_color),
            (false_negatives, true_epitope_color)
        ]:
            if residues:
                view.addStyle(
                    {'chain': self.chain_id, 'resi': list(residues)},
                    {style: {'color': color}}
                )
        
        # 先添加基础表面
        if show_surface:
            # Base surface with good visibility for overall structure
            view.addSurface(py3Dmol.VDW, {
                'opacity': surface_opacity * 0.9,  # Keep base surface visible
                'color': base_color
            })
            
            # Colored surfaces for specific categories overlay on base surface
            for residues, color in [
                (true_positives, true_positive_color),
                (false_positives, false_positive_color),
                (false_negatives, true_epitope_color)
            ]:
                if residues:
                    view.addSurface(py3Dmol.VDW, {
                        'opacity': surface_opacity,  # Full opacity for clear coloring
                        'color': color
                    }, {'resi': list(residues)})
        
        # 为覆盖区域内的True Negative添加表面
        if center_res is not None and radius is not None and covered_residues and show_coverage:
            true_negatives = [res for res in covered_residues 
                             if res not in true_epitopes and res not in predicted_epitopes]
            
            if true_negatives:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity,
                    'color': coverage_color
                }, {'resi': true_negatives})

    def _add_shape_visualization(self, view, center_res, radius,
                            coverage_color, center_color,
                            show_center, center_radius,
                            shape_opacity, wireframe):
        """添加球形可视化"""
        center_idx = self.resnum_to_index.get(center_res)
        if center_idx is None:
            return
            
        ca_idx = RC.atom_order["CA"]  # Get CA atom index from atom order
        center_coord = self.atom37_positions[center_idx, ca_idx, :]
        
        # 添加球形
        sphere_params = {
            'center': {'x': float(center_coord[0]), 
                    'y': float(center_coord[1]), 
                    'z': float(center_coord[2])},
            'radius': float(radius),
            'color': coverage_color
        }
        if wireframe:
            sphere_params.update({'wireframe': True, 'linewidth': 1.5})  # 增加线宽
        else:
            sphere_params.update({'opacity': shape_opacity})
        view.addSphere(sphere_params)
        
        # 添加中心点标记
        if show_center:
            view.addSphere({
                'center': {'x': float(center_coord[0]), 
                        'y': float(center_coord[1]), 
                        'z': float(center_coord[2])},
                'radius': float(center_radius),
                'color': center_color,
                'opacity': 1.0
            })

    def _add_coverage_visualization(self, view, style, center_res, radius,
                              base_color, coverage_color, true_positive_color, true_epitope_color,
                              show_surface, show_shape, show_center,
                              surface_opacity, shape_opacity, center_radius,
                              n_points, center_color, wireframe, show_epitope):
        """添加覆盖区域可视化"""
        # 首先设置基础样式和颜色
        view.setStyle({'chain': self.chain_id}, {style: {'color': base_color}})
        
        # 获取覆盖区域
        coverage_dict, _, _ = self.get_surface_coverage(
            radius=radius, threshold=0.25, index=False  # Use residue numbers for visualization
        )
        
        covered_res_list = coverage_dict.get(center_res, [[], [], 0, 0])[0]
        covered_residues = covered_res_list

        
        if show_epitope:
            true_epitopes = set(self.get_epitope_residue_numbers())
        else:
            true_epitopes = set()
        
        # 计算不同类别的残基
        true_positives = set(covered_residues) & true_epitopes  # 被覆盖的表位
        false_negatives = true_epitopes - set(covered_residues)  # 未被覆盖的表位
        covered_non_epitopes = set(covered_residues) - true_epitopes  # 被覆盖的非表位
        
        # 添加表面渲染
        if show_surface:
            # 1. 添加基础表面，保持可见性
            view.addSurface(py3Dmol.VDW, {
                'opacity': surface_opacity * 1.0,  # Keep base surface visible
                'color': base_color
            })
            
            # 2. 添加未被覆盖的表位表面
            if false_negatives:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity,  # Full opacity for clear coloring
                    'color': true_epitope_color
                }, {'resi': list(false_negatives)})
            
            # 3. 添加被覆盖的表位表面
            if true_positives:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity,  # Full opacity for clear coloring
                    'color': true_positive_color
                }, {'resi': list(true_positives)})
            
            # 4. 添加被覆盖的非表位表面
            if covered_non_epitopes:
                view.addSurface(py3Dmol.VDW, {
                    'opacity': surface_opacity * 0.9,  # Slightly reduced for distinction
                    'color': coverage_color
                }, {'resi': list(covered_non_epitopes)})
            
            # 添加样式
            if false_negatives:
                view.addStyle(
                    {'chain': self.chain_id, 'resi': list(false_negatives)},
                    {style: {'color': true_epitope_color}}
                )
            
            if true_positives:
                view.addStyle(
                    {'chain': self.chain_id, 'resi': list(true_positives)},
                    {style: {'color': true_positive_color}}
                )
            
            if covered_non_epitopes:
                view.addStyle(
                    {'chain': self.chain_id, 'resi': list(covered_non_epitopes)},
                    {style: {'color': coverage_color}}
                )
            
            # 为中心残基添加黄色样式
            view.addStyle(
                {'chain': self.chain_id, 'resi': center_res},
                {style: {'color': '#FFD700'}}  # 使用更鲜艳的黄色
            )
        
        # 添加形状
        if show_shape:
            self._add_shape_visualization(
                view, center_res, radius,
                coverage_color,
                center_color,
                show_center, center_radius,
                shape_opacity, wireframe
            )

    def _add_multi_shape_visualization(self, view, regions_data, radius, max_spheres, 
                                   show_center, center_radius, shape_opacity, wireframe):
        """Add multiple spherical regions with different colors"""
        if not regions_data:
            return
            
        # Limit number of spheres if max_spheres is specified
        regions_to_show = regions_data[:max_spheres] if max_spheres else regions_data
        
        # Enhanced color scheme for different regions with more distinct colors
        sphere_colors = [
            '#FF6B6B',  # Soft red
            '#4ECDC4',  # Teal
            '#FFD93D',  # Bright yellow
            '#6BCF7F',  # Green
            '#98D8C8',  # Light teal
            '#A8E6CF',  # Mint green
            '#FFB3BA',  # Light pink
            '#BFBFFF',  # Light blue
            '#FFDAB9',  # Peach
            '#D8BFD8',  # Thistle
            '#98D8C8',  # Light teal
            '#F7DC6F',  # Light yellow
        ]
        
        for i, region_data in enumerate(regions_to_show):
            if isinstance(region_data, dict):
                # For prediction/evaluation results format
                center_res = region_data['center_residue']
            else:
                # For simple center residue format
                center_res = region_data
                
            sphere_color = sphere_colors[i % len(sphere_colors)]
            self._add_shape_visualization(
                view, center_res, radius or 18.0,
                sphere_color, '#FFD700',  # Gold for center
                show_center, center_radius, shape_opacity, wireframe
            )

    @classmethod
    def from_pdb(
        cls,
        path: Optional[PathOrBuffer] = None,
        chain_id: str = "detect",
        id: Optional[str] = None,
        is_predicted: bool = False,
    ) -> "AntigenChain":
        """
        Return a AntigenChain object from a pdb file.

        If `path` is not provided, the function will try multiple possible paths:
            1. {id}_{chain_id}.pdb
            2. {id}.pdb
            3. {id.lower()}_{chain_id}.pdb
            4. {id.upper()}_{chain_id}.pdb
        If none of these paths exist, it will download the structure from RCSB PDB
        and save it to the antigen_structs directory.

        Args:
            path (Optional[PathOrBuffer]): Path or buffer to read pdb file from. If None,
                the default path is constructed from DATA_DIR.
            chain_id (str, optional): Select a chain corresponding to (author) chain id.
                "detect" uses the first detected chain.
            id (Optional[str], optional): Protein identifier (pdb_id). If not provided and `path`
                is given, the id will be inferred from the file name.
            is_predicted (bool, optional): If True, reads b factor as the confidence readout.

        Returns:
            AntigenChain: The constructed antigen chain.
        """
        # If no path is provided, try multiple possible paths
        id = id.lower()
        
        if path is None:
            if id is None:
                raise ValueError("Either 'path' or 'id' must be provided to locate the pdb file.")
            
            # Try multiple possible paths
            possible_paths = [
                Path(BASE_DIR) / "data" / "antigen_structs" / f"{id}_{chain_id}.pdb",
                Path(BASE_DIR) / "data" / "antigen_structs" / f"{id}.pdb",
                # Path(BASE_DIR) / "data" / "antigen_structs" / f"{id.lower()}_{chain_id}.pdb",
                # Path(BASE_DIR) / "data" / "antigen_structs" / f"{id.upper()}_{chain_id}.pdb",
                # Path(BASE_DIR) / "data" / "pdb" / f"{id.lower()}.pdb",
                # Path(BASE_DIR) / "data" / "pdb" / f"{id.upper()}.pdb",
            ]
            
            # Try each path
            path = None
            for p in possible_paths:
                if p.exists():
                    path = p
                    print(f"Found pdb file at {path}")
                    break
            
            # If no path exists, download from RCSB
            if path is None:
                try:
                    # Create directory if it doesn't exist
                    save_dir = Path(BASE_DIR) / "data" / "pdb"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Download from RCSB
                    rcsb.fetch(id, "pdb", target_path=save_dir)
                    
                    path = save_dir / f"{id}.pdb"
                    print(f"No existing pdb file for {id}_{chain_id}, downloaded {id} complex pdb file to {path}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to download pdb file for {id}: {str(e)}")
                    return None
        else:
            path = Path(path)  # Ensure path is a Path object

        # Determine the file_id from the provided id or from the path.
        if id is not None:
            file_id = id
        else:
            # Infer file_id from the file name if id is not provided.
            file_id = path.with_suffix("").name

        # Read the pdb file.
        try:
            atom_array = PDBFile.read(path).get_structure(model=1, extra_fields=["b_factor"])
        except Exception as e:
            print(f"[ERROR] Failed to read pdb file {path}: {str(e)}")
            return None
        
        # If chain_id is "detect", use the first detected chain.
        if chain_id == "detect":
            chain_id = atom_array.chain_id[0]
            print(f"[WARNING] No chain_id provided, using the first detected chain: {chain_id}")

        # Filter the AtomArray: amino acids, non-hetero atoms, and matching chain.
        atom_array = atom_array[
            bs.filter_amino_acids(atom_array)
            & ~atom_array.hetero
            & (atom_array.chain_id == chain_id)
        ]

        # Set entity_id as 1 (not supplied in PDB files)
        entity_id = 1

        # Build the sequence by converting three-letter codes to one-letter codes.
        sequence = "".join(
            (
                r if len((r := PDBData.protein_letters_3to1.get(monomer[0].res_name, "X"))) == 1 else "X"
            )
            for monomer in bs.residue_iter(atom_array)
        )
        num_res = len(sequence)

        # Prepare arrays for atom coordinates, mask, residue indices, etc.
        atom_positions = np.full([num_res, RC.atom_type_num, 3], np.nan, dtype=np.float32)
        atom_mask = np.full([num_res, RC.atom_type_num], False, dtype=bool)
        residue_index = np.full([num_res], -1, dtype=np.int64)
        insertion_code = np.full([num_res], "", dtype="<U4")
        confidence = np.ones([num_res], dtype=np.float32)

        # Populate arrays from the pdb data.
        for i, res in enumerate(bs.residue_iter(atom_array)):
            for atom in res:
                atom_name = atom.atom_name
                if atom_name == "SE" and atom.res_name == "MSE":
                    atom_name = "SD"
                if atom_name in RC.atom_order:
                    atom_positions[i, RC.atom_order[atom_name]] = atom.coord
                    atom_mask[i, RC.atom_order[atom_name]] = True
                    if is_predicted and atom_name == "CA":
                        confidence[i] = atom.b_factor
            residue_index[i] = res[0].res_id
            insertion_code[i] = res[0].ins_code

        # Ensure that sequence is valid.
        assert all(sequence), "Some residue name was not specified correctly"

        return cls(
            id=file_id,
            sequence=sequence,
            chain_id=chain_id,
            entity_id=entity_id,
            atom37_positions=atom_positions,
            atom37_mask=atom_mask,
            residue_index=residue_index,
            insertion_code=insertion_code,
            confidence=confidence,
        )