import os
import subprocess

from scipy.spatial import distance
import numpy as np
import pandas as pd

from Bio.Blast import NCBIXML
from pathlib import Path
import sys
from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO, PPBuilder, Select
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb

import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import re
import requests
import pickle
import tempfile

from collections import defaultdict
import shutil



# Suppress warnings from the PDB parser
warnings.simplefilter('ignore', PDBConstructionWarning)

# 0. Some tools for data processing
def get_pdb_release_date_and_resolution(pdb_id, dir="PDB_db", verbose=True):
    """
    Retrieve PDB resolution and release date, prioritizing local file data if available.

    Parameters:
        pdb_id (str): The PDB ID (e.g., '101m').
        pdb_path (str): Optional path to the local PDB file.

    Returns:
        tuple: (release_date, resolution) where:
            - release_date (str or None): Release date in 'YYYY-MM-DD' format or None.
            - resolution (float or None): Resolution in Ångströms or None.
    """
    release_date = None
    resolution = None

    # Try to get the data from the local PDB file
    pdb_path = f"{dir}/structs/{pdb_id.upper()}.pdb"
    try:
        with open(pdb_path, "r") as file:
            for line in file:
                    # Extract release date from the HEADER line
                if line.startswith("HEADER"):
                    date_str = line[50:59].strip()  # Extract the date (e.g., '15-FEB-94')
                    release_date = pd.to_datetime(date_str, format='%d-%b-%y').strftime('%Y-%m-%d')

                # Extract resolution from the REMARK line
                if line.startswith("REMARK   2 RESOLUTION."):
                    resolution = float(line.split()[3])

                # Stop reading if both resolution and release date are found
                if release_date and resolution and verbose:
                    print(f"Data found locally: Release Date = {release_date}, Resolution = {resolution} Å")
                    return release_date, resolution
    except Exception as e:
        print(f"Error reading data from local file {pdb_path}: {e}")

    # If not available locally, fetch the data from the RCSB PDB API
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract release date and resolution from API response
        release_date = data.get('rcsb_accession_info', {}).get('initial_release_date', None)
        resolution = data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]

        # Format release date if available
        release_date = release_date.split("T")[0] if release_date else None

        if verbose and release_date and resolution:
            print(f"Data fetched from RCSB API: Release Date = {release_date}, Resolution = {resolution} Å")
        return release_date, resolution

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {pdb_id}: {e}")
        return None, None
    
def getxyz(df):
    """Extracts the (x, y, z) coordinates from a DataFrame."""
    return np.array([df["x_coord"], df["y_coord"], df["z_coord"]]).T

def aa_3_to_1(resn):
    """Convert three-letter amino acid codes to one-letter codes."""
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    return d[resn]


def kalign(seq1, seq2, dir="PDB_db"):
    """
    Align two sequences using kalign, ensuring unique temporary files for multithreaded safety.
    """
    if not seq1 or not seq2:
        print("Error: One or both sequences are empty.")
        return "", ""
    if len(seq1) < 5 or len(seq2) < 5:
        print("Error: Sequences are too short for meaningful alignment.")
        return "", ""

    try:
        # Create unique temporary FASTA file
        with tempfile.NamedTemporaryFile(mode="w", dir=dir, delete=False, suffix=".fasta") as input_fasta:
            input_fasta.write(f">1\n{seq1}\n>2\n{seq2}\n")
            input_fasta_path = input_fasta.name

        # Run kalign and capture output
        result = subprocess.check_output(f"cat {input_fasta_path} | kalign -f fasta", shell=True)
        alignment = result.decode("UTF-8").split("\n")

        aligned_seqs = {}
        current_id = None
        for line in alignment:
            if line.startswith(">"):
                current_id = line[1:]
                aligned_seqs[current_id] = []
            elif current_id:
                aligned_seqs[current_id].append(line.strip())

        seq1_aligned = "".join(aligned_seqs.get("1", []))
        seq2_aligned = "".join(aligned_seqs.get("2", []))

        if not seq1_aligned or not seq2_aligned:
            print("Error: Kalign alignment failed. Check input sequences.")
            return "", ""

        return seq1_aligned, seq2_aligned
    except subprocess.CalledProcessError as e:
        print(f"Error running kalign: {e}")
        return "", ""
    finally:
        # Clean up temporary file
        if os.path.exists(input_fasta_path):
            os.remove(input_fasta_path)

def mafft_align(s1, s2, strict=True):
    """Align two sequences using MAFFT."""
    with open("m.fasta", 'w') as fo:
        fo.write(f">1\n{s1}\n>2\n{s2}\n")
    if not strict:
        d = subprocess.check_output("mafft --anysymbol --op 0.1 m.fasta", shell=True)
    else:
        d = subprocess.check_output("mafft --anysymbol --auto m.fasta", shell=True)

    res_ = d.decode("UTF-8").split("\n")
    res = []
    for l in res_:
        if len(l) == 0:
            continue
        if l[0] == ">":
            res.append("")
            continue
        res[-1] += l.rstrip()
    return res

def filter_chains_by_resolution(input_csv, output, dir="PDB_db", resolution_threshold=3.0):
    """
    Filters chains based on PDB resolution, saves chains with resolution < threshold to a text file,
    saves a CSV of rows passing the filter, and returns the list of remaining chains.

    Args:
        input_csv (str): Path to the input CSV file.
        output_txt (str): Path to save chains with resolution < threshold.
        dir (str): Directory containing PDB files.
        resolution_threshold (float): Resolution threshold (default: 3.0 Å).

    Returns:
        list: List of chains with resolution < threshold.
    """
    input_csv_path = Path(f"{dir}/{input_csv}")
    data = pd.read_csv(input_csv_path)

    # Store resolution for each row
    resolutions = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Filtering by Resolution"):
        pdb_chain = row["antigen_chain"]
        pdb_id = pdb_chain.split("_")[0]
        res = get_pdb_release_date_and_resolution(pdb_id, dir)[1]
        resolutions.append(res)

    data["Resolution"] = resolutions
    filtered_data = data[(data["Resolution"].notnull()) & (data["Resolution"] <= resolution_threshold)]
    remaining_chains = filtered_data["antigen_chain"].tolist()

    # Save filtered chains to txt
    output_txt_path = Path(dir) / f"{output}_{resolution_threshold}.txt"
    with open(output_txt_path, mode="w") as txtfile:
        txtfile.write("\n".join(remaining_chains))
    print(f"Chains with resolution < {resolution_threshold} Å saved to {output_txt_path}.")

    # Also save filtered CSV
    filtered_csv_path = Path(dir) / f"{output}_{resolution_threshold}.csv"
    filtered_data.to_csv(filtered_csv_path, index=False)
    print(f"Filtered CSV saved to {filtered_csv_path}.")

    print(f"Remaining chains: {len(remaining_chains)} with resolution < {resolution_threshold} Å.\n")
    return remaining_chains


## 1. Fetch PDB sequences and generate blast DB (Download data in 2024/11/04)
# Remark: it is very time-consuming to filter the data. 
def fetch_seq_pdb_data(output_dir="PDB_db", date_filter=None):
    """
    Fetches PDB sequences and prepares a BLAST database. Optionally filters the dataset by date.

    Parameters:
        output_dir (str): The directory to store PDB files.
        date_filter (str): Optional. Include only entries published after this date (format: YYYY-MM-DD).
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pdb_seqres_path = os.path.join(output_dir, "pdb_seqres.txt")

    # Download the PDB sequence database if not already downloaded
    if not os.path.exists(pdb_seqres_path):
        print("Downloading pdb_seqres.txt...")
        subprocess.call("wget https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz", shell=True)
        subprocess.call("gzip -d pdb_seqres.txt.gz", shell=True)
        subprocess.call(f"mv pdb_seqres.txt {output_dir}/", shell=True)
    else:
        print("PDB sequence database already exists. Skipping download...")

    # Create a BLAST database if not already prepared
    if not os.path.exists(os.path.join(output_dir, "pdb_seqres.txt.psq")):
        print("Creating BLAST database...")
        subprocess.call(f"makeblastdb -in pdb_seqres.txt -dbtype prot -title pdb", shell=True, cwd=output_dir)

    return pdb_seqres_path

## 2. Preliminary screen for proteins in the PDB database with homology to fragment antigen-binding region
def parse_blast_output(input_path, save_path="fab_hits.txt"):
        """
        Parse the BLAST XML output to extract PDB IDs with high alignment.

        Parameters:
            input_path (str): Path to the BLAST XML output file.

        Returns:
            set: A set of PDB IDs with high alignment scores.
        """
        print(f"Parsing BLAST output: {input_path}...")
        with open(input_path, "r") as result:
            records = NCBIXML.parse(result)
            item = next(records)  # Retrieve the first BLAST record from the results
            pdb_fabs = set()      # Set to store full PDB IDs with chain information
            pdb_fabs_ = set()     # Set to store only PDB IDs without chain information
            for alignment in item.alignments:
                for hsp in alignment.hsps:
                    # Extract PDB ID and chain from the alignment title
                    pdb_id = alignment.title.split()[1]
                    pdb_id_id = pdb_id.split("_")[0]  # Extract PDB ID without chain identifier
                    pdb_fabs.add(pdb_id)               # Add full PDB ID with chain
                    pdb_fabs_.add(pdb_id_id)           # Add PDB ID without chain
            print(f"Found {len(pdb_fabs)} hits in {input_path}.\n")
            return pdb_fabs
        
def screen_fab_sequences(output_dir = "PDB_db"):
    """
    Screen the PDB database for sequences homologous to fragment antigen-binding regions.

    Writes light and heavy chain sequences to separate files, runs BLAST searches if needed, and parses the results.

    Returns:
        set: A combined set of PDB IDs matching light and heavy chains.
    
    Example: {"1ABC_A", "2XYZ_B", "1DEF_C"}
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    save_path = os.path.join(output_dir, "fab_hits.txt")
    if os.path.exists(save_path):
        print(f"Loading existing PDB hits from {save_path}...")
        with open(save_path, "r") as f:
            pdb_fab_hits = {line.strip() for line in f}
        print(f"Loaded {len(pdb_fab_hits)} PDB hits.\n")
        return pdb_fab_hits

    light = "DILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELK"
    print("Writing light chain sequence to fab_light.fasta...")
    with open(os.path.join(output_dir, "fab_light.fasta"), 'w') as fo:
        fo.write(">input_light\n")
        fo.write(light)

    heavy = "QVQLKQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSINKDNSKSQVFFKMNSLQSNDTAIYYCARALTYYDYEFAYWGQGTLVTVSA"
    print("Writing heavy chain sequence to fab_heavy.fasta...")
    with open(os.path.join(output_dir, "fab_heavy.fasta"), 'w') as fo:
        fo.write(">input_heavy\n")
        fo.write(heavy)

    # Run BLAST for light chain if results do not already exist
    if not os.path.exists(os.path.join(output_dir, "hits_fabs_light.xml")):
        print("Running BLAST search for light chain...")
        subprocess.call("blastp -db pdb_seqres.txt -num_alignments 99999 -evalue 1e-9 -query fab_light.fasta -out hits_fabs_light.xml -outfmt 5", shell=True, cwd=output_dir)
    else:
        print("BLAST results for light chain already exist. Skipping search...")

    # Run BLAST for heavy chain if results do not already exist
    if not os.path.exists(os.path.join(output_dir, "hits_fabs_heavy.xml")):
        print("Running BLAST search for heavy chain...")
        subprocess.call("blastp -db pdb_seqres.txt -num_alignments 99999 -evalue 1e-9 -query fab_heavy.fasta -out hits_fabs_heavy.xml -outfmt 5", shell=True, cwd=output_dir)
    else:
        print("BLAST results for heavy chain already exist. Skipping search...")
    
    # Parse BLAST results for light and heavy chains
    print("Parsing BLAST results for light chain...")
    pdb_fab_hits_1 = parse_blast_output(os.path.join(output_dir, "hits_fabs_light.xml"))
    print("Parsing BLAST results for heavy chain...")
    pdb_fab_hits_2 = parse_blast_output(os.path.join(output_dir, "hits_fabs_heavy.xml"))

    # Combine results from light and heavy chain BLAST searches
    pdb_fab_hits = pdb_fab_hits_1 | pdb_fab_hits_2
    print(f"Total unique PDB hits: {len(pdb_fab_hits)}")

    if not os.path.exists(save_path):

        with open(save_path, "w") as outfile:
            for pdb_id in pdb_fab_hits:
                outfile.write(f"{pdb_id}\n")
        print(f"Saved PDB hits to {save_path}")
    
    return pdb_fab_hits

## 3. Screen for heavy and light fab chains using ANARCI
def load_fasta(path):
        """
        Load sequences from a FASTA file.

        Parameters:
            path (str): Path to the FASTA file.

        Returns:
            list: A list of [header, sequence] pairs.
        """
        print(f"Loading FASTA file: {path}...")
        r = []
        with open(path) as f:
            for line in f:
                if line[0] == ">":
                    r.append([])
                r[-1].append(line.rstrip())
        r = [[r_[0], "".join(r_[1:])] for r_ in r]
        print(f"Loaded {len(r)} sequences from {path}.")
        return r

def process_fab_chains(pdb_fab_hits, pdb_seqres_path="PDB_db/pdb_seqres.txt", output_dir="PDB_db"):
    """
    Process heavy and light Fab chains using ANARCI and filter sequences based on BLAST hits.

    Parameters:
        pdb_fab_hits (set): Set of PDB IDs matching Fab chains.
        pdb_seqres_path (str): Path to the PDB sequence file.
        output_dir (str): Directory for output files.

    Returns:
        None
    """
    # Load all PDB sequences
    print("Loading PDB sequences...")
    with open(pdb_seqres_path) as f:
        r = []
        for line in f:
            if line[0] == ">":
                r.append([])
            r[-1].append(line)

    # Filter sequences based on BLAST hits
    print("Filtering sequences based on BLAST hits...")
    rfabs = []
    for r_ in r:
        title = r_[0].split(" ")[0][1:]
        if title not in pdb_fab_hits:
            continue
        rfabs.append([r_[0].split(" ")[0][1:], r_[1]])
    print(f"Filtered {len(rfabs)} sequences matching BLAST hits.")

    # Save filtered FAB sequences to a new FASTA file
    filtered_fasta_path = os.path.join(output_dir, "putative_fabs.fasta")
    print(f"Saving filtered sequences to {filtered_fasta_path}...")
    if not os.path.exists(filtered_fasta_path):
        with open(filtered_fasta_path, 'w') as fo:
            for r in rfabs:
                fo.write("".join([">" + r[0] + "\n", r[1]]) + "\n")
    else:
        print(f"{filtered_fasta_path} already exists. Skipping save.")

    # Run ANARCI for heavy chains if not already done
    heavy_anarci_path = os.path.join(output_dir, "all_fabs_heavy.anarci")
    if not os.path.exists(heavy_anarci_path):
        print("Running ANARCI for heavy chains...")
        anarci_command = f"ANARCI -i putative_fabs.fasta -o all_fabs_heavy.anarci -s chothia -r ig --ncpu 8 --bit_score_threshold 100 --restrict heavy"
        subprocess.call(anarci_command, shell=True, cwd=output_dir)
    else:
        print("ANARCI results for heavy chains already exist. Skipping ANARCI run...")

    # Run ANARCI for light chains if not already done
    light_anarci_path = os.path.join(output_dir, "all_fabs_light.anarci")
    if not os.path.exists(light_anarci_path):
        print("Running ANARCI for light chains...")
        anarci_command = f"ANARCI -i putative_fabs.fasta -o all_fabs_light.anarci -s chothia -r ig --ncpu 8 --bit_score_threshold 100 --restrict light"
        subprocess.call(anarci_command, shell=True, cwd=output_dir)
    else:
        print("ANARCI results for light chains already exist. Skipping ANARCI run...\n")

## 4. Parse ANARCI output
def parse_anarci_annotation(path="light.anarci", n=108, save_path=None):
    """
    Parse ANARCI annotation output to extract aligned amino acid sequences.

    Parameters:
        path (str): Path to the ANARCI output file.
        n (int): Maximum sequence length for alignment positions.
        save_path (str): Optional. Path to save the parsed results as a file.

    Returns:
        dict: Parsed alignment data where keys are sequence names and values are aligned residues.
    """
    # Check if the save_path exists and load it directly if it does
    if save_path and os.path.exists(save_path):
        print(f"Loading existing parsed results from {save_path}...")
        out_ = {}
        with open(save_path, "r") as f:
            current_name = None
            for line in f:
                if line.startswith(">"):
                    current_name = line[1:].strip()
                    out_[current_name] = []
                elif line.startswith("Position"):
                    position_residue = line.split(": ")[-1].strip()
                    out_[current_name].append(position_residue)
        print(f"{len(out_)} sequences from ANARCI output.\n")
        return out_

    print(f"Parsing ANARCI output file: {path}...")
    seqs = []
    seqs.append([[] for _ in range(n)])
    used = set()
    data = {}

    with open(path) as f:
        w = f.readlines()
        data = [[]]
        for line in w:
            data[-1].append(line)
            if line[0] == "/":
                data.append([])

    out = {}
    for d in data:
        if len(d) == 0:
            continue
        name = d[0].rstrip().split()[-1]
        if name in out:
            continue
        out[name] = [[] for _ in range(n)]
        for d_ in d:
            if d_[0] == "#" or d_[0] == "/":
                continue
            id_ = d_.split()[1]
            id_ = int(id_)
            if d_[10] == "-":
                continue
            out[name][id_].append(d_[10])

    out_ = {name: ["".join(c) for c in alignment if c] for name, alignment in out.items() if any(alignment)}

    # Save the parsed results to a file if required
    if save_path:
        with open(save_path, "w") as f:
            for name, alignment in out_.items():
                f.write(f">{name}\n")
                for pos, residues in enumerate(alignment):
                    if residues:
                        f.write(f"Position {pos + 1}: {residues}\n")
                f.write("\n")

    print(f"Parsed {len(out_)} sequences from ANARCI output.")
    return out_

## 5. Fetch PDB structures and metadata
def standardize_date_format(date_str):
    """
    Convert dates into a standard 'YYYY-MM-DD' format.

    Parameters:
        date_str (str): Date string in formats like '11-APR-22' or '2023-10-25'.

    Returns:
        str: Date in 'YYYY-MM-DD' format or None if invalid.
    """
    if not isinstance(date_str, str) or not date_str.strip():
        return None  # Handle non-string or empty input
    try:
        if re.match(r"\d{2}-[A-Z]{3}-\d{2}", date_str):
            return datetime.strptime(date_str, "%d-%b-%y").strftime("%Y-%m-%d")
        elif re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        pass
    return None

def fetch_pdb_structures(pdb_fab_hits, anarci_list_light, anarci_list_heavy, dir="PDB_db"):
    """
    Fetch all PDB structures containing light and heavy chains for Fab complexes,
    and store additional metadata (release date and resolution).

    Parameters:
        pdb_fab_hits (set): Set of PDB IDs matching Fab chains.
        anarci_list_light (list): List of light chain IDs from ANARCI.
        anarci_list_heavy (list): List of heavy chain IDs from ANARCI.
        output_dir (str): Directory to store downloaded PDB structures.

    Returns:
        dict: A dictionary containing PDB metadata (release date, resolution, and chain details).
    """

    output_dir = os.path.join(dir, "structs")
    save_metadata_path = os.path.join(dir, "metadata.csv")

    if os.path.exists(save_metadata_path):
        print(f"Structure data has been downloaded! Loading existing metadata from {save_metadata_path}...")
        metadata_df = pd.read_csv(save_metadata_path, index_col=0)
        metadata_df['release_date'] = metadata_df['release_date'].apply(standardize_date_format)
        metadata_df['resolution'] = pd.to_numeric(metadata_df['resolution'], errors='coerce')
        pdb_3 = metadata_df.to_dict(orient="index")
        print(f"Loaded metadata for {len(pdb_3)} PDB entries.")
        return pdb_3
    
    # Create a dictionary to store light and heavy chain information
    pdb_3 = {r[:4]: {"light": [], "heavy": [], "release_date": None, "resolution": None} for r in pdb_fab_hits}

    # Combine and populate light and heavy chains in a single loop
    for chain_dict, chain_type in [(anarci_list_light, "light"), (anarci_list_heavy, "heavy")]:
        for h in chain_dict.keys():  # Use the keys of the dictionary
            h4 = h[:4]
            if h4 in pdb_3:
                pdb_3[h4][chain_type].append(h)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def download_and_extract_metadata(pdb_id):
        """
        Download a PDB structure and extract its metadata.

        Parameters:
            pdb_id (str): The PDB ID to process.

        Returns:
            tuple: The updated metadata for the PDB ID.
        """
        pdb_name = pdb_id.upper() + ".pdb.gz"
        pdb_path = os.path.join(output_dir, pdb_name)
        unzipped_path = pdb_path.rstrip(".gz")

        # Skip if already downloaded and processed
        if os.path.exists(pdb_path) or os.path.exists(unzipped_path):
            release_date, resolution = get_pdb_release_date_and_resolution(pdb_id)
            return pdb_id, standardize_date_format(release_date), resolution

        with open(os.devnull, 'w') as devnull:
            subprocess.call(f"wget https://files.rcsb.org/download/{pdb_name}",
                            shell=True, cwd=output_dir, stdout=devnull, stderr=devnull)

        # Decompress the file if downloaded
        if os.path.exists(pdb_path):
            with open(os.devnull, 'w') as devnull:
                subprocess.call(f"gzip -d {pdb_name}", shell=True, cwd=output_dir, stdout=devnull, stderr=devnull)

        # Extract metadata from the file and select the first model
        if os.path.exists(pdb_path):
            release_date, resolution = None, None
            pdb_data = []
            with open(pdb_path, "r") as f:
                for line in f:
                    pdb_data.append(line)
                    if line.startswith("HEADER"):
                        release_date = line[50:59].strip()
                    if line.startswith("REMARK   2") and "RESOLUTION." in line:
                        resolution = line.split("RESOLUTION.")[1].split()[0].strip()
                    if line.startswith("ENDMDL"):
                        break

            # Write back only the first model
            with open(pdb_path, "w") as fo:
                fo.writelines(pdb_data)

        return pdb_id, standardize_date_format(release_date), resolution

    # Use multithreading to speed up downloads and metadata extraction
    print("Fetching PDB structures and metadata...")
    total_pdbs = len(pdb_3)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(download_and_extract_metadata, pdb_3.keys()), total=total_pdbs, desc="Processing PDBs"))

    # Update the metadata in pdb_3
    for i, (pdb_id, release_date, resolution) in enumerate(results):
        if pdb_id in pdb_3:
            pdb_3[pdb_id]["release_date"] = release_date
            pdb_3[pdb_id]["resolution"] = resolution
        if (i + 1) % (total_pdbs // 20) == 0:  # Print progress every 5%
            print(f"Processed {i + 1}/{total_pdbs} PDB structures...")

    # Save metadata to a CSV file
    if save_metadata_path:
        metadata_df = pd.DataFrame.from_dict(pdb_3, orient="index")
        metadata_df.to_csv(save_metadata_path)
        print(f"Metadata saved to {save_metadata_path}")

    print("PDB structure processing complete.\n")
    return pdb_3

def filter_structures_by_resolution_and_date(pdb_3, resolution_threshold=3.0, date_threshold=None):
    """
    Filter PDB structures based on resolution and release date.

    Parameters:
        pdb_3 (dict): Dictionary containing PDB metadata.
        resolution_threshold (float): Maximum resolution to include.
        date_threshold (str): Minimum release date to include (format: 'YYYY-MM-DD').

    Returns:
        dict: Filtered dictionary of PDB structures.
    """
    filtered_pdbs = {}
    date_threshold = datetime.strptime(date_threshold, "%Y-%m-%d") if date_threshold else None

    for pdb_id, metadata in pdb_3.items():
        resolution = metadata.get("resolution")
        release_date = metadata.get("release_date")
        
        if resolution and resolution <= resolution_threshold:
            if date_threshold:
                if release_date:
                    release_date = datetime.strptime(release_date, "%Y-%m-%d")
                    if release_date >= date_threshold:
                        filtered_pdbs[pdb_id] = metadata
            else:
                filtered_pdbs[pdb_id] = metadata

    print(f"Filtered PDBs count: {len(filtered_pdbs)}")
    if date_threshold:
        print(f"Data all published after: {date_threshold.strftime('%Y-%m-%d')}")
    print(f"Resolution threshold: {resolution_threshold}\n")
    return filtered_pdbs

## 6. Prepare PDB dataframes and align full sequence (from pdb seq-res) on sequence of resloved protein (may contain some gaps)
def remove_alternative_conformations(pdb_dataframe):
    """Remove alternative conformations from a PDB DataFrame."""
    return pdb_dataframe[(pdb_dataframe["alt_loc"] == "A") | (pdb_dataframe["alt_loc"] == " ") | (pdb_dataframe["alt_loc"] == "")]

def remove_unk(pdb_dataframe):
    """Remove unknown residues (UNK) from a PDB DataFrame."""
    return pdb_dataframe[pdb_dataframe["residue_name"] != "UNK"]

def consider_insertions(pdb_dataframe):
    """Generate residue keys considering insertions in a PDB DataFrame."""
    pdb_dataframe["residue_key"] = list(zip(pdb_dataframe["residue_number"], 
                                             pdb_dataframe["insertion"], 
                                             pdb_dataframe["chain_id"], 
                                             pdb_dataframe["residue_name"]))
    return pdb_dataframe

def put_full_sequence(pdb_dataframe, full_seq):
    """Align PDB sequence with full sequence and combine data."""
    pdb_dataframe = remove_alternative_conformations(pdb_dataframe)
    pdb_dataframe = remove_unk(pdb_dataframe)
    pdb_dataframe = consider_insertions(pdb_dataframe)

    if pdb_dataframe.empty:
        print("Empty PDB DataFrame.")
        return None
    pdb_ca = pdb_dataframe[pdb_dataframe["atom_name"] == "CA"]
    residue_numbers = []
    residue_seq = []
    used = set()

    for _, row in pdb_ca.iterrows():
        one_letter = aa_3_to_1(row["residue_name"])
        if not one_letter:
            print(f"Unknown residue: {row['residue_name']}. Skipping...")
            continue
        residue_number = row["residue_key"]
        if residue_number in used:
            continue
        residue_numbers.append(residue_number)
        residue_seq.append(one_letter)

    pdb_seq = "".join(residue_seq)

    if len(pdb_seq) <= 5:
        print("PDB sequence is too short.")
        return None

    pdb_seq_aligned, full_seq_aligned = kalign(pdb_seq, full_seq)
    print(f"PDB-aligned sequence: {pdb_seq_aligned}")
    print(f"Full-aligned sequence: {full_seq_aligned}")


    assert full_seq_aligned.replace("-", "") == full_seq

    n_pdb = -1
    n_pdb_map = []

    for a_pdb, a_fullseq in zip(pdb_seq_aligned, full_seq_aligned):
        if a_pdb != '-':
            n_pdb += 1
        n_pdb_map.append({"resi": None if a_pdb == "-" else residue_numbers[n_pdb],
                          "a_pdb": a_pdb if a_pdb != "-" else None,
                          "a_full": a_fullseq})

    full_df = []
    for mapping in n_pdb_map:
        if mapping["resi"] is None:
            empty_row = pd.DataFrame(np.nan, index=[0], columns=pdb_ca.columns)
            empty_row["atom_name"] = "CA"
            empty_row["seqres"] = mapping["a_full"]
            full_df.append(empty_row)
            continue

        pdb_residue = pdb_dataframe[pdb_dataframe["residue_key"] == mapping["resi"]].copy()
        pdb_residue.loc[:, "seqres"] = mapping["a_full"]
        pdb_residue.loc[:, "aa"] = mapping["a_pdb"]
        full_df.append(pdb_residue)
    return pd.concat(full_df, axis=0, ignore_index=True)

def get_PDBDataFrame(pdb_id, chains, dir="PDB_db"):
    """Process PDB chains into DataFrames and save to .pkl files."""
    pdb_path = f"{dir}/structs/{pdb_id.upper()}.pdb"
    if not os.path.exists(pdb_path):
        print(f"Error: PDB file {pdb_id} not found in {pdb_path}.")
        return

    pdb_structure = PandasPdb().read_pdb(pdb_path).df["ATOM"]
    if pdb_structure.empty:
        print(f"Error: No ATOM records found in PDB file {pdb_id}.")
        return

    sequences = {}
    output_dir = f"{dir}/structs_per_chain/"
    os.makedirs(output_dir, exist_ok=True)

    # Extract sequences
    for record in SeqIO.parse(pdb_path, "pdb-seqres"):
        chain = record.id[-1]
        sequences[chain] = record.seq

    # Process chains
    for chain in chains:
        if chain not in sequences or not sequences[chain]:
            print(f"Warning: No sequence data for chain {chain} in PDB {pdb_id}. Skipping...")
            continue

        pdb_chain = pdb_structure[pdb_structure["chain_id"] == chain]
        if pdb_chain.empty:
            print(f"Warning: No ATOM data for chain {chain} in PDB {pdb_id}. Skipping...")
            continue

        output_file = f"{output_dir}/{pdb_id}_{chain}.pkl"
        if os.path.exists(output_file):
            print(f"Output file already exists: {output_file}. Skipping...")
            continue

        print(f"Processing {pdb_id}, chain {chain}.")
        full_df = put_full_sequence(pdb_chain, sequences[chain])
        if full_df is not None:
            pickle.dump(full_df, open(output_file, "wb"))
            print(f"Saved processed DataFrame for {pdb_id}, chain {chain} to {output_file}.")
        else:
            print(f"Failed to process chain {chain} in PDB {pdb_id}.")

def get_tasks(dir="PDB_db"):
    """Identify unprocessed chains from PDB files."""
    tasks_file = f"{dir}/all_pdbids_and_chains.txt"
    structs_dir = Path(f"{dir}/structs/")
    processed_dir = Path(f"{dir}/structs_per_chain/")

    if not os.path.exists(tasks_file):
        all_chains = set()
        for pdb_file in structs_dir.glob("*.pdb"):
            with open(pdb_file, "r") as f:
                for line in f:
                    if line.startswith("ATOM") and len(line) > 21 and line[13:15] == "CA":
                        all_chains.add(f"{pdb_file.stem}_{line[21]}")
        with open(tasks_file, "w") as f:
            f.write("\n".join(all_chains))
        print(f"Created tasks file with {len(all_chains)} chains.")

    all_chains = {line.strip() for line in open(tasks_file).readlines()}
    processed_chains = {file.stem for file in processed_dir.glob("*.pkl")}

    tasks = {}
    for chain in all_chains - processed_chains:
        pdb_id, chain_id = chain.split("_")
        tasks.setdefault(pdb_id, set()).add(chain_id)

    print(f"Found {len(processed_chains)} processed PDB files.")
    print(f"Found {len(tasks)} unprocessed PDB files.")
    return tasks


def process_pdb_task(pdb_id, chains, dir="PDB_db"):
    """Process a single PDB ID with its chains."""
    try:
        print(f"Processing {pdb_id} with chains: {', '.join(chains)}")
        get_PDBDataFrame(pdb_id, chains, dir)
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")


def run_parallel_tasks(dir="PDB_db"):
    """Run PDB processing tasks in parallel with progress tracking."""
    jobs = get_tasks()  # Get all unprocessed tasks
    print(f"Found {len(jobs)} PDB files to process.")

    from os import cpu_count
    max_workers = min(16, cpu_count())  # Dynamically determine workers based on system resources

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_pdb_task, pdb_id, jobs[pdb_id], dir): pdb_id
            for pdb_id in jobs
        }

        for future in tqdm(futures, desc="Processing PDB Files", total=len(futures)):
            pdb_id = futures[future]
            try:
                future.result()  # Wait for the task to complete
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
    
    jobs = get_tasks()  # Get all unprocessed tasks
    print(f"Found {len(jobs)} PDB files unprocessed.")

## 7. Put ANARCI annotation into antibodies dataframes prepared in the previous step
def realign_sequences(pdb_seq, anarci_, firstLetterException = False):
    seq_aa = []
    seq_i  = []
    for i,s_ in enumerate(anarci_):
        if len(s_)==0:
            continue
        seq_aa+=s_
        seq_i +=[i for i_ in range(len(s_))]
    al = kalign("".join(seq_aa),"".join(pdb_seq))
    
    n_anarci = 0
    n_pdb    = 0
    pdb_anarci_map = [None for i in pdb_seq]
 
    for i,[a_anarci,a_pdb] in enumerate(zip(*al)):
        if a_anarci!="-" and a_pdb!="-":#i!=0:
            pdb_anarci_map[n_pdb] = i
            if n_anarci == 0 and firstLetterException:
                n_pdb+=1
                n_anarci+=1
                continue
            if a_pdb!=a_anarci:
                return None
        if a_pdb!="-":
            n_pdb+=1
        if a_anarci!="-":            
            n_anarci+=1

    return pdb_anarci_map

def put_anarci_annotation(pdb_dataframe, fab_id, heavy_list, light_list, firstLetterException = False):
    pdb_id,chain,fab_type = fab_id
    if fab_type == "light":
        anarci_seq     = light_list[pdb_id.lower()+"_"+chain]
    else:
        anarci_seq     = heavy_list[pdb_id.lower()+"_"+chain]
    pdb_ca        = pdb_dataframe[pdb_dataframe["atom_name"] == "CA"]#["seqres"]
    pdb_seq = "".join(pdb_ca["seqres"])
    pdb_anarci_map = realign_sequences(pdb_seq, anarci_seq, firstLetterException)
    
    if pdb_anarci_map is None:
        return None
    
    pdb_anarci_map = [fab_type[0].upper()+str(i)  if i is not None else None for i in pdb_anarci_map]    
    pdb_dataframe["anarci"] = None
    
    for anarci_id, residue_number in zip(pdb_anarci_map, pdb_ca["residue_key"]):
        ids = pdb_dataframe["residue_key"] == residue_number
        pdb_dataframe.loc[ids,"anarci"] = anarci_id     
        
    return pdb_dataframe

def collect_jobs(anarci_list_heavy, anarci_list_light):
    """Collect jobs for heavy and light chain annotations."""
    jobs = []
    for anarci_id, _ in anarci_list_heavy.items():
        jobs.append((anarci_id[:4].upper(), anarci_id[-1], "heavy"))
    for anarci_id, _ in anarci_list_light.items():
        jobs.append((anarci_id[:4].upper(), anarci_id[-1], "light"))
    return jobs

def process_antibody_chain(pdb_id, chain, fab_type, input_dir, output_dir, heavy_list, light_list, strange_error_list, firstLetterException=True):
    """Process and annotate a single antibody chain."""
    pdb_path = f"{input_dir}/{pdb_id}_{chain}.pkl"
    out_path = f"{output_dir}/{pdb_id}_{chain}_{fab_type}.pkl"

    # Skip if the input file does not exist or the output file already exists
    if not os.path.exists(pdb_path):
        #print(f"Input file does not exist: {pdb_path}")
        return False
    if os.path.exists(out_path):
        #print(f"Output file already exists: {out_path}")
        return True

    # Load the PDB data
    fab = pickle.load(open(pdb_path, 'rb'))

    # Annotate the data
    fab_annotated = put_anarci_annotation(fab, (pdb_id, chain, fab_type), heavy_list, light_list, firstLetterException)

    # If annotation failed, add to error list and return False
    if fab_annotated is None:
        #print(f"Annotation failed for {pdb_id} chain {chain} ({fab_type})")
        strange_error_list.add((pdb_id, chain, fab_type))
        return False

    # Save the annotated data
    # print(f"Saving annotated data to {out_path}")
    pickle.dump(fab_annotated, open(out_path, 'wb'))
    return True

def annotate_antibody_chains(anarci_list_heavy, anarci_list_light, dir = "PDB_db"):
    """Annotate antibody chains and save the results."""
    # Ensure the output directory exists
    
    input_dir = os.path.join(dir, "structs_per_chain")
    output_dir = os.path.join(dir, "structs_antibodies")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all jobs
    jobs = collect_jobs(anarci_list_heavy, anarci_list_light)

    # Initialize the error list
    strange_error_list = set()
    input_file_not_exists_count = 0

    # Process each job
    for pdb_id, chain, fab_type in tqdm(jobs, desc="Processing Antibody Chains", unit="job", leave=True, dynamic_ncols=True):
        if not process_antibody_chain(pdb_id, chain, fab_type, input_dir, output_dir, anarci_list_heavy, anarci_list_light , strange_error_list):
            input_file_not_exists_count += 1

    print(f"Number of input files that do not exist or annotation failed: {input_file_not_exists_count}")

    # Return the list of errors
    return strange_error_list

## 8. Find heavy/light chain fab pairs
def get_pdb_list(dir = "PDB_db"):
    list = [p.name[:4] for p in Path(f"{dir}/structs_antibodies/").glob(f"*.pkl")]
    print(f"Found {len(list)} PDB IDs.")
    return list
    
def get_fabs_pdbid(pdb_id = "1LK3", dir = "PDB_db"):
    fab_path = Path(f"{dir}/structs_antibodies/").glob(f"{pdb_id}*.pkl")
    fab_ids  = {"heavy":[],"light":[]}
    for struct_id in fab_path:
        pdb_id, _, fab_type = struct_id.name.rstrip(".pkl").split("_")
        fab_ids[fab_type].append(struct_id)
    return fab_ids

def get_pair_interface(path_light, path_heavy, threshold = 4.5):
    pdb_light = pickle.load(open(path_light,'rb'))
    pdb_heavy = pickle.load(open(path_heavy,'rb'))
    
    ### interface residues of heavy and light fab chains
    heavy_interface = list(range(32,39)) + list(range(44,50)) + list(range(85,95))
    light_interface = list(range(34,39)) + list(range(45,51)) + list(range(90,108))

    heavy_ids = ["H"+str(i) for i in heavy_interface]
    light_ids = ["L"+str(i) for i in light_interface]
    
    heavy_interface = pdb_heavy[pdb_heavy["anarci"].isin(heavy_ids)]
    light_interface = pdb_light[pdb_light["anarci"].isin(light_ids)]
    
    xyz_heavy = getxyz(heavy_interface)
    xyz_light = getxyz(light_interface)
    
    cd = distance.cdist(xyz_heavy,xyz_light)
    ids = np.where(cd<threshold)
    
    return len(set(ids[0]))+len(set(ids[1]))
    
def screen_fab_pairs(pdb_id, threshold = 4.5, dir = "PDB_db"):
    fab_path = get_fabs_pdbid(pdb_id, dir)
    contacts = {}
    for heavy_path in fab_path["heavy"]:
        for light_path in fab_path["light"]:
            n = get_pair_interface(path_light = light_path, path_heavy = heavy_path, threshold = threshold)
            ### cut off to select interface that interact with each other
            if n > 3:# 10:
                contacts[(light_path.name.rstrip(".pkl"), heavy_path.name.rstrip(".pkl"))] = n
    return contacts

def process_pdb_id(pdb_id, dir="PDB_db"):
    """
    Process a single PDB ID to identify heavy-light chain pairs.
    
    Parameters:
        pdb_id (str): The PDB ID to process.
        dir (str): Directory containing structure files.
    
    Returns:
        tuple: A tuple containing the PDB ID and its interacting pairs.
    """
    return pdb_id, screen_fab_pairs(pdb_id, dir=dir)

def find_fab_pairs(dir="PDB_db"):
    """
    Find and save all heavy-light chain pairs for all PDB IDs.
    
    Parameters:
        dir (str): Directory containing PDB structure files.
        output_file (str): File path to save the results.
    """
    output_file = os.path.join(dir, "fab_pairs.pkl")
    temp_file = os.path.join(dir, "fab_pairs_temp.pkl")

    if os.path.exists(output_file):
        print(f"Results already exist at {output_file}. Skipping processing.\n")
        return pickle.load(open(output_file, 'rb'))

    # Get all PDB IDs
    pdb_list = get_pdb_list(dir)

    # Initialize result dictionary
    fab_contacts = {}

    # Load intermediate results if available
    if os.path.exists(temp_file):
        print(f"Resuming from temporary file: {temp_file}")
        with open(temp_file, 'rb') as f:
            fab_contacts = pickle.load(f)
        # Remove already processed PDB IDs
        pdb_list = [pdb for pdb in pdb_list if pdb not in fab_contacts]

    # Use ThreadPoolExecutor for parallel processing
    try:
        with ThreadPoolExecutor() as executor:
            for pdb_id, contacts in tqdm(
                executor.map(lambda pdb_id: process_pdb_id(pdb_id, dir), pdb_list),
                total=len(pdb_list),
                desc="Processing PDB IDs",
                unit="PDB",
                dynamic_ncols=True, leave=True
            ):
                fab_contacts[pdb_id] = contacts
                # Save progress to temp file
                with open(temp_file, 'wb') as f:
                    pickle.dump(fab_contacts, f)
    except KeyboardInterrupt:
        print("Process interrupted. Progress saved to temporary file.")

    # Save the final results
    with open(output_file, 'wb') as f:
        pickle.dump(fab_contacts, f)
    print(f"Saved fab pairs to {output_file}\n")

    # Remove the temporary file after completion
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Temporary file {temp_file} removed.\n")
    
    return fab_contacts

## 9. Find antigens and corresponding interacting antibodies
def get_all_antigens_list(anarci_list_heavy=None, anarci_list_light=None, dir="PDB_db"):
    save_path = os.path.join(dir, "antigens_path_list.pkl")
    pdbid_save_path = os.path.join(dir, "antigen_pdbids.pkl")

    # Check if the antigens list already exists and load it
    if os.path.exists(save_path):
        #print(f"Loading existing antigens list from {save_path}...")
        with open(save_path, "rb") as f:
            pdb_ids = pickle.load(f)
        #print(f"Loaded {len(pdb_ids)} antigen_chains.")
        
        # Load PDB IDs if available
        if os.path.exists(pdbid_save_path):
            with open(pdbid_save_path, "rb") as f:
                antigen_pdbids = pickle.load(f)
            #print(f"Loaded {len(antigen_pdbids)} antigen PDB IDs.")
            return pdb_ids, antigen_pdbids
        return pdb_ids, []

    # Generate the list of antigen_chains
    fab_ids = set(anarci_list_heavy) | set(anarci_list_light)
    fab_ids = {f[:4].upper() + "_" + f[-1] for f in fab_ids}
    print(fab_ids)
    print(f"Found {len(fab_ids)} antibody chains.")
    
    # Extract first 4 letters from fab_ids for comparison
    fab_prefixes = {p[:4] for p in fab_ids}

    if "8H64" in fab_prefixes:
        print("8H64 found in fab_prefixes")

    # Store chains whose first 4 letters exist in fab_ids but first 6 is different
    pdb_ids = [p for p in Path(f"{dir}/structs_per_chain/").glob("*.pkl") 
               if p.name[:4] in fab_prefixes 
               and p.name[:6] not in fab_ids]
    antigen_pdbids = list({p.name[:4] for p in pdb_ids})
    print(f"Found {len(pdb_ids)} antigen chains.")
    
    # Save the list of antigen chains
    with open(save_path, "wb") as f:
        pickle.dump(pdb_ids, f)
    print(f"Saved antigens list to {save_path}.")

    # Save the list of antigen PDB IDs
    with open(pdbid_save_path, "wb") as f:
        pickle.dump(antigen_pdbids, f)
    print(f"Saved antigen PDB IDs to {pdbid_save_path}.")

    return pdb_ids, antigen_pdbids

def get_antigens_PDBID(pdb_id, anarci_list_heavy=None, anarci_list_light=None):
    all_antigens, _ = get_all_antigens_list(anarci_list_heavy, anarci_list_light, dir="PDB_db")
    list = [a for a in all_antigens if a.name[:4] == pdb_id]
    print(f"Found {len(list)} chains for PDB ID {pdb_id}.")
    return list

def find_antigen_contacts(pdb_id, dir="PDB_db", threshold=4.0, cdr=True, fab_contacts=None):
    """
    Find antigen contacts with all antibody chains.

    Parameters:
        pdb_id (str): PDB ID to process.
        dir (str): Directory containing structure files.
        threshold (float): Distance threshold for defining contacts.

    Returns:
        dict: A dictionary with contact results for each antigen chain.
    """
    # Collect all antibody chains with the same PDB ID prefix
    antibody_dir = Path(f"{dir}/structs_antibodies")
    all_antibody_chains = [
        fab_path for fab_path in antibody_dir.glob("*.pkl")
        if fab_path.name.startswith(pdb_id)
    ]

    if all_antibody_chains == []:
        print(f"No antibody chains found for PDB ID {pdb_id}.")
        return {}
    #print(f"Found {len(all_antibody_chains)} antibody chains for PDB ID {pdb_id}.")

    # Initialize results
    results = {}

    # Get antigen file paths
    antigens = get_antigens_PDBID(pdb_id, dir)

    if fab_contacts is None:
        fab_contacts = pickle.load(open(f"{dir}/fab_pairs.pkl", 'rb'))
    fab_pairs = fab_contacts[pdb_id]

    for antigen in antigens:
        antigen_path = Path(antigen)

        antigen_df = pickle.load(antigen_path.open('rb'))
        total_antigen_residues = antigen_df["residue_number"].nunique()
        if total_antigen_residues < 25:
            print(f"Skipping antigen {antigen_path.stem} with no more than 25 residues.")
            continue

        #print(f"Processing antigen chain {antigen_path.name}...")
        antigen_results = {"all_chain": set(), "light": set(), "heavy": set(), "pair": set()}
        chain_contacts = {}

        # Iterate through antibody chains
        for fab_path in all_antibody_chains:
            contact_data = test_contacts(antigen_path, fab_path, threshold, cdr)

            if contact_data["n_contacts"] > 0:
                residues = contact_data["contacting_residues"]
                antigen_results["all_chain"].update(residues)
                if contact_data["fab_type"] == "light":
                    antigen_results["light"].update(residues)
                elif contact_data["fab_type"] == "heavy":
                    antigen_results["heavy"].update(residues)
                chain_contacts[fab_path.stem] = residues

        # Check for heavy-light pairs
        combined_chains = set()
        for (light_chain, heavy_chain), _ in fab_pairs.items():
            combined_chains.add(light_chain)
            combined_chains.add(heavy_chain)
        
        for chain in combined_chains:
            if chain in chain_contacts:
                antigen_results["pair"].update(chain_contacts[chain])

        results[antigen.stem] = {
            "all_chain": sorted(antigen_results["all_chain"]),
            "light": sorted(antigen_results["light"]),
            "heavy": sorted(antigen_results["heavy"]),
            "pair": sorted(antigen_results["pair"])
        }

    return results


def test_contacts(antigen_path, fab_path, threshold=4.0, use_cdr=True):
    """
    Test contacts between antigen and antibody chain with optional ANARCI filtering.

    Parameters:
        antigen_path (Path): Path to the antigen pickle file.
        fab_path (Path): Path to the antibody chain pickle file.
        threshold (float): Distance threshold for defining contacts.
        use_cdr (bool): Whether to filter antibody residues using CDR (ANARCI regions).

    Returns:
        dict: A dictionary with contact information:
            - "n_contacts": Number of contacting residues.
            - "contacting_residues": List of contacting antigen residues (e.g., 131_GLU).
            - "fab_type": Type of the FAB chain ("light" or "heavy").
    """
    # Load antigen and antibody data
    antigen_df = pickle.load(antigen_path.open('rb'))
    if antigen_df is None:
        return {"n_contacts": 0, "contacting_residues": []}

    fab_df = pickle.load(fab_path.open('rb'))

    # Determine antibody type and ANARCI-defined interface residues
    fab_type = None
    if fab_path.name.split("_")[-1] == "light.pkl":
        fab_type = "light"
        if use_cdr:  # Only filter if use_cdr is True
            interface = ["L" + str(i) for i in list(range(23, 35)) + list(range(66, 72)) + list(range(89, 98))]
        else:
            interface = []
    elif fab_path.name.split("_")[-1] == "heavy.pkl":
        fab_type = "heavy"
        if use_cdr:  # Only filter if use_cdr is True
            interface = ["H" + str(i) for i in list(range(23, 35)) + list(range(51, 57)) + list(range(93, 102))]
        else:
            interface = []
    else:
        interface = []

    # Filter antibody residues based on ANARCI interface
    if interface:
        fab_df = fab_df[fab_df["anarci"].isin(interface)]

    # Get antibody coordinates
    xyz_fab = getxyz(fab_df)
    if xyz_fab.size == 0:
        return {"n_contacts": 0, "contacting_residues": [], "fab_type": fab_type}

    # Initialize set for contacting residues
    contacting_residues = set()

    # Group antigen residues by residue_number and check for contacts
    for residue_number, residue_df in antigen_df.groupby("residue_number"):
        xyz_residue = getxyz(residue_df)  # Get residue coordinates
        if np.any(distance.cdist(xyz_residue, xyz_fab) < threshold):
            # Combine residue_number and residue_name (e.g., 131_GLU)
            residue_name = residue_df['residue_name'].iloc[0]
            contacting_residues.add(f"{int(residue_number)}_{residue_name}")

    # Return the results
    return {
        "n_contacts": len(contacting_residues),
        "contacting_residues": sorted(contacting_residues),
        "fab_type": fab_type
    }


def store_antigen_contacts_csv(results, dir="PDB_db", threshold=4, cdr = True):
    """
    Save antigen contacts into four separate CSV files.

    Parameters:
        results (dict): Dictionary with antigen contact results.
        dir (str): Directory where CSV files will be stored.
    """
    import pandas as pd

    csv_results = {
        "pair_contacts": [],
        "all_chain_contacts": [],
        "light_chain_contacts": [],
        "heavy_chain_contacts": []
    }

    for antigen, data in results.items():
        # Save all-chain results
        if "all_chain" in data and data["all_chain"]:
            csv_results["all_chain_contacts"].append((antigen, ", ".join(data["all_chain"])))
        # Save light-chain results
        if "light" in data and data["light"]:
            csv_results["light_chain_contacts"].append((antigen, ", ".join(data["light"])))
        # Save heavy-chain results
        if "heavy" in data and data["heavy"]:
            csv_results["heavy_chain_contacts"].append((antigen, ", ".join(data["heavy"])))
        # Save pair results
        if "pair" in data and data["pair"]:
            csv_results["pair_contacts"].append((antigen, ", ".join(data["pair"])))

    # Write to CSV files
    for key, content in csv_results.items():
        if content:
            df = pd.DataFrame(content, columns=["antigen_chain", "Epitopes (resi_resn)"])
            if cdr:
                output_dir = Path(f"{dir}/epitopes_cdr")
            else:
                output_dir = Path(f"{dir}/epitopes_no_cdr")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = output_dir / f"{key}_{threshold}.csv"

            df.to_csv(output_path, index=False)
            print(f"Saved {key} to {output_path}")
            print(f"Number of antigens with epitopes for {key}: {len(df) - 1}\n")



def process_pdb_id(pdb_id, dir="PDB_db", threshold=4.0, cdr = True):
    """
    Wrapper function to process a single PDB ID and find antigen contacts.
    """
    try:
        #print(f"Processing PDB ID: {pdb_id}")
        return pdb_id, find_antigen_contacts(pdb_id, dir, threshold, cdr)
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return pdb_id, None

def process_all_pdb_ids(pdb_ids=None, dir="PDB_db", threshold=4.0, cdr = True, max_workers=16):
    """
    Process all PDB IDs in parallel and store results into CSV files.

    Parameters:
        pdb_ids (list): List of PDB IDs to process.
        dir (str): Directory containing structure files.
        threshold (float): Distance threshold for defining contacts.
        max_workers (int): Maximum number of worker threads to use.
    """

    results = {}

    if pdb_ids is None:
        ag, pdb_ids = get_all_antigens_list(dir=dir)

    print(f"Processing {len(pdb_ids)} PDB IDs.")
    print(f"Processing {len(ag)} antigens.")
    print(f"Using distance threshold of {threshold} Å.\n")
    if cdr:
        print("Processing epitopes with considering CDRs.")
        epitopes_dir = Path(f"{dir}/epitopes_cdr/")
    else:
        print("Processing epitopes without considering CDRs.")
        epitopes_dir = Path(f"{dir}/epitopes_no_cdr/")

    if epitopes_dir.exists() and all(
        (epitopes_dir / f"{key}_{threshold}.csv").exists() for key in [
            "pair_contacts", "all_chain_contacts", "light_chain_contacts", "heavy_chain_contacts"
        ]
    ):
        print("All epitope CSV files already exist. Skipping processing.\n")
        
        # Load and print the number of antigens for each CSV
        for key in ["pair_contacts", "all_chain_contacts", "light_chain_contacts", "heavy_chain_contacts"]:
            csv_path = epitopes_dir / f"{key}_{threshold}.csv"
            df = pd.read_csv(csv_path)
            print(f"Number of antigens with epitopes for {key}: {len(df) - 1}")
        print("\n")
        return
    

    # Parallelize processing of PDB IDs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdb_id, pdb_id, dir, threshold, cdr): pdb_id for pdb_id in pdb_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDB IDs", unit="PDB"):
            pdb_id, result = future.result()
            if result:
                results.update(result)
            else:
                print(f"Skipping PDB ID {pdb_id} due to an error.")

    # Store results in CSV files
    print("Storing results into CSV files...")
    store_antigen_contacts_csv(results, dir=dir, threshold=threshold, cdr=cdr)
    print("All results stored successfully!")

# Modified version of find_antigen_contacts to include heavy-light pairs
def find_antigen_contacts_v2(pdb_id, dir="PDB_db", threshold=4.0, cdr=False, fab_contacts=None):
    """
    Find antigen contacts with antibody heavy-light pairs.

    Parameters:
        pdb_id (str): PDB ID to process.
        dir (str): Directory containing structure files.
        threshold (float): Distance threshold for defining contacts.
        cdr (bool): Whether to filter antibody residues by CDR in test_contacts.
        fab_contacts (dict): Pre-loaded dictionary of fab_pairs. If None, loads from pickled file.

    Returns:
        dict: 
            Nested dict with structure:
                {
                    antigen_chain1: {
                        (light_chain, heavy_chain): {
                            "light": [...],
                            "heavy": [...],
                            "pair": [...]
                        },
                        ...
                    },
                    antigen_chain2: { ... },
                    ...
                }
    """

    # Directory with antibody chain .pkl files
    antibody_dir = Path(f"{dir}/structs_antibodies")
    
    # Collect .pkl files that match this PDB ID
    all_antibody_chains = [
        fab_path for fab_path in antibody_dir.glob("*.pkl")
        if fab_path.name.startswith(pdb_id)
    ]

    if not all_antibody_chains:
        print(f"[WARNING] No antibody chains found for PDB ID {pdb_id}.")
        return {}

    # Get antigen file paths
    antigens = get_antigens_PDBID(pdb_id, dir)
    if not antigens:
        print(f"[WARNING] No antigen chains found for PDB ID {pdb_id}.")
        return {}

    # Load fab_pairs if not provided
    if fab_contacts is None:
        with open(f"{dir}/fab_pairs.pkl", "rb") as f:
            fab_contacts = pickle.load(f)

    fab_pairs = fab_contacts.get(pdb_id, {})
    # Ensure fab_pairs is not empty
    if not fab_pairs:
        print(f"[WARNING] No Fab pairs found for PDB ID {pdb_id}.")
        return {}

    # Prepare the results data structure
    results = {}  # {antigen_stem: {(light_chain, heavy_chain): {"light":[], "heavy":[], "pair":[]}}}

    for antigen_path_str in antigens:
        antigen_path = Path(antigen_path_str)
        antigen_stem = antigen_path.stem  # e.g. '1A4J_A'
        
        # Load antigen
        antigen_df = pickle.load(antigen_path.open('rb'))
        if antigen_df is None or antigen_df.empty:
            print(f"[INFO] Empty or missing antigen data for {antigen_stem}. Skipping.")
            continue
        
        # Quick residue count check
        total_antigen_residues = antigen_df["residue_number"].nunique()
        if total_antigen_residues < 25:
            print(f"[INFO] Skipping antigen {antigen_stem} with <= 25 residues.")
            continue
        
        # Dictionary for this antigen's results
        results_for_antigen = {}

        # Process each (light, heavy) pair in fab_pairs
        for (light_chain, heavy_chain), _ in fab_pairs.items():
            # Find the corresponding .pkl files for the light and heavy chains
            light_path = next((p for p in all_antibody_chains if light_chain in p.name), None)
            heavy_path = next((p for p in all_antibody_chains if heavy_chain in p.name), None)

            if (light_path is None) or (heavy_path is None):
                print(f"[WARNING] Missing chain file for pair ({light_chain}, {heavy_chain}). Skipping.")
                continue

            # Calculate contacts for the light and heavy chains
            light_data = test_contacts(antigen_path, light_path, threshold=threshold, use_cdr=cdr)
            heavy_data = test_contacts(antigen_path, heavy_path, threshold=threshold, use_cdr=cdr)

            # Build the combined "pair" contact set (union of light + heavy)
            pair_contacts = set(light_data["contacting_residues"]) | set(heavy_data["contacting_residues"])

            results_for_antigen[(light_chain, heavy_chain)] = {
                "light": sorted(light_data["contacting_residues"]),
                "heavy": sorted(heavy_data["contacting_residues"]),
                "pair": sorted(pair_contacts)
            }

        # Add to final dictionary
        results[antigen_stem] = results_for_antigen

    return results

def store_antigen_contacts_csv_v2(results, dir="PDB_db", threshold=4, cdr=True):
    """
    Save antigen contacts into three separate CSV files: pair, heavy, and light.

    Expected structure of `results`:
        {
            antigen_chain1: {
                (light_chain, heavy_chain): {
                    "light": [...],
                    "heavy": [...],
                    "pair": [...]
                },
                ...
            },
            ...
        }

    Columns in CSV:
        - antigen_chain
        - antibody_chains (either a single chain or the (light, heavy) tuple)
        - Epitopes (resi_resn)
    """
    import pandas as pd

    # Prepare lists to accumulate rows for each output CSV
    pair_contacts = []
    light_contacts = []
    heavy_contacts = []

    # Iterate over each antigen chain in results
    for antigen_chain, pairs_dict in results.items():
        # pairs_dict: { (light_chain, heavy_chain): {"light": [...], "heavy": [...], "pair": [...]} }
        for (light_chain, heavy_chain), contact_data in pairs_dict.items():
            # contact_data has keys: "light", "heavy", "pair"

            # 1) Light CSV
            if contact_data["light"]:
                # Example row: (antigen_chain, "12E8_L_light", "123_GLU, 124_ARG, ...")
                light_contacts.append(
                    (
                        antigen_chain,
                        light_chain,  # store the name of the light chain
                        ", ".join(contact_data["light"])
                    )
                )

            # 2) Heavy CSV
            if contact_data["heavy"]:
                heavy_contacts.append(
                    (
                        antigen_chain,
                        heavy_chain,  # store the name of the heavy chain
                        ", ".join(contact_data["heavy"])
                    )
                )

            # 3) Pair CSV
            if contact_data["pair"]:
                # For 'antibody_chains', we can combine the pair into a single string
                pair_name = f"{light_chain}+{heavy_chain}"
                pair_contacts.append(
                    (
                        antigen_chain,
                        pair_name,
                        ", ".join(contact_data["pair"])
                    )
                )

    # Now create DataFrames and save to CSV
    # Decide on output directory: cdr vs no_cdr
    if cdr:
        output_dir = Path(f"{dir}/epitopes_cdr")
    else:
        output_dir = Path(f"{dir}/epitopes_no_cdr")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1) Pair CSV
    if pair_contacts:
        df_pair = pd.DataFrame(pair_contacts,
                               columns=["antigen_chain", "antibody_chains", "Epitopes (resi_resn)"])
        output_path = output_dir / f"pair_contacts_{threshold}.csv"
        df_pair.to_csv(output_path, index=False)
        print(f"[INFO] Saved pair contacts to: {output_path}")
        print(f"Number of rows in pair CSV: {len(df_pair)}\n")

    # 2) Light CSV
    if light_contacts:
        df_light = pd.DataFrame(light_contacts,
                                columns=["antigen_chain", "antibody_chains", "Epitopes (resi_resn)"])
        output_path = output_dir / f"light_chain_contacts_{threshold}.csv"
        df_light.to_csv(output_path, index=False)
        print(f"[INFO] Saved light contacts to: {output_path}")
        print(f"Number of rows in light CSV: {len(df_light)}\n")

    # 3) Heavy CSV
    if heavy_contacts:
        df_heavy = pd.DataFrame(heavy_contacts,
                                columns=["antigen_chain", "antibody_chains", "Epitopes (resi_resn)"])
        output_path = output_dir / f"heavy_chain_contacts_{threshold}.csv"
        df_heavy.to_csv(output_path, index=False)
        print(f"[INFO] Saved heavy contacts to: {output_path}")
        print(f"Number of rows in heavy CSV: {len(df_heavy)}\n")

def process_pdb_id_v2(pdb_id, dir="PDB_db", threshold=4.0, cdr=False):
    """
    Wrapper function to process a single PDB ID and find antigen contacts.

    Parameters:
        pdb_id (str): PDB ID to process.
        dir (str): Directory containing structure files.
        threshold (float): Distance threshold for defining contacts.
        cdr (bool): Whether to filter antibody residues using CDR.

    Returns:
        tuple: (pdb_id, results_dict) or (pdb_id, None) in case of an error.
    """
    try:
        print(f"Processing PDB ID: {pdb_id}")
        results = find_antigen_contacts_v2(pdb_id, dir=dir, threshold=threshold, cdr=cdr)
        return pdb_id, results
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return pdb_id, None


def process_all_pdb_ids_v2(
    pdb_ids=None, dir="PDB_db", threshold=4.0, cdr=False, max_workers=16
):
    """
    Process all PDB IDs in parallel and store results into CSV files.

    Parameters:
        pdb_ids (list): List of PDB IDs to process.
        dir (str): Directory containing structure files.
        threshold (float): Distance threshold for defining contacts.
        cdr (bool): Whether to filter antibody residues using CDR.
        max_workers (int): Maximum number of worker threads to use.

    Returns:
        None
    """
    results = {}

    # If no PDB IDs are provided, get them from the database
    if pdb_ids is None:
        ag, pdb_ids = get_all_antigens_list(dir=dir)

    print(f"Processing {len(pdb_ids)} PDB IDs.")
    if cdr:
        print("Processing epitopes considering CDR regions.")
        epitopes_dir = Path(f"{dir}/epitopes_cdr/")
    else:
        print("Processing epitopes without considering CDR regions.")
        epitopes_dir = Path(f"{dir}/epitopes_no_cdr/")

    # Check if CSVs already exist
    csv_keys = ["pair_contacts", "light_chain_contacts", "heavy_chain_contacts"]
    if epitopes_dir.exists() and all(
        (epitopes_dir / f"{key}_{threshold}.csv").exists() for key in csv_keys
    ):
        print("All epitope CSV files already exist. Skipping processing.\n")
        for key in csv_keys:
            csv_path = epitopes_dir / f"{key}_{threshold}.csv"
            df = pd.read_csv(csv_path)
            print(f"Number of antigens with epitopes for {key}: {len(df)}")
        return

    # Parallel processing of PDB IDs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_pdb_id_v2, pdb_id, dir, threshold, cdr): pdb_id
            for pdb_id in pdb_ids
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing PDB IDs",
            unit="PDB",
        ):
            pdb_id, result = future.result()
            if result:
                results.update(result)
            else:
                print(f"Skipping PDB ID {pdb_id} due to an error.")

    # Store results into CSV files
    print("Storing results into CSV files...")
    store_antigen_contacts_csv_v2(results, dir=dir, threshold=threshold, cdr=cdr)
    print("All results stored successfully!")


## 10. Epitope Processing
def process_epitope_data(dir="PDB_db", cdr=False, fasta = False):
    """
    Process all epitope CSV data files in the directory to:
    - Count total epitopes, total antigens, and unique residues.
    - Extract sequences in FASTA format after filtering unique residues.
    - Remove rows with less than 5 epitopes and save the updated CSV.

    Parameters:
        dir (str): Directory containing antigen PDB pickle files and epitope CSV files.
    """
    if cdr:
        epitope_dir = Path(f"{dir}/epitopes_cdr")
        processed_dir = Path(f"{dir}/processed_epitopes_cdr")
        processed_dir.mkdir(parents=True, exist_ok=True)
    else:
        epitope_dir = Path(f"{dir}/epitopes_no_cdr")
        processed_dir = Path(f"{dir}/processed_epitopes_no_cdr")
        processed_dir.mkdir(parents=True, exist_ok=True)

    for epitope_csv in epitope_dir.glob("*.csv"):
        csv_name = epitope_csv.stem
        processed_csv_path = processed_dir / f"{csv_name}.csv"

        print(f"Processing file: {epitope_csv}")

        # Load the epitope CSV file
        df = pd.read_csv(epitope_csv)

        # Filter out rows with less than 5 epitopes
        df['Epitope Count'] = df['Epitopes (resi_resn)'].apply(lambda x: len(x.split(", ")))
        df_filtered = df[df['Epitope Count'] >= 5].drop(columns=['Epitope Count'])

        # Print original and filtered number of antigens
        original_antigens = df['antigen_chain'].nunique()
        filtered_antigens = df_filtered['antigen_chain'].nunique()
        unique_epitopes = df['Epitopes (resi_resn)'].nunique()
        print(f"Original number of antigens: {original_antigens}")
        print("Filter the antigens with less than 5 epitopes.")
        print(f"Filtered number of antigens: {filtered_antigens}")
        print(f"Number of antigens with unique epitopes: {unique_epitopes}\n")

        if fasta:
            # Initialize counters
            total_epitopes = 0
            total_residues = 0
            total_antigens = df_filtered['antigen_chain'].nunique()  # Count unique antigen chains
            sequences = set()

            # Prepare FASTA content
            fasta_output = ""

            # Iterate through each antigen chain
            for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing Epitopes", unit="epitope"):
                pdb_chain = row['antigen_chain']  # e.g., "7XCZ_A"
                epitopes = row['Epitopes (resi_resn)'].split(", ")  # List of residue labels

                pdb_id = pdb_chain.split("_")[0]  # Extract PDB ID
                chain_id = pdb_chain.split("_")[1]  # Extract chain ID

                # Locate the corresponding antigen .pkl file
                antigen_file = Path(dir) / f"structs_per_chain/{pdb_id}_{chain_id}.pkl"

                if not antigen_file.exists():
                    print(f"Warning: Antigen file not found for {pdb_chain}")
                    continue

                # Load the antigen DataFrame
                with open(antigen_file, 'rb') as f:
                    antigen_df = pickle.load(f)

                # Filter unique residues based on 'residue_number'
                antigen_df_unique = antigen_df.drop_duplicates(subset=['residue_number'])

                # Extract and concatenate the sequence from the 'seqres' column
                sequence = "".join(antigen_df_unique['seqres'].dropna().values)

                # Add sequence to the set of unique sequences
                sequences.add(sequence)

                # Add to FASTA output
                fasta_output += f">{pdb_chain}\n{sequence}\n"

                total_residues += len(sequence)
                total_epitopes += len(epitopes)

            # Print counts
            print(f"Total Antigens: {total_antigens}")
            print(f"Total Epitopes: {total_epitopes}")
            print(f"Total Residues: {total_residues}")
            print(f"Total Number of Unique Sequences: {len(sequences)}\n")

            # Save the FASTA file
            if cdr:
                fasta_dir = Path(f"{dir}/antigen_sequences_cdr")
            else:
                fasta_dir = Path(f"{dir}/antigen_sequences_no_cdr")
            fasta_dir.mkdir(parents=True, exist_ok=True)
            fasta_path = fasta_dir / f"{csv_name}_sequences.fasta"

            if fasta_path.exists():
                print(f"FASTA file already exists in : {fasta_path}")
            else:
                with open(fasta_path, "w") as fasta_file:
                    fasta_file.write(fasta_output)
                print(f"FASTA sequences saved to {fasta_path}")

        # Save the filtered CSV file
        if processed_csv_path.exists():
            print(f"Processed file already exist in: {processed_csv_path}")
        else:
            df_filtered.to_csv(processed_csv_path, index=False)
            print(f"Filtered CSV saved to {processed_csv_path}")
        print("\n")

# 12. Combine Epitope Sequences
def combine_epitope_sequences(
    input_csv,
    output_csv,
    fasta_filename="sequences.fasta",
    dir="PDB_db"
):
    """
    Combine the epitopes for the antigens with same antigen sequences.
    
    1. Read `input_csv` from the `dir` folder into a dataframe.
    2. For each row, load the chain's sequence from {pdb_id}_{chain_id}.pkl using multithreading.
    3. Group by 'sequence'.
       - Keep only the first PDB chain in each group.
       - Merge (union) all epitopes across that sequence.
    4. Save CSV with columns: ['PDB chain', 'Epitopes (resi_resn)'].
    5. Also save each unique sequence in FASTA format to 'fasta_filename'.
    """

    # -------------------------------------------------------------------
    # Step 0: Read the input CSV
    # -------------------------------------------------------------------
    csv_path = Path(dir) / input_csv
    df_filtered = pd.read_csv(csv_path)

    # Check necessary columns
    required_cols = {"antigen_chain", "Epitopes (resi_resn)"}
    if not required_cols.issubset(df_filtered.columns):
        raise ValueError(f"CSV must contain columns {required_cols}")

    # We'll add a new column "sequence" (initially empty)
    df_filtered["sequence"] = None

    # -------------------------------------------------------------------
    # Step 1: Function to process each row in parallel
    # -------------------------------------------------------------------
    def load_sequence(idx_and_row):
        """
        Given (idx, row), load the .pkl file corresponding to that chain,
        extract the chain's sequence, and return (idx, sequence).
        """
        idx, row = idx_and_row
        pdb_chain = row["antigen_chain"]  # e.g., "7XCZ_A"

        # Split PDB id and chain id
        parts = pdb_chain.split("_")
        if len(parts) != 2:
            # If parsing fails, return None
            return (idx, None)
        pdb_id, chain_id = parts

        # Path to the .pkl file
        antigen_file = Path(dir) / f"structs_per_chain/{pdb_id}_{chain_id}.pkl"
        if not antigen_file.exists():
            print(f"Warning: Antigen file not found for {pdb_chain}")
            return (idx, None)

        # Load the antigen DataFrame and extract sequence
        try:
            with open(antigen_file, "rb") as f:
                antigen_df = pickle.load(f)
        except Exception as e:
            print(f"Error reading {antigen_file}: {e}")
            return (idx, None)

        # Remove duplicate residue entries
        antigen_df_unique = antigen_df.drop_duplicates(subset=["residue_number"])
        # Concatenate the sequence
        sequence = "".join(antigen_df_unique["seqres"].dropna().values)

        return (idx, sequence)

    # -------------------------------------------------------------------
    # Step 2: Use ThreadPoolExecutor to load sequences in parallel
    # -------------------------------------------------------------------
    futures = []
    with ThreadPoolExecutor() as executor:
        for idx, row in df_filtered.iterrows():
            futures.append(executor.submit(load_sequence, (idx, row)))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading sequences", unit="chain"):
            idx, seq = future.result()
            df_filtered.at[idx, "sequence"] = seq

    # -------------------------------------------------------------------
    # Step 3: Group by 'sequence' (union epitopes, choose first PDB)
    # -------------------------------------------------------------------
    def combine_epitopes(epitope_series):
        """
        Given a series of strings like:
            ["448_ASN, 449_TYR", "452_LEU, 455_LEU", ...]
        parse and merge them uniquely.
        """
        all_epitopes = []
        for e in epitope_series:
            if not isinstance(e, str):
                continue
            # Split by comma
            splitted = [item.strip() for item in e.split(",")]
            all_epitopes.extend(splitted)

        # Remove duplicates and empty entries
        all_epitopes = list(filter(None, all_epitopes))
        unique_epitopes = sorted(set(all_epitopes))
        return ", ".join(unique_epitopes)

    # - We pick the 'first' PDB chain from each unique sequence
    # - We combine all epitopes from that sequence
    # - The 'sequence' column is kept so we can still write FASTA
    df_combined = (
        df_filtered
        .groupby("sequence", dropna=False, as_index=False)
        .agg({
            "antigen_chain": "first",  # pick just one
            "Epitopes (resi_resn)": combine_epitopes,
        })
    )

    # -------------------------------------------------------------------
    # Step 4: Save the final CSV with only the required columns
    # -------------------------------------------------------------------
    # The user wants ONLY 'PDB chain' and 'Epitopes (resi_resn)'
    # (We have them in df_combined; 'sequence' is still there though.)
    final_df = df_combined[["antigen_chain", "Epitopes (resi_resn)"]].copy()
    output_csv_path = Path(dir) / output_csv
    final_df.to_csv(output_csv_path, index=False)

    print(f"\nFinal CSV saved to: {output_csv_path}")
    print(f"Number of unique sequences: {df_combined.shape[0]}")

    # -------------------------------------------------------------------
    # Step 5: Write each unique sequence to a FASTA file
    # -------------------------------------------------------------------
    fasta_path = Path(dir) / fasta_filename

    with open(fasta_path, "w") as fasta_file:
        for _, row in df_combined.iterrows():
            seq = row["sequence"]
            pdb_chain = row["antigen_chain"]
            # Skip empty or None sequences
            if not seq:
                continue

            # Write in FASTA format
            # Example:
            # >7XCZ_A
            # MKKLLLLVVAVSV...
            fasta_file.write(f">{pdb_chain}\n{seq}\n")

    print(f"FASTA file with unique sequences saved to: {fasta_path}")

def prepare_antigen_structures_from_csv(csv_path, dir="PDB_db"):
    """
    Extracts specific chains from PDB files and saves them into a new directory,
    along with their sequences in FASTA format. Skips already processed chains.
    """
    # File paths
    base_dir = Path(dir)
    csv_path = base_dir / csv_path
    source_dir = base_dir / "structs"
    dest_dir_pdb = base_dir / "antigen_structs"

    # Ensure destination directories exist
    os.makedirs(dest_dir_pdb, exist_ok=True)

    # Load PDB chain list from the CSV
    epitopes_df = pd.read_csv(csv_path)
    pdb_chain_list = epitopes_df.iloc[:, 0].unique()  # Unique PDB chains (e.g., '8YJ8_A')

    # Filter out already processed chains
    unprocessed_chains = [
        pdb_chain for pdb_chain in pdb_chain_list 
        if not (dest_dir_pdb / f"{pdb_chain}.pdb").exists()
    ]

    if not unprocessed_chains:
        print("All chains have already been processed. Nothing to do.")
        return

    # PDB Parser
    parser = PDBParser(QUIET=True)

    class ChainSelect(Select):
        """Custom PDBIO Select class for extracting a single chain."""
        def __init__(self, chain_id):
            self.chain_id = chain_id

        def accept_chain(self, chain):
            return chain.id == self.chain_id

    # Process each unprocessed PDB chain with tqdm progress bar
    for pdb_chain in tqdm(unprocessed_chains, desc="Processing PDB Chains", unit="chain"):
        if "_" not in pdb_chain:
            print(f"Invalid PDB chain format: {pdb_chain}")
            continue

        pdb_id, chain_id = pdb_chain.split("_")  # Split into PDB ID and chain
        source_path = source_dir / f"{pdb_id}.pdb"
        dest_path = dest_dir_pdb / f"{pdb_chain}.pdb"

        if source_path.exists():
            try:
                # Parse PDB file
                structure = parser.get_structure(pdb_id, str(source_path))

                # Check if the chain exists
                if chain_id not in [chain.id for chain in structure[0]]:
                    print(f"Chain {chain_id} not found in {pdb_id}")
                    continue

                # Save the selected chain
                io = PDBIO()
                io.set_structure(structure)
                io.save(str(dest_path), select=ChainSelect(chain_id))
                print(f"Extracted and saved: {pdb_chain}.pdb")
            except Exception as e:
                print(f"Error processing {pdb_chain}: {e}")
        else:
            print(f"Warning: PDB file not found for {pdb_id} at {source_path}")

    print(f"\nCompleted. Chain-specific PDB files saved to: {dest_dir_pdb}.")
    print(f"Processed {len(unprocessed_chains)} new chains.")
    print(f"Skipped {len(pdb_chain_list) - len(unprocessed_chains)} already existing chains.")

## 11. Cluster Representatives
# Step 1: Parse BLAST Results
def parse_blast_results(blast_result_path):
    """
    Parse BLAST results and map query-target pairs with alignment information.
    """
    print("Parsing BLAST results...")
    blast_mappings = defaultdict(list)
    with open(blast_result_path, "r") as file:
        for line in file:
            query, target, *rest = line.strip().split("\t")
            blast_mappings[query].append(target)
    print(f"Parsed {len(blast_mappings)} representative mappings from BLAST.")
    return blast_mappings

# Step 2: Map Epitopes to Representatives
def map_epitopes_to_representatives(epitope_csv_path, blast_mappings):
    """
    Map epitopes from the original antigen sequences to their cluster representatives.
    """
    print("Mapping epitopes to cluster representatives...")
    epitopes_df = pd.read_csv(epitope_csv_path)
    epitopes_dict = {
        row["antigen_chain"]: set(row["Epitopes (resi_resn)"].split(", "))
        for _, row in epitopes_df.iterrows()
    }

    mapped_epitopes = defaultdict(set)
    for rep, queries in blast_mappings.items():
        for query in queries:
            if query in epitopes_dict:
                mapped_epitopes[rep].update(epitopes_dict[query])

    print(f"Mapped epitopes to {len(mapped_epitopes)} representatives.")
    return mapped_epitopes

# Step 3: Prepare Final DataFrame
def prepare_final_dataframe(mapped_epitopes):
    """
    Prepare the final DataFrame for saving.
    """
    print("Preparing final DataFrame...")
    rows = []
    for rep, epitopes in mapped_epitopes.items():
        epitope_list = ", ".join(sorted(epitopes, key=lambda r: int(r.split("_")[0])))
        rows.append({"antigen_chain": rep, "Epitopes (resi_resn)": epitope_list})

    final_df = pd.DataFrame(rows)
    print(f"Final DataFrame prepared with {final_df.shape[0]} rows.")
    return final_df

# Step 4: Verify Row Count and Save
def save_final_epitopes(final_df, representative_fasta_path, final_output_path):
    """
    Verify the row count and save the final DataFrame.
    """
    print("Verifying row count and saving final CSV...")
    rep_count = sum(1 for line in open(representative_fasta_path) if line.startswith(">"))
    print(f"Number of representatives in FASTA: {rep_count}")
    print(f"Number of rows in final DataFrame: {final_df.shape[0]}")

    if rep_count != final_df.shape[0]:
        print("Warning: Mismatch between representative count and DataFrame rows!")

    final_df.to_csv(final_output_path, index=False)
    print(f"Final representative epitope file saved to: {final_output_path}")

# Main Workflow

def process_representative_epitopes(dir="PDB_db"):
    """
    End-to-end process for mapping epitopes and saving the final representative CSV.
    """
    blast_result_path = f"{dir}/antigen_sequences_no_cdr/cluster/blastp_results.tsv"
    epitope_csv_path = f"{dir}/processed_epitopes_no_cdr/pair_contacts_4.0.csv"
    representative_fasta_path = f"{dir}/antigen_sequences_no_cdr/cluster/DB_clu_rep.fasta"
    final_output_path = f"{dir}/processed_epitopes_no_cdr/representative_pair_contacts_4.0.csv"

    blast_mappings = parse_blast_results(blast_result_path)
    mapped_epitopes = map_epitopes_to_representatives(epitope_csv_path, blast_mappings)
    final_df = prepare_final_dataframe(mapped_epitopes)
    save_final_epitopes(final_df, representative_fasta_path, final_output_path)