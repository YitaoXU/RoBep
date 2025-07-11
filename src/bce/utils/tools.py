import torch

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def get_chain_organism(pdb_id, chain_id):
    """
    Use RCSB PDB API to get the organism of the chain.
    """
    import requests
    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    res = requests.get(entry_url)
    if res.status_code != 200:
        return "Unknown"
    entry_data = res.json()
    
    # Find the polymer_entity_id of the chain
    chain_to_entity = {}
    for entity_id in entry_data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", []):
        entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.lower()}/{entity_id}"
        entity_res = requests.get(entity_url)
        if entity_res.status_code != 200:
            continue
        entity_data = entity_res.json()
        chains = entity_data.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", [])
        for c in chains:
            chain_to_entity[c] = entity_id
        if chain_id in chains:
            organism = entity_data.get("rcsb_entity_source_organism", [{}])[0].get("scientific_name", "Unknown")
            return organism
    return "Unknown"

def classify_antigen(organisms):
    for org in organisms:
        if "virus" in org.lower() or "coronavirus" in org.lower():
            return "viral"
        elif "homo sapiens" in org.lower():
            return "human"
        elif "bacteria" in org.lower() or "bacillus" in org.lower():
            return "bacterial"
        elif "tumor" in org.lower() or "cancer" in org.lower():
            return "tumor"
    return "other"