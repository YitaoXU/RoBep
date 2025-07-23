from tqdm import tqdm

import torch
import h5py
from pathlib import Path

from bce.antigen.antigen import AntigenChain
from bce.utils.loading import load_epitopes_csv
from bce.data.data import create_datasets


if __name__ == "__main__":
    _, antigens, _ = load_epitopes_csv()
    for id, chain_id in tqdm(antigens, desc="Processing antigens"):
        try:
            antigen_chain = AntigenChain.from_pdb(chain_id=chain_id, id=id)
            embeddings, backbone_atoms, rsa, coverage_dict = antigen_chain.data_preparation(override=True)
            
            if embeddings.shape[0] != len(antigen_chain.sequence):
                print('Length of Embeddings does not match for ', id, chain_id)
            if backbone_atoms.shape[0] != len(antigen_chain.sequence):
                print('Length of Backbone Atoms does not match for ', id, chain_id)
            if rsa.shape[0] != len(antigen_chain.sequence):
                print('Length of RSA does not match for ', id, chain_id)
            #print(embeddings.shape, backbone_atoms.shape, rsa.shape)
        except Exception as e:
            print(f"Error processing {id}_{chain_id}: {str(e)}")
    
    datasets = create_datasets(verbose=True, force_rebuild=True)
