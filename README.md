# RoBep
## Website
RoBep provides a user-friendly web interface accessible through [Hugging Face Spaces](https://huggingface.co/spaces/NielTT/RoBep). This interface allows you to easily predict epitope residues without any local installation.

### How to Use the Web Interface

1. **Accessing the Website**
   - Visit https://huggingface.co/spaces/NielTT/RoBep
   - Note: If you see a "restart" message, please wait for a few minutes as the service initializes

2. **Input Options**
   - Option 1: Enter a PDB ID and Chain ID
   - Option 2: Upload your own PDB file
   - Note: If you only have a protein sequence, you can obtain its predicted structure using AlphaFold3 [https://alphafoldserver.com/]

3. **View Results**
   - After processing (typically several seconds to 1-2 minutes), you'll see the prediction results
   - An interactive HTML visualization will be available for download

4. **Visualization Options**
   - Three display modes are available:
     * Predicted Epitopes: Shows the predicted epitope residues
     * Probability Gradient: Displays probability scores (darker color indicates higher probability)
     * Top-k Regions: Highlights predicted binding regions
   - Additional Features:
     * Use "Show Spheres" to visualize the selected spheres used in epitope prediction
     * Interact with the 3D structure using mouse controls (rotate, zoom, pan)

![RoBep Web Interface](figures/website.png)

## Environment Install

### Option 1: Manual Installation (Step by Step)
```bash
git clone https://github.com/YitaoXU/ReCEP.git
cd ReCEP

conda create -n ReCEP python=3.10 -y
conda activate ReCEP

# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121_full

pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric==2.6.1

pip install -r requirements.txt

pip install -e .
```

### Option 2: One-click Installation with Shell
```bash
git clone https://github.com/YitaoXU/ReCEP.git
cd ReCEP

# Make the script executable and run it
chmod +x install.sh
./install.sh

# Activate the environment
conda activate ReCEP
```

## Inference
### Data Preparation
```python
from bce.antigen.antigen import AntigenChain

pdb_id = "5i9q"
chain_id = "A"

antigen_chain = AntigenChain.from_pdb(id=pdb_id, chain_id = chain_id)

embeddings, backbone_atoms, rsa, coverage_dict= antigen_chain.data_preparation(radius=19.0)
```

### Epitope Prediction
You can see our [tutorials](notebooks/example.ipynb) to learn how to use ReCEP.

```bash
prediction_results = antigen_chain.predict(
    device_id=0,
    radius=19.0,
    k=7,
    encoder="esmc",
    verbose=True,
    use_gpu=False
)
```

## Training
```bash
# Data Preparation (1 hour)
python create_datasets.py

python main.py
```


<!-- ### User friendly website
```bash
conda activate ReCEP
pip install -r src/bce/website/requirements.txt

cd src/bce/website

python run_server.py --host 0.0.0.0 --port 8000
``` -->
