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
```bash
git clone https://github.com/YitaoXU/RoBep.git
cd RoBep

conda create -n RoBep python=3.10 -y
conda activate RoBep

# Install PyTorch and basic dependencies first
pip install -r requirements.txt

# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# Install the package in development mode
pip install -e .
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
You can see our [tutorials](notebooks/example.ipynb) to learn how to use RoBep.

```bash
prediction_results = antigen_chain.predict(
    device_id=0,
    radius=18.0,
    k=7,
    encoder="esmc",
    verbose=True,
    use_gpu=False
)
```
## Evaluation
```bash
python -u main.py --mode eval --model_path models/RoBep/20250626_110438/best_mcc_model.bin --radius 18.0 --k 7

```

## Training
```bash
# Data Preparation (1 hour)
python data_preparation.py

# Training
python main.py --mode train
```


<!-- ### User friendly website
```bash
conda activate RoBep
pip install -r src/bce/website/requirements.txt

cd src/bce/website

python run_server.py --host 0.0.0.0 --port 8000
``` -->
