## ReCEP

### Environment
```bash
git clone https://github.com/YitaoXU/ReCEP.git
cd ReCEP

conda create -n ReCEP python=3.10 -y
conda activate ReCEP

# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121_full

pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric==2.6.1

pip install -r requirements.txt

pip install -e .
```

### Data Preparation
```python
from bce.antigen.antigen import AntigenChain

pdb_id = "5i9q"
chain_id = "A"

antigen_chain = AntigenChain.from_pdb(id=pdb_id, chain_id = chain_id)

embeddings, backbone_atoms, rsa = antigen_chain.data_preparation()
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


```bash
# website
pip install -r src/bce/website/requirements.txt

cd src/bce/website
conda activate ReCEP

python run_server.py --host 0.0.0.0 --port 8000
```
