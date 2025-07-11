## ReCEP

```bash
conda create -n ReCEP python=3.10 -y
conda activate ReCEP

# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric==2.6.1

pip install -r requirements.txt

pip install -e .
```

### Data Preparation
```python
from bce.antigen.antigen import AntigenChain

pdb_id = "8urf"
chain_id = "A"

antigen = AntigenChain.from_pdb(id=pdb_id, chain_id = chain_id)

embeddings, backbone_atoms, rsa = antigen_chain.data_preparation()
```

```bash
# website
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6
pip install jinja2==3.1.2
pip install aiofiles==23.2.1

cd src/bce/website
conda activate ReCEP

python run_server.py --host 0.0.0.0 --port 8001
```
