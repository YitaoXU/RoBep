from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]  # Root directory of the project
EMBEDDING_DIR = BASE_DIR / "PDB_db/embeddings"

DISK_DIR = Path("/disk18T3/Yitao/project_1")
FULL_EMBEDDING_DIR = DISK_DIR / "embeddings/protein"
