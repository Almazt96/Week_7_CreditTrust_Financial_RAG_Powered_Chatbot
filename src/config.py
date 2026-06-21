import os
from pathlib import Path
from typing import List

BASE_DIR: Path = Path(__file__).resolve().parent.parent

RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
VECTOR_STORE_DIR: Path = BASE_DIR / "vector_store" / "local_cfpb_chroma"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
CHUNK_SEPARATORS: List[str] = ["\n\n", "\n", " ", ""]
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME: str = "cfpb_complaints_idx"