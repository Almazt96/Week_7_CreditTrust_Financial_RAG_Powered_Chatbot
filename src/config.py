"""
Configuration module for the CrediTrust RAG pipeline.
Defines paths, model hyperparameters, and environmental variables.
"""

import os
from pathlib import Path
from typing import List

# Base Directory Routing
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Data and Vector Store Tracking Directories
RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
VECTOR_STORE_DIR: Path = BASE_DIR / "vector_store" / "local_cfpb_chroma"

# Ensure runtime structural directories exist cleanly
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters for Chunking Validation
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
CHUNK_SEPARATORS: List[str] = ["\n\n", "\n", " ", ""]

# Model Architectures
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"