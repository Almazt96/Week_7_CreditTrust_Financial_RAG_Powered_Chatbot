"""Task_2: # Chunking & embedding scripts
Core indexing and vector database ingestion pipeline component.
Handles document splitting, text chunk embedding transformations, and local storage persistence.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import sys
import os

# Adds the root project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Your existing imports continue below...
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR
)

# Setup diagnostic logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CrediTrustIndexer:
    """Encapsulates embedding mechanics and vector persistence strategies."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        """Initializes the vector transformation model components."""
        try:
            logger.info(f"Initializing embedding transformation matrix utilizing: {model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=CHUNK_SEPARATORS
            )
        except Exception as e:
            logger.error(f"Fatal initialization failure during model construction: {str(e)}")
            raise

    def process_and_index_complaints(
        self, 
        raw_complaints: List[Dict[str, Any]]
    ) -> Optional[Chroma]:
        """
        Transforms raw dictionaries into structured semantic vector database layouts.

        Args:
            raw_complaints (List[Dict[str, Any]]): Normalized consumer complaints containing text strings.

        Returns:
            Optional[Chroma]: A successfully persisted local Chroma instance, or None if errors materialize.
        """
        if not raw_complaints:
            logger.warning("Empty raw data array supplied. Ingestion pipeline aborted.")
            return None

        try:
            documents: List[Document] = []
            
            # Form standard structured document models
            for item in raw_complaints:
                text_content = item.get("consumer_complaint_narrative", "")
                if not text_content.strip():
                    continue
                
                # Encapsulate rich metadata tags for targeted stakeholder sorting
                metadata = {
                    "complaint_id": item.get("complaint_id", "UNKNOWN"),
                    "product_category": item.get("product", "UNSPECIFIED"),
                    "sub_issue": item.get("sub_issue", "GENERAL")
                }
                documents.append(Document(page_content=text_content, metadata=metadata))

            logger.info(f"Executing recursive text parsing on {len(documents)} parent source files...")
            split_chunks = self.splitter.split_documents(documents)
            logger.info(f"Generated {len(split_chunks)} semantic context nodes.")

            # Instant database generation and safe persistence mapping
            logger.info(f"Persisting vectors inside directory: {VECTOR_STORE_DIR}")
            vector_db = Chroma.from_documents(
                documents=split_chunks,
                embedding=self.embeddings,
                persist_directory=str(VECTOR_STORE_DIR),
                collection_name="cfpb_complaints_idx)]"
            )
            logger.info("[SUCCESS] Vector storage layer built and successfully synchronized.")
            return vector_db

        except Exception as e:
            logger.critical(f"Inversion layer error encountered while writing database entries: {str(e)}")
            return None
print("Indexing completed successfully!")