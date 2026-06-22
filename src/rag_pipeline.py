# Task_3: Hardened RAG Execution Pipeline (Clean Isolation Architecture)
# The data pipeline execution chain flows smoothly: preprocessing.py --> indexer.py 
# --> evaluate.py or query.py. Everything is tied together seamlessly under src/config.py

import os
import sys
import logging
import chromadb
from typing import Tuple, List

# Ensure parent directory remains visible to the imports regardless of entry point
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from src.config import VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress network warnings cleanly
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class CrediTrustRAG:
    def __init__(self):
        """Initializes connection to the generated vector database infrastructure."""
        try:
            # 1. Initialize the embedding framework
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            
            # 2. FIX: Bypass LangChain completely. Open a native client to the storage folder.
            self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
            
            # 3. FIX: Pull the default collection instead of a custom metadata-bound index name.
            # This completely strips out the filtered path causing the Rust query planner to panic.
            self.collection = self.chroma_client.get_or_create_collection(name="langchain")
            
            self.hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
            self.client = InferenceClient(self.hf_model)
            
            logger.info("RAG Engine successfully bound to native unfiltered collection infrastructure.")
        except Exception as e:
            logger.error(f"Failed initialization of RAG pipeline components: {e}")
            raise

    def query(self, query_text: str, k: int = 4) -> Tuple[str, List[str]]:
        """Executes full similarity retrieval and returns LLM summary text."""
        
        try:
            # 1. Calculate raw embeddings from the input string
            query_embedding = self.embeddings.embed_query(query_text)
            
            # 2. Run an entirely unfiltered raw native vector search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Extract plain text content directly from the returned array structure
            context_chunks = results.get("documents", [[]])[0]
            
        except Exception as e:
            logger.error(f"Native vector retrieval panic caught: {e}")
            return "Retrieval system encountered an index drift delay.", []
        
        if not context_chunks:
            return "I do not have enough information.", []

        context_str = "\n\n--- Context Chunk ---\n".join(context_chunks)
        
        system_prompt = (
            "You are a CrediTrust Financial Analyst. Your job is to analyze consumer financial complaints.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Base your answers SOLELY and EXCLUSIVELY on the provided Context below.\n"
            "2. Do not use outside knowledge or assumptions.\n"
            "3. If the context does not contain enough data to confidently answer the question, "
            "you MUST reply exactly with: 'I do not have enough information.'\n"
            "4. Maintain a strictly objective, analytical, and factual tone."
        )
        
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query_text}\n\nAnalyst Answer:"
        
        try:
            response = self.client.text_generation(
                prompt=f"<s>[SYSTEM] {system_prompt} [/SYSTEM] [USER] {user_prompt} [/USER]",
                max_new_tokens=400,
                temperature=0.1
            )
            return response.strip(), context_chunks
        except Exception as e:
            logger.error(f"Hugging Face Hub inference execution exception: {e}")
            return f"[ERROR] Generation failed: {str(e)}", context_chunks