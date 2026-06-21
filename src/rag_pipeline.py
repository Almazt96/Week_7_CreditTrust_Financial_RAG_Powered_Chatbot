# Task_3:
# The data pipeline execution chain flows smoothly: preprocessing.py --> indexer.py 
# --> evaluate.py or query.py. Everything is tied together seamlessly under src/config.py
import os
import sys
import logging
from typing import Tuple, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from huggingface_hub import InferenceClient
from src.config import VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress network warnings cleanly
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class CrediTrustRAG:
    def __init__(self):
        """Initializes connection to the generated vector database infrastructure."""
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            self.vector_db = Chroma(
                persist_directory=str(VECTOR_STORE_DIR),
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
            self.hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
            logger.info(f"RAG Engine successfully bound to collection: {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed initialization of RAG pipeline components: {e}")
            raise

    def query(self, query_text: str, k: int = 4) -> Tuple[str, List[str]]:
        """Executes full similarity retrieval and returns LLM summary text."""
        # Retrieve context matching index structure
        docs = self.vector_db.similarity_search(query_text, k=k)
        context_chunks = [doc.page_content for doc in docs]
        
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
            client = InferenceClient(self.hf_model)
            response = client.text_generation(
                prompt=f"<s>[SYSTEM] {system_prompt} [/SYSTEM] [USER] {user_prompt} [/USER]",
                max_new_tokens=400,
                temperature=0.1
            )
            return response.strip(), context_chunks
        except Exception as e:
            logger.error(f"Hugging Face Hub inference execution exception: {e}")
            return f"[ERROR] Generation failed: {str(e)}", context_chunks