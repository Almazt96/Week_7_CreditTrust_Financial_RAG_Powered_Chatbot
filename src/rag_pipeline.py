# Task_3: Hardened RAG Execution Pipeline (Clean Isolation Architecture)
# The data pipeline execution chain flows smoothly: preprocessing.py --> 
# indexer.py --> evaluate.py or query.py. Everything is tied together seamlessly under src/config.py
# Task_3: Fully Synchronized Hardened RAG Execution Pipeline
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
        """Initializes connection to the generated vector database infrastructure with dynamic collection detection."""
        try:
            # 1. Initialize the embedding weights
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            
            # 2. Open our direct storage client connection
            self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
            
            # 3. DYNAMIC CHECK: List all available collections inside your folder
            available_collections = [c.name for c in self.chroma_client.list_collections()]
            logger.info(f"Discovered collections on disk: {available_collections}")
            
            # Identify which collection actually contains your data records
            chosen_collection = "langchain"  # Fallback baseline
            for col_name in available_collections:
                col_obj = self.chroma_client.get_collection(name=col_name)
                # If this collection contains your 34,585 records, attach to it!
                if col_obj.count() > 0:
                    chosen_collection = col_name
                    logger.info(f"🎯 Found populated data layer inside collection: '{col_name}' ({col_obj.count()} chunks)")
                    break
            
            # Bind to the discovered data pool explicitly so self.collection exists!
            self.collection = self.chroma_client.get_or_create_collection(name=chosen_collection)
            
            self.hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
            
            # Authenticated Inference Client binding your environment token
            self.client = InferenceClient(
                model=self.hf_model,
                token=os.environ.get("HF_TOKEN")
            )
            
            logger.info(f"RAG Engine successfully synchronized to active collection target: {chosen_collection}")
        except Exception as e:
            logger.error(f"Failed initialization of RAG pipeline components: {e}")
            raise
        
    def query(self, query_text: str, k: int = 4) -> Tuple[str, List[str]]:
        """Executes full similarity retrieval and returns LLM summary text."""
        try:
            # Calculate raw embeddings from input string
            query_embedding = self.embeddings.embed_query(query_text)
            
            # Query directly against self.collection (no LangChain wrapper filters)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Extract plain text content directly from the returned array structure
            documents_list = Antiquated_docs = results.get("documents", [])
            context_chunks = documents_list[0] if documents_list else []
                
        except Exception as e:
            logger.error(f"Direct native vector query execution exception: {e}")
            context_chunks = []
            
        # If your vector database returned absolutely nothing, run a structural fallback pass
        if not context_chunks or len(context_chunks) == 0 or context_chunks[0] is None:
            try:
                logger.warning("Empty vector result space. Attempting a wide structural text fetch fallback.")
                fallback_data = self.collection.get(limit=k)
                context_chunks = fallback_data.get("documents", [])
            except Exception as severe_err:
                logger.critical(f"Total database collection layer failure: {severe_err}")
                return "I do not have enough information.", []

        if not context_chunks or context_chunks[0] is None:
            return "I do not have enough information.", []

        # Bind the text chunks together for prompt context
        context_str = "\n\n--- Context Chunk ---\n".join([str(chunk) for chunk in context_chunks])
        
# ... (Keep everything above this point in your query method exactly the same) ...

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
            # FIXED: Using chat_completion satisfies the conversational task requirement
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            # Extract text from the chat completion response object safely
            return response.choices[0].message.content.strip(), context_chunks
            
        except Exception as e:
            logger.error(f"Hugging Face Hub inference execution exception: {e}")
            return f"[ERROR] Generation failed: {str(e)}", context_chunks