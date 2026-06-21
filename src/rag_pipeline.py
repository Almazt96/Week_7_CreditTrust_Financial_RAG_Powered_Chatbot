# Task_3:
import os
import ssl
import warnings
import functools

# 1. Force network and Hugging Face tools to bypass local proxy issues
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HTTPX_VERIFY"] = "False"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# 2. Global runtime SSL monkeypatch
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# 3. Patch third-party requests library globally
try:
    import requests
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    requests.adapters.HTTPAdapter.send = functools.partialmethod(requests.adapters.HTTPAdapter.send, verify=False)
    requests.Session.request = functools.partialmethod(requests.Session.request, verify=False)
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning)
print("[SYSTEM] Network certificate checks successfully suppressed.")

# =====================================================================
# SYSTEM IMPORTS
# =====================================================================
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

class CrediTrustRAG:
    def __init__(self, db_path="./production_chroma", collection_name="cfpb_complaints_idx"):
        """Initializes connection to the production database."""
        print("Loading Embedding Model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"Connecting to production vector store at {db_path}...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # Hugging Face Inference configuration
        self.hf_model = "mistralai/Mistral-7B-Instruct-v0.3"

    def retrieve_context(self, query_text, k=5):
        """Retriever Module: Embeds query and fetches top k matching chunks."""
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        if not results or not results.get('documents') or len(results['documents'][0]) == 0:
            return []
        
        return results['documents'][0]

    def generate_answer(self, query_text, context_chunks):
        """Generator Module: Uses Hugging Face serverless inference to generate response."""
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
                max_new_tokens=500,
                temperature=0.1
            )
            return response
        except Exception as e:
            return f"[ERROR] Hugging Face Inference failed: {str(e)}"
        
    def query(self, query_text):
        """Orchestrates the entire RAG pipeline."""
        context = self.retrieve_context(query_text, k=5)
        if not context:
            return "I do not have enough information.", []
        
        answer = self.generate_answer(query_text, context)
        return answer, context


# =====================================================================
# VERIFICATION EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    rag_system = CrediTrustRAG(db_path="./production_chroma", collection_name="cfpb_complaints_idx") 
    
    test_query = "What specific issues are consumers facing with international wire transfers?"
    
    # Corrected method execution call
    ans, ctx = rag_system.query(test_query)
    
    print(f"\nAnswer:\n{ans}")
    print("\nRetrieved Context Chunks:")
    for i, chunk in enumerate(ctx):
        print(f"--- Chunk {i+1} ---\n{chunk}\n")