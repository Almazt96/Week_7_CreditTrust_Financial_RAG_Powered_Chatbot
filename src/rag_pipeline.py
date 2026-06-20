# Core retrieval &amp; generator wrapper
""" (Ollama or Hugging Face Inference Client) so your system can query the database and generate answers efficiently 
without grinding your machine to a halt.

Here is the complete implementation protocol code covering the Retriever, Prompt Engineering, 
Generator, and Evaluation Suite.

1. Complete RAG Pipeline (rag_core.py)
This script establishes the Retriever Module, enforces the CrediTrust Financial Analyst Prompt, 
and integrates the Generator. """

import os
import chromadb
from sentence_transformers import SentenceTransformer
import requests  # Used for communicating with a local LLM server like Ollama
import ssl
import warnings

# 1. Force lower-level urllib, requests, and httpx to completely ignore SSL certificates
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HTTPX_VERIFY"] = "False"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# 2. Monkeypatch the urllib/ssl context defaults globally inside the runtime
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# 3. Patch the third-party requests library specifically 
try:
    import requests
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    # Force requests to default verify=False
    import functools
    requests.adapters.HTTPAdapter.send = functools.partialmethod(requests.adapters.HTTPAdapter.send, verify=False)
    requests.Session.request = functools.partialmethod(requests.Session.request, verify=False)
except ImportError:
    pass

# Silence local warning clutter
warnings.filterwarnings("ignore", category=UserWarning)

print("[SYSTEM] Network certificate checks successfully suppressed.")

# =====================================================================
# NOW PLACE ALL YOUR ORIGINAL IMPORTS BELOW THIS LINE
# =====================================================================
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Disable SSL verification globally for standard python libraries
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 2. Tell Hugging Face Hub / Transformers to ignore local SSL proxy issues
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

class CrediTrustRAG:
    def __init__(self, db_path="./production_chroma", collection_name="cfpb_complaints_idx"):
        """Initializes connection to the 1.37M chunk production database."""
        print("Loading Embedding Model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"Connecting to production vector store at {db_path}...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name=collection_name)
        
        # Ollama endpoint configuration (Default local setup running Llama 3 or Mistral)
        self.llm_url = "http://localhost:11434/api/generate"
        self.llm_model = "mistral" # Replace with 'mistral' or your chosen local model

    def retrieve_context(self, query_text, k=5):
        """Retriever Module: Embeds query and fetches top k=5 matching chunks."""
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Querying the massive 1.37M index
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            return []
        
        return results['documents'][0]

"""     def generate_answer(self, query_text, context_chunks):
        Prompt Engineering & Generator Integration.
        # Join the retrieved chunks into a solid block of context
        context_str = "\n\n--- Context Chunk ---\n".join(context_chunks)
        
        # Bulletproof System Prompt enforcing strict boundaries
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

        # Combine into standard instruction format
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Payload structure for Ollama API
        payload = {
            "model": self.llm_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0 # Force deterministic factual alignment
            }
        }
        
        try:
            response = requests.post(self.llm_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "Error: No response generated.")
        except requests.exceptions.ConnectionError:
            return "[ERROR] Local LLM server not running. Please start Ollama (`ollama run llama3`)."
        except Exception as e:
            return f"[ERROR] Generation failed: {str(e)}" """
from huggingface_hub import InferenceClient

def generate_answer(self, query_text, context_chunks):
    """Bypasses Ollama and uses Hugging Face serverless inference."""
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
        # Connects to Hugging Face's serverless pipeline for Mistral
        client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")
        
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

# Verification block to test a single query
if __name__ == "__main__":
    # Update paths to your 1.37M chunk DB path if different
    rag_system = CrediTrustRAG(db_path="./production_chroma", collection_name="cfpb_complaints_idx") 
    
    test_query = "What specific issues are consumers facing with international wire transfers?"
    ans, ctx = rag_system.query(test_query)
    print(f"\nAnswer:\n{ans}")

""" Task 4: Creating an Interactive Chat Interface
To empower non-technical support, compliance, and product managers, you will encapsulate your 
backend RAG pipeline into an interactive web UI.
Step-by-Step Implementation Protocol:
•	UI Framework Selection: Build the application interface in `app.py` utilizing Streamlit or 
Gradio. Streamlit is highly recommended for multi-column analytic dashboard layouts,
while Gradio fits perfectly into conversational interfaces.
•	Conversational & Input Components: Provide a clean, full-width text input box or a dedicated 
chat interface (`st.chat_input` or `gr.ChatInterface`). Implement a responsive submit button and 
a separate 'Clear Conversation' option to flush system state memory.
•	Source & Evidence Display: Crucial for enterprise trust! Design an expanding or dedicated 
secondary metadata container below or beside the AI answer. Loop over the retrieved source chunks, displaying the raw text narrative snippet, the actual `complaint_id`, and its corresponding sub-issue category.
•	Response Streaming (Recommended): Configure token-by-token generation streaming using g
enerator loops. This provides visual feedback and eliminates latency frustration for business stakeholders.
 """