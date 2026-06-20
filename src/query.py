import os
import chromadb
from sentence_transformers import SentenceTransformer
import ssl

# Disable SSL verification for Python's standard library
ssl._create_default_https_context = ssl._create_unverified_context

# Disable SSL verification for huggingface_hub / urllib3 / requests
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def initialize_rag_system(db_path="./production_chroma", collection_name="cfpb_complaints_idx"):
    """Initializes the vector store and embedding model."""
    if not os.path.exists(db_path):
        print(f"[ERROR] Vector store not found at {db_path}. Please run indexer.py first!")
        return None, None

    print("--- Initializing Chatbot System ---")
    print("Loading embedding model (matching indexer)...")
    embedding_model = SentenceTransformer("./local_model")
    print(f"Connecting to ChromaDB at: '{db_path}'...")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Use the collection name specified during your indexing phase.
    # LangChain defaults to "langchain" unless explicitly renamed.
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"[SUCCESS] Connected! Total records in collection: {collection.count()}\n")
        return collection, embedding_model
    except Exception as e:
        print(f"[ERROR] Could not find collection '{collection_name}'. Available collections: {chroma_client.list_collections()}")
        return None, None

def query_chatbot(query_text, collection, embedding_model, n_results=3):
    """Retrieves relevant complaints and prints them contextually."""
    # 1. Generate dense 384-dim embedding for the user query
    query_embedding = embedding_model.encode(query_text).tolist()

    # 2. Search the vector database for the top matching documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    print(f"\n{'='*30} RETRIEVED CONTEXT {'='*30}")
    
    # Handle empty or missing results gracefully
    if not results or not results['documents'] or len(results['documents'][0]) == 0:
        print("No highly relevant matching complaints found in the database.")
        print('='*79)
        return

    # 3. Present matched documents along with their metadata properties
    for i, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        product = metadata.get('Standardized_Product', 'Unknown Product')
        # Closer distance means higher semantic similarity
        similarity_score = max(0, 1 - distance) 
        
        print(f"\n[Match #{i+1}] | Product Category: {product} | Similarity: {similarity_score:.2%}")
        print(f"Complaint Snippet:\n{doc.strip()}")
        print("-" * 50)
        
    print('='*79 + '\n')

def main():
    # If your indexer named your collection something else, change it here
    collection, embedding_model = initialize_rag_system(
        db_path="./production_chroma", 
        collection_name="cfpb_complaints_idx" 
    )

    if not collection or not embedding_model:
        return

    print("Welcome to the CFPB Complaint RAG Assistant!")
    print("Type your questions below to query indexed complaints. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_query = input("User Question: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['exit', 'quit']:
                print("Shutting down assistant. Goodbye!")
                break

            query_chatbot(user_query, collection, embedding_model)

        except KeyboardInterrupt:
            print("\nShutting down assistant. Goodbye!")
            break

if __name__ == "__main__":
    main()