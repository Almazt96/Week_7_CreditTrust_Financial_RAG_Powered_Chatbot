# Chunking &amp; embedding scripts
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --- Step 1: Stratified Sampling Pipeline ---
def extract_stratified_sample(df: pd.DataFrame, target_sample_size: int = 12000) -> pd.DataFrame:
    """
    Extracts a balanced, stratified sample from the dataset maintaining exact
    proportional class distributions across the 'Standardized_Product' category.
    """
    print("--- Step 1: Executing Stratified Sampling ---")
    
    # Calculate sampling fraction safely
    total_records = len(df)
    test_frac = target_sample_size / total_records
    
    if test_frac >= 1.0:
        print(f"Dataset size ({total_records}) is smaller than target sample size. Using full dataset.")
        return df.copy()

    # Stratify split using sklearn on our primary mapped product column
    _, df_stratified = train_test_split(
        df,
        test_size=test_frac,
        stratify=df['Standardized_Product'],
        random_state=42  # For reproducible seeding
    )
    
    print(f"Successfully sampled {len(df_stratified):,} records.")
    print("Sample Class Distribution Summary:")
    print(df_stratified['Standardized_Product'].value_counts(normalize=True))
    print("-" * 50)
    
    return df_stratified.copy()


# --- Step 2 & 3: Recursive Chunking & Embedding Generation Loop ---
def process_chunks_and_index(df_sample: pd.DataFrame, db_path: str = "./chroma_db"):
    """
    Processes complaints through Recursive Text Chunking, ties appropriate 
    metadata dict contexts to splits, and loads them into a local ChromaDB collection.
    """
    print("\n--- Steps 2-4: Chunking, Embedding & Vector Store Indexing ---")
    
    # # Initialize LangChain's recommended recursive splitter
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,
    #     chunk_overlap=50,
    #     length_function=len,
    #     separators=["\n\n", "\n", " ", ""]
    # )
    # 2. Explicitly use tiktoken for splitting to completely bypass PyTorch
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=500,
        chunk_overlap=50
    )
    # Initialize Local Chroma Client (Persistent Storage variant)
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Initialize HuggingFace Embedding pipeline utility directly inside Chroma
    # This automatically converts chunks to 384-dimensional dense vectors on execution
    hf_embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Get or create the vector target collection
    collection = chroma_client.get_or_create_collection(
        name="cfpb_complaints_idx",
        embedding_function=hf_embedding_func,
        metadata={"hnsw:space": "cosine"} # Set metric space optimization
    )
    
    # Batched tracking lists for efficient loading metrics
    chunked_texts = []
    chunked_metadatas = []
    chunked_ids = []
    
    global_chunk_counter = 0

    print("Chunking documents and parsing relational structural metadata maps...")
    for idx, row in df_sample.iterrows():
        narrative = str(row['Cleaned_Narrative'])
        
        # Split individual narrative into semantic overlapping boundaries
        chunks = text_splitter.split_text(narrative)
        
        # Collect unique identifier values from row columns safely
        # .get() avoids unexpected schema crashes if optional columns are absent
        comp_id = str(row.get('Complaint ID', idx))
        prod_cat = str(row.get('Standardized_Product', 'Unknown'))
        issue = str(row.get('Issue', 'Not Provided'))
        sub_issue = str(row.get('Sub-issue', 'Not Provided'))
        
        for chunk_idx, chunk_text in enumerate(chunks):
            # Skip empty slices
            if not chunk_text.strip():
                continue
                
            chunked_texts.append(chunk_text)
            
            # Pack rich structured metadata payload dictionary for multi-product upstream filters
            chunked_metadatas.append({
                "complaint_id": comp_id,
                "product_category": prod_cat,
                "issue": issue,
                "sub_issue": sub_issue,
                "chunk_index": chunk_idx
            })
            
            # Form unique composite Primary Key index IDs for each distinct chunk
            chunked_ids.append(f"{comp_id}_chunk_{chunk_idx}")
            global_chunk_counter += 1

    print(f"Total parent documents processed: {len(df_sample):,}")
    print(f"Generated a total of {global_chunk_counter:,} semantic sub-chunks.")
    
    # ChromaDB Batch Upload processing loop (Chroma limits maximum items per upsert call)
    print("\nUpserting chunks and generating dense 384-dim embeddings inside local ChromaDB...")
    batch_size = 5000
    for i in range(0, len(chunked_texts), batch_size):
        end_idx = i + batch_size
        collection.upsert(
            documents=chunked_texts[i:end_idx],
            metadatas=chunked_metadatas[i:end_idx],
            ids=chunked_ids[i:end_idx]
        )
        print(f" Indexed records {i:,} to {min(end_idx, len(chunked_texts)):,} successfully.")

    print(f"\n[SUCCESS] Vector store built and saved safely at: '{db_path}'")
    print(f"Total validation record index count inside collection: {collection.count()}")


# --- Execution Mock Script Implementation ---
if __name__ == "__main__":
    # Simulating the finalized clean dataframe output from your Task 1 step.
    mock_processed_data = {
        'Complaint ID': [f"ID_{i}" for i in range(100)],
        'Standardized_Product': ['Credit Card', 'Personal Loan', 'Savings Account', 'Money Transfer'] * 25,
        'Issue': ['Incorrect information on report', 'Fees charged', 'Unauthorized transfer', 'Billing dispute'] * 25,
        'Sub-issue': ['Account info incorrect', 'Unexpected fee', 'Hacking event', 'Card lost'] * 25,
        'Cleaned_Narrative': [
            "This is a long verified testing complaint block about credit cards and billing errors. "
            "The customer service was unresponsive when dealing with unexpected hidden fees. "
            "Need immediate correction regarding account status modifications."
        ] * 100
    }
    
    df_cleaned_task1 = pd.DataFrame(mock_processed_data)
    
    # Run the Task 2 pipeline steps sequentially 
    # (Using a small sample target limit for standard dummy evaluation validation)
    df_sampled = extract_stratified_sample(df_cleaned_task1, target_sample_size=40)
    process_chunks_and_index(df_sampled, db_path="./local_cfpb_chroma")
    
""" Key Architectural Choices Made Here:
The Layered Metadata Dictionary: Every chunk gets mapped with its respective tracking elements 
(complaint_id, product_category, etc.). This enables upstream multi-product filtering so you can later run complex vector queries constrained to single categories, such as:
where={"product_category": "Savings Account"}.

Native Embedding Automation: Instead of manually calling SentenceTransformer.encode(), the script 
uses Chroma's built-in SentenceTransformerEmbeddingFunction. This architecture encapsulates embedding generation seamlessly directly inside the storage upsert pipeline block.

Batch-Safe Upserting: Large datasets can encounter limits during bulk insert operations.
Processing vectors in explicitly constrained blocks (batch_size = 5000) safeguards against 
internal payload memory overflows. """