# Task 1 - Modular text cleaning logic# Task 1 - Modular text cleaning logic
import os
import re
import logging
import pandas as pd

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definitive 4-product target vertical mapping
TARGET_PRODUCT_MAPPING = {
    'Credit card or prepaid card': 'Credit Card',
    'Credit card': 'Credit Card',
    'Consumer Loan': 'Personal Loan',
    'Student loan': 'Personal Loan',  # Added to ensure student loans map cleanly to personal loans
    'Checking or savings account': 'Savings Account',
    'Savings account': 'Savings Account',
    'Money transfer, virtual currency, or money service': 'Money Transfer'
}

def clean_complaint_narrative(text: str) -> str:
    """Cleans raw text fields by removing boilerplate, masks, and normalizing space."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Strip common redaction masks and boilerplate phrases
    text = re.sub(r'xxxx|xx/xx/\d{4}', '', text)
    text = re.sub(r'i am writing to file a complaint regarding', '', text)
    # Remove special characters, maintaining basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?\-\'"]', '', text)
    # Normalize multiple whitespace characters into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_filter_data(filepath: str) -> pd.DataFrame:
    """Loads dataset chunks, normalizes targeted categories, and cleans narratives."""
    logging.info(f"Initiating pipeline data load from: {filepath}")
    
    try:
        # Process via large chunk iterations to guarantee stability under memory pressure
        chunks = pd.read_csv(filepath, low_memory=False, chunksize=500000)
    except FileNotFoundError as e:
        logging.error(f"Critical execution error: Data file not found at {filepath}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected file I/O error reading {filepath}: {str(e)}")
        raise e

    filtered_chunks = []
    
    for chunk in chunks:
        # Match only rows where raw values are part of our targeted keys
        chunk_filtered = chunk[chunk["Product"].isin(TARGET_PRODUCT_MAPPING.keys())].copy()
        
        # Apply standardized taxonomy mapping
        chunk_filtered["product_clean"] = chunk_filtered["Product"].map(TARGET_PRODUCT_MAPPING)
        filtered_chunks.append(chunk_filtered)
        
    if not filtered_chunks:
        logging.warning("No records matched the target product criteria during chunk processing.")
        return pd.DataFrame()
        
    df = pd.concat(filtered_chunks, ignore_index=True)
    
    # Prune rows missing structural fields
    df = df.dropna(subset=['product_clean', 'Consumer complaint narrative']).copy()
    
    # Execute modular text transformation
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_complaint_narrative)
    
    # Strip records left completely empty after cleaning step
    df = df[df['cleaned_narrative'] != ''].copy()
    
    logging.info(f"Successfully processed and matched {len(df):,} clean target records.")
    return df

def save_canonical_artifact(df: pd.DataFrame, output_path: str) -> None:
    """Persists the final cleaned data to disk as a reproducible artifact."""
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        df.to_csv(output_path, index=False)
        logging.info(f"Canonical pipeline data artifact written to: {output_path}")
    except IOError as e:
        logging.error(f"Failed to write file artifact to {output_path}: {str(e)}")
        raise e

if __name__ == "__main__":
    # Point this to your real raw source data path
    raw_data_path = "../data/raw/complaints.csv"
    processed_artifact_path = "data/processed/filtered_complaints.csv"
    
    if os.path.exists(raw_data_path):
        processed_df = load_and_filter_data(raw_data_path)
        save_canonical_artifact(processed_df, processed_artifact_path)