# Task 1 - Modular text cleaning logic
import re
import pandas as pd

def clean_complaint_narrative(text: str) -> str:
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
    df = pd.read_csv(filepath)
    # Ensure exact mappings to target product verticals
    product_mapping = {
        'Credit card or prepaid card': 'Credit Card',
        'Credit card': 'Credit Card',
        'Consumer Loan': 'Personal Loan',
        'Checking or savings account': 'Savings Account',
        'Money transfer, virtual currency, or money service': 'Money Transfer'
    }
    df['product_clean'] = df['Product'].map(product_mapping)
    df = df.dropna(subset=['product_clean', 'Consumer complaint narrative'])
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_complaint_narrative)
    return df[df['cleaned_narrative'] != '']