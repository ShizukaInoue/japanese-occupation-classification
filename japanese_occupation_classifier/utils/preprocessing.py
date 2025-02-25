import re
import jaconv
import pandas as pd

def preprocess_text(text: str) -> str:
    """
    Comprehensive text preprocessing for Japanese occupation names.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Processed text
    """
    if pd.isna(text):
        return ''

    # Basic cleaning: strip leading/trailing whitespace
    text = str(text).strip()

    # Character normalization
    text = jaconv.normalize(text)
    # Convert half-width kana to full-width kana
    text = jaconv.h2z(text, kana=True, ascii=False, digit=False)

    # Remove noise
    text = re.sub(r'[\(\\（].*?[\)\\）]', '', text)  # Remove text within parentheses
    text = re.sub(r'[０-９0-9]+', '', text)        # Remove full-width and half-width digits
    text = re.sub(r'[^\w\s]', '', text)           # Remove punctuation
    text = re.sub(r'[\s　]+', ' ', text)          # Replace multiple spaces with a single space

    return text 