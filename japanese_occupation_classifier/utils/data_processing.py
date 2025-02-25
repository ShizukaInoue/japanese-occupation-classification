import torch
from torch.utils.data import TensorDataset
import pandas as pd
import jaconv
import re
from typing import List, Any, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def create_tensor_dataset(texts, labels, tokenizer, max_length):
    """
    Creates PyTorch dataset from texts and labels.
    
    Args:
        texts: List of text samples
        labels: List of corresponding labels
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        
    Returns:
        TensorDataset: Dataset ready for training
    """
    # Tokenization
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
        add_special_tokens=True,
        return_attention_mask=True
    )
    
    # Extract tensors
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return TensorDataset(input_ids, attention_mask, labels_tensor)

def augment_data(texts: List[str], labels: List[Any]) -> Tuple[List[str], List[Any]]:
    """
    Augment training data with various text variations.
    
    Args:
        texts: List of original text samples
        labels: List of corresponding labels
        
    Returns:
        Tuple containing augmented texts and labels
    """
    if len(texts) != len(labels):
        raise ValueError("Length of texts and labels must match")
        
    augmented_texts = []
    augmented_labels = []
    
    # Character variations dictionary
    char_variations = {
        # Katakana/Hiragana variations
        'ヶ': ['ケ', 'か', 'カ', 'ケ', 'がつ'],
        'ヵ': ['カ', 'か', 'ケ'],
        'ヴ': ['ブ', 'ぶ', 'う゛'],
        'ヷ': ['バ', 'ば'],
        'ヸ': ['ビ', 'び'],
        'ヹ': ['ベ', 'べ'],
        'ヺ': ['ボ', 'ぼ'],
        
        # Common business terms
        '株': ['（株）', '㈱'],
        '有限': ['（有）', '㈲'],
        '合同': ['（同）', '㈾'],
        '財団': ['（財）', '㈶'],
        '社団': ['（社）', '㈳'],
        
        # Position/Title variations
        '課長': ['課長補佐', 'かちょう'],
        '部長': ['部長補佐', 'ぶちょう'],
        '社長': ['代表', 'しゃちょう'],
        '主任': ['主任補佐', 'しゅにん'],
        
        # Common occupation variations
        '會社員': ['会社員', 'かいしゃいん'],
        '商人': ['しょうにん', '商売人'],
        '農家': ['のうか', '農業'],
        '工場': ['こうじょう', '製造所'],
        '職工': ['しょっこう', '工員']
    }
    
    for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Augmenting data"):
        try:
            if pd.isna(text) or not str(text).strip():
                continue
                
            text = str(text).strip()
            variations = set()
            
            # Add original text
            variations.add(text)
            
            # Basic normalizations
            variations.add(jaconv.normalize(text))
            variations.add(jaconv.h2z(text, kana=True, ascii=True, digit=True))
            variations.add(jaconv.z2h(text, kana=True, ascii=True, digit=True))
            
            # Kana conversions
            variations.add(jaconv.hira2kata(text))
            variations.add(jaconv.kata2hira(text))
            
            # Character variation replacements
            for base_char, var_chars in char_variations.items():
                if base_char in text:
                    for var_char in var_chars:
                        variations.add(text.replace(base_char, var_char))
            
            # Remove spaces variations
            variations.add(re.sub(r'\s+', '', text))
            
            # Add variations with labels
            for variation in variations:
                if len(variation) > 0 and not variation.isspace():
                    augmented_texts.append(variation)
                    augmented_labels.append(label)
                    
        except Exception as e:
            logger.warning(f"Error processing text '{text}': {str(e)}")
            continue
    
    if len(augmented_texts) == 0:
        logger.warning("No valid augmented data generated")
        return texts, labels
        
    logger.info(f"Generated {len(augmented_texts)} samples from {len(texts)} original samples")
    
    return augmented_texts, augmented_labels 