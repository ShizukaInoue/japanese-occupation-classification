import torch
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def predict_batch(texts, model, tokenizer, label_encoder, device, max_length=256):
    """
    Predict categories for a batch of texts.
    
    Args:
        texts: List of input texts
        model: BERT model
        tokenizer: BERT tokenizer
        label_encoder: Label encoder instance
        device: Device to run predictions on
        max_length: Maximum sequence length
        
    Returns:
        tuple: (predictions, confidence scores)
    """
    # Input validation and cleaning
    cleaned_texts = []
    valid_indices = []
    
    for i, text in enumerate(texts):
        if pd.isna(text) or str(text).strip() == "":
            continue
        cleaned_texts.append(str(text).strip())
        valid_indices.append(i)
    
    if not cleaned_texts:
        return [None] * len(texts), [None] * len(texts)
    
    # Process in smaller batches
    batch_size = 32
    all_categories = [None] * len(texts)
    all_confidences = [None] * len(texts)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i + batch_size]
            batch_indices = valid_indices[i:i + batch_size]
            
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**encodings)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predictions = torch.max(probabilities, dim=-1)
            
            batch_categories = label_encoder.inverse_transform(predictions.cpu().numpy())
            batch_confidences = confidence.cpu().numpy()
            
            for idx, (cat, conf) in enumerate(zip(batch_categories, batch_confidences)):
                original_idx = batch_indices[idx]
                all_categories[original_idx] = cat
                all_confidences[original_idx] = conf
    
    return all_categories, all_confidences 