import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
import logging
import joblib
import os
from typing import List, Tuple

from .data_processing import create_tensor_dataset, augment_data
from .prediction import predict_batch

logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.3
MAX_LENGTH = 256

def train_epoch(model, train_loader, optimizer, scheduler, device):
    """
    Train model for one epoch.
    
    Args:
        model: BERT model
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to train on
        
    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, val_loader, device, confidence_threshold=0.8):
    """
    Evaluate model performance.
    
    Args:
        model: BERT model
        val_loader: DataLoader for validation data
        device: Device to evaluate on
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        tuple: (average loss, accuracy, average confidence)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    confidence_scores = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predictions = torch.max(probabilities, dim=-1)
            
            high_confidence_mask = confidence > confidence_threshold
            correct += ((predictions == labels) & high_confidence_mask).sum().item()
            total += high_confidence_mask.sum().item()
            total_loss += outputs.loss.item()
            confidence_scores.extend(confidence.cpu().tolist())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    return avg_loss, accuracy, avg_confidence

def train_model_fold(train_texts, train_labels, val_texts, val_labels, label_encoder, fold):
    """
    Train model on a single cross-validation fold.
    
    Args:
        train_texts: Training text samples
        train_labels: Training labels
        val_texts: Validation text samples
        val_labels: Validation labels
        label_encoder: Label encoder instance
        fold: Current fold number
        
    Returns:
        tuple: (trained model, tokenizer)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        hidden_dropout_prob=DROPOUT_RATE,
        attention_probs_dropout_prob=DROPOUT_RATE,
        classifier_dropout=DROPOUT_RATE
    ).to(device)
    
    train_dataset = create_tensor_dataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = create_tensor_dataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=len(train_loader) * EPOCHS
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    for epoch in range(EPOCHS):
        logger.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, avg_confidence = evaluate(model, val_loader, device)
        
        logger.info(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        logger.info(f"Average Confidence: {avg_confidence:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }
            patience_counter = 0
            logger.info(f"New best model for fold {fold + 1} saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping for fold {fold + 1}")
                break
    
    model.load_state_dict(best_model_state['model_state_dict'])
    return model, tokenizer

def cross_validate_model_with_majority_vote(
    df: pd.DataFrame,
    n_splits: int = 5,
    text_column: str = 'text',
    label_column: str = 'label'
) -> Tuple[List[str], LabelEncoder]:
    """
    Perform k-fold cross-validation with majority voting.
    
    Args:
        df: Input DataFrame
        n_splits: Number of cross-validation folds
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Tuple containing:
            - List of predictions
            - Fitted LabelEncoder instance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Save label encoder for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    
    # Data augmentation
    texts, labels_encoded = augment_data(texts, labels_encoded)
    labels_encoded = np.array(labels_encoded)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_fold_predictions = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(texts, labels_encoded)):
        logger.info(f"\nFold {fold + 1}/{n_splits}")
        
        train_texts = [texts[i] for i in train_index]
        val_texts = [texts[i] for i in val_index]
        train_labels = labels_encoded[train_index]
        val_labels = labels_encoded[val_index]
        
        model, tokenizer = train_model_fold(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            label_encoder=label_encoder,
            fold=fold
        )
        
        # Save models and tokenizers
        model_save_path = f'models/model_fold_{fold}'
        tokenizer_save_path = f'models/tokenizer_fold_{fold}'
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(tokenizer_save_path)
        
        # Get predictions for this fold
        fold_preds, _ = predict_batch(
            texts=df[text_column].tolist(),
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            device=device
        )
        all_fold_predictions.append(fold_preds)
    
    # Aggregate predictions using majority voting
    aggregated_predictions = []
    num_samples = len(df)
    
    for i in range(num_samples):
        sample_preds = [
            all_fold_predictions[fold][i] 
            for fold in range(n_splits) 
            if all_fold_predictions[fold][i] is not None
        ]
        
        if sample_preds:
            pred = max(set(sample_preds), key=sample_preds.count)
        else:
            pred = None
        
        aggregated_predictions.append(pred)
    
    return aggregated_predictions, label_encoder 