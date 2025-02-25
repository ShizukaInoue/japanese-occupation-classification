import os
import logging
import numpy as np
import pandas as pd
import torch
import joblib
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from .data_processing import create_tensor_dataset, augment_data
from .prediction import predict_batch

# Constants
MODEL_NAME = 'cl-tohoku/bert-base-japanese-v2'
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
DROPOUT_RATE = 0.1
PATIENCE = 3

# Set up logging
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

def train_model_fold(
    train_texts, 
    train_labels, 
    val_texts, 
    val_labels, 
    label_encoder, 
    fold,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE
):
    """
    Train a model on a single cross-validation fold.
    
    Args:
        train_texts: Training text samples.
        train_labels: Training labels.
        val_texts: Validation text samples.
        val_labels: Validation labels.
        label_encoder: The label encoder instance.
        fold: The current fold number.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        dropout_rate: Dropout rate for model.
        
    Returns:
        Tuple containing the trained model and tokenizer.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Log class distribution in training and validation sets
    train_class_dist = pd.Series(train_labels).value_counts()
    val_class_dist = pd.Series(val_labels).value_counts()
    logger.info(f"Training set class distribution:\n{train_class_dist}")
    logger.info(f"Validation set class distribution:\n{val_class_dist}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Get the number of labels
    num_labels = len(label_encoder.classes_)
    logger.info(f"Number of classes: {num_labels}")
    logger.info(f"Classes: {label_encoder.classes_}")
    
    # Initialize model with dropout
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        hidden_dropout_prob=dropout_rate,
        attention_probs_dropout_prob=dropout_rate
    )
    model.to(device)
    
    # Create datasets
    train_dataset = create_tensor_dataset(train_texts, train_labels, tokenizer, label_encoder)
    val_dataset = create_tensor_dataset(val_texts, val_labels, tokenizer, label_encoder)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Track per-class metrics for training
        train_class_correct = {label: 0 for label in label_encoder.classes_}
        train_class_total = {label: 0 for label in label_encoder.classes_}
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Track per-class metrics
            for i, (pred, label) in enumerate(zip(predictions, labels)):
                true_label = label_encoder.inverse_transform([label.item()])[0]
                train_class_total[true_label] += 1
                if pred.item() == label.item():
                    train_class_correct[true_label] += 1
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        
        # Calculate per-class training accuracy
        train_class_accuracy = {}
        for label in label_encoder.classes_:
            if train_class_total[label] > 0:
                train_class_accuracy[label] = train_class_correct[label] / train_class_total[label]
            else:
                train_class_accuracy[label] = 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Track per-class metrics for validation
        val_class_correct = {label: 0 for label in label_encoder.classes_}
        val_class_total = {label: 0 for label in label_encoder.classes_}
        
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                # Track metrics
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                # Track per-class metrics
                for i, (pred, label) in enumerate(zip(predictions, labels)):
                    true_label = label_encoder.inverse_transform([label.item()])[0]
                    val_class_total[true_label] += 1
                    if pred.item() == label.item():
                        val_class_correct[true_label] += 1
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        # Calculate per-class validation accuracy
        val_class_accuracy = {}
        for label in label_encoder.classes_:
            if val_class_total[label] > 0:
                val_class_accuracy[label] = val_class_correct[label] / val_class_total[label]
            else:
                val_class_accuracy[label] = 0.0
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        logger.info(f"Per-class training accuracy: {train_class_accuracy}")
        logger.info(f"Per-class validation accuracy: {val_class_accuracy}")
        
        # Create confusion matrix for validation set
        val_preds_labels = label_encoder.inverse_transform(all_val_preds)
        val_true_labels = label_encoder.inverse_transform(all_val_labels)
        cm = confusion_matrix(val_true_labels, val_preds_labels, labels=label_encoder.classes_)
        cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
        logger.info(f"Validation confusion matrix:\n{cm_df}")
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation loss. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model state")
    
    return model, tokenizer

def cross_validate_model_with_majority_vote(
    df: pd.DataFrame,
    n_splits: int = 5,
    text_column: str = 'text',
    label_column: str = 'label',
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    dropout_rate: float = DROPOUT_RATE
) -> Tuple[List[str], LabelEncoder]:
    """
    Perform k-fold cross-validation with majority voting.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        n_splits (int): The number of cross-validation folds.
        text_column (str): The name of the text column.
        label_column (str): The name of the label column.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        dropout_rate (float): Dropout rate for model.
    
    Returns:
        Tuple[List[str], LabelEncoder]: A tuple containing the aggregated predictions and the fitted LabelEncoder.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Log class distribution in original data
    class_distribution = df[label_column].value_counts()
    logger.info(f"Original class distribution:\n{class_distribution}")
    
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # Fit the LabelEncoder on all labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    # Save the LabelEncoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    
    # Augment the data
    augmented_texts, augmented_labels = augment_data(texts, labels)
    
    # Log class distribution after augmentation
    augmented_df = pd.DataFrame({label_column: augmented_labels})
    augmented_distribution = augmented_df[label_column].value_counts()
    logger.info(f"Augmented class distribution:\n{augmented_distribution}")
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize a list to store predictions for each fold
    all_fold_predictions = []
    
    # Iterate through each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(augmented_texts, augmented_labels)):
        logger.info(f"Training fold {fold+1}/{n_splits}")
        
        # Split the data
        train_texts = [augmented_texts[i] for i in train_idx]
        train_labels = [augmented_labels[i] for i in train_idx]
        val_texts = [augmented_texts[i] for i in val_idx]
        val_labels = [augmented_labels[i] for i in val_idx]
        
        # Log training and validation set class distributions
        train_dist = pd.Series(train_labels).value_counts()
        val_dist = pd.Series(val_labels).value_counts()
        logger.info(f"Fold {fold+1} training set class distribution:\n{train_dist}")
        logger.info(f"Fold {fold+1} validation set class distribution:\n{val_dist}")
        
        # Train the model
        model, tokenizer = train_model_fold(
            train_texts, 
            train_labels, 
            val_texts, 
            val_labels, 
            label_encoder, 
            fold,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate
        )
        
        # Ensure the model parameters are contiguous
        model = model.to(device)
        
        # Save the model and tokenizer
        os.makedirs(f'models/fold_{fold}', exist_ok=True)
        model.save_pretrained(f'models/fold_{fold}/model')
        tokenizer.save_pretrained(f'models/fold_{fold}/tokenizer')
        
        # Make predictions on the validation set
        val_dataset = create_tensor_dataset(val_texts, val_labels, tokenizer, label_encoder)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        fold_predictions = []
        fold_true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                fold_predictions.extend(predictions)
                fold_true_labels.extend(labels.cpu().numpy())
        
        # Convert numeric predictions back to original labels
        fold_predictions = label_encoder.inverse_transform(fold_predictions)
        fold_true_labels = label_encoder.inverse_transform(fold_true_labels)
        
        # Log prediction distribution for this fold
        pred_dist = pd.Series(fold_predictions).value_counts()
        logger.info(f"Fold {fold+1} prediction distribution:\n{pred_dist}")
        
        # Calculate and log accuracy for this fold
        fold_accuracy = accuracy_score(fold_true_labels, fold_predictions)
        logger.info(f"Fold {fold+1} accuracy: {fold_accuracy:.4f}")
        
        # Log confusion matrix for this fold
        cm = confusion_matrix(fold_true_labels, fold_predictions, labels=label_encoder.classes_)
        cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
        logger.info(f"Fold {fold+1} confusion matrix:\n{cm_df}")
        
        # Store the predictions for this fold
        all_fold_predictions.append(fold_predictions)
    
    # Transpose the list of predictions to get predictions for each sample from each fold
    sample_predictions = list(zip(*all_fold_predictions))
    
    # Aggregate predictions using majority voting
    aggregated_predictions = []
    for sample_preds in sample_predictions:
        # Get the most common prediction for this sample
        most_common_pred = max(set(sample_preds), key=sample_preds.count)
        aggregated_predictions.append(most_common_pred)
    
    # Log final prediction distribution
    final_pred_dist = pd.Series(aggregated_predictions).value_counts()
    logger.info(f"Final aggregated prediction distribution:\n{final_pred_dist}")
    
    # Calculate and log final accuracy
    final_accuracy = accuracy_score(labels, aggregated_predictions[:len(labels)])
    logger.info(f"Final accuracy on original data: {final_accuracy:.4f}")
    
    # Log final confusion matrix
    final_cm = confusion_matrix(labels, aggregated_predictions[:len(labels)], labels=label_encoder.classes_)
    final_cm_df = pd.DataFrame(final_cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    logger.info(f"Final confusion matrix:\n{final_cm_df}")
    
    return aggregated_predictions, label_encoder 