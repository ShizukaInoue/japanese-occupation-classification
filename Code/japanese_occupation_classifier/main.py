import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
import os
from sklearn.utils import resample

from japanese_occupation_classifier.utils.preprocessing import preprocess_text
from japanese_occupation_classifier.utils.data_processing import create_tensor_dataset, augment_data
from japanese_occupation_classifier.utils.training import train_model_fold, cross_validate_model_with_majority_vote
from japanese_occupation_classifier.utils.prediction import predict_batch
from japanese_occupation_classifier.utils.pattern_matching import create_training_data

# Constants
MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.2
MAX_LENGTH = 256

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Create sample data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Path to input Excel file
input_file = "data/your_data.xlsx"
text_column = "occupation"  # Name of the column containing occupation text
label_column = "sector"  # Name of the column containing labels

# Step A: Load and preprocess data
logger.info("Loading and preprocessing data...")
df = pd.read_excel(input_file)
df['cleaned_occupation'] = df[text_column].apply(preprocess_text)

# Create training dataset that contains matched occupation text and label
logger.info("Creating training data...")
training_df, unknown_df = create_training_data(df, text_column)

# Log the distribution of categories after pattern matching
logger.info("Category distribution after pattern matching:")
category_counts = training_df['main_category'].value_counts()
for category, count in category_counts.items():
    logger.info(f"  {category}: {count}")

# For BERT training, you need two columns: 'text' and 'label'
# Let's create a DataFrame suitable for BERT
bert_training_df = training_df[['cleaned_occupation', 'main_category']].rename(columns={'cleaned_occupation': 'text', 'main_category': 'label'})

# Step B: Train a BERT model using the matched occupations from Step A and assess its performance. This consists of 4 steps.
# Step B1: Split data into training and test sets
logger.info("Splitting data into train and test sets...")
train_df, test_df = train_test_split(bert_training_df, test_size=0.2, random_state=42, stratify=bert_training_df['label'])

# Balance the training data by oversampling minority classes
# Get class distribution
class_counts = train_df['label'].value_counts()
max_samples = class_counts.max()
logger.info(f"Balancing training data. Max samples per class: {max_samples}")

# Oversample each class to have the same number of samples
balanced_dfs = []
for class_label in class_counts.index:
    class_df = train_df[train_df['label'] == class_label]
    # Only resample if we need more samples
    if len(class_df) < max_samples:
        resampled_df = resample(class_df, 
                               replace=True, 
                               n_samples=max_samples,
                               random_state=42)
        balanced_dfs.append(resampled_df)
    else:
        balanced_dfs.append(class_df)

# Combine all balanced classes
train_df = pd.concat(balanced_dfs)

# Log the distribution after balancing
logger.info("Category distribution after balancing:")
balanced_counts = train_df['label'].value_counts()
for category, count in balanced_counts.items():
    logger.info(f"  {category}: {count}")

# Step B2: Perform cross-validation on training data. Assume that the column name of the text is 'text' and the column name of the label is 'label'.
logger.info("Performing cross-validation...")
predictions, label_encoder = cross_validate_model_with_majority_vote(
    train_df, 
    n_splits=5, 
    text_column='text', 
    label_column='label',
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE
)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the label encoder
joblib.dump(label_encoder, 'models/label_encoder.joblib')

# Step B3: Use models from cross-validation to predict on test data to assess performance
logger.info("Evaluating model on test data...")

# Load the label encoder
label_encoder = joblib.load('models/label_encoder.joblib')

# Load the trained models and tokenizers
models = []
tokenizers = []
n_splits = 5  # Ensure this matches your cross-validation setting

for fold in range(n_splits):
    model_save_path = f'models/model_fold_{fold}'
    tokenizer_save_path = f'models/tokenizer_fold_{fold}'
    
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    model.eval()  # Set model to evaluation mode
    
    models.append(model)
    tokenizers.append(tokenizer)
    
# Predict on test data using majority voting
test_texts = test_df['text'].tolist()
df_predictions = []
df_confidences = []

for i in range(len(test_texts)):
    sample_text = test_texts[i]
    sample_preds = []
    sample_confs = []
    
    for model, tokenizer in zip(models, tokenizers):
        categories, confidences = predict_batch([sample_text], model, tokenizer, label_encoder, device)
        if categories[0] is not None:
            sample_preds.append(categories[0])
            sample_confs.append(confidences[0])
    
    if sample_preds:
        pred = max(set(sample_preds), key=sample_preds.count)
        conf = np.mean(sample_confs)
    else:
        pred = None
        conf = None
    
    df_predictions.append(pred)
    df_confidences.append(conf)

# Evaluate model performance on test data
test_df['predicted_category'] = df_predictions
logger.info("Test Data Performance:")
logger.info(classification_report(test_df['label'], df_predictions))

# Step C: Use the trained model to predict on unknown data
logger.info("Making predictions on all data...")

# Make predictions on the dataset 
all_texts = df['cleaned_occupation'].tolist()

# Initialize lists to store predictions and confidences
df_predictions = []
df_confidences = []

# Optimize prediction by batch processing
# Collect predictions from each model
all_model_preds = []
all_model_confs = []

for model, tokenizer in zip(models, tokenizers):
    categories, confidences = predict_batch(all_texts, model, tokenizer, label_encoder, device)
    all_model_preds.append(categories)
    all_model_confs.append(confidences)

# Perform majority voting for each sample
for i in range(len(all_texts)):
    sample_preds = [model_preds[i] for model_preds in all_model_preds if model_preds[i] is not None]
    sample_confs = [model_confs[i] for model_confs in all_model_confs if model_confs[i] is not None]
    
    if sample_preds:
        pred = max(set(sample_preds), key=sample_preds.count)
        conf = np.mean(sample_confs)
    else:
        pred = None
        conf = None
    
    df_predictions.append(pred)
    df_confidences.append(conf)

# Add predictions to the DataFrame
df['predicted_category'] = df_predictions
df['confidence'] = df_confidences

# Save results
df.to_excel('data/results.xlsx', index=False)
logger.info("Results saved to data/results.xlsx")

logger.info("Processing complete!")    
