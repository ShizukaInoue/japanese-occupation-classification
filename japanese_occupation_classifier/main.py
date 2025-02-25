import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple

from .utils.preprocessing import preprocess_text
from .utils.data_processing import create_tensor_dataset, augment_data
from .utils.training import train_model_fold, cross_validate_model_with_majority_vote
from .utils.prediction import predict_batch
from .utils.pattern_matching import create_training_data
from japanese_occupation_classifier import match_occupation, create_training_data, OCCUPATION_PATTERNS

# Constants
MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.3
MAX_LENGTH = 256

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

input_file = "data/your_data.xlsx" # Path to input Excel file
text_column="occupation" # Name of the column containing occupation text
label_column="sector" # Name of the column containing labels

"""
    input_file (str): Path to input Excel file
    text_column (str): Name of the column containing occupation text
    label_column (str): Name of the column containing labels
"""

# Step A: Load and preprocess data
df = pd.read_excel(input_file)
df['cleaned_occupation'] = df[text_column].apply(preprocess_text)

# Create training dataset that contains matched occupation text and label
training_df = create_training_data(df, text_column)

# For BERT training, you need two columns: 'text' and 'label'
# Let's create a DataFrame suitable for BERT
bert_training_df = training_df[['cleaned_occupation', 'main_category']].rename(columns={'cleaned_occupation': 'text', 'main_category': 'label'})

# Step B: Train a BERT model using the matched occupations from Step A and assess its performance. This consists of 4 steps.
# Step B1: Split data into training and test sets
train_df, test_df = train_test_split(bert_training_df, test_size=0.2, random_state=42, stratify=bert_training_df['label'])

# Step B2: Perform cross-validation on training data. Assume that the column name of the text is 'text' and the column name of the label is 'label'.
predictions, label_encoder = cross_validate_model_with_majority_vote(train_df, n_splits=5, 'text', 'label')

# Step B3: Use models from cross-validation to predict on test data to assess performance
# Load models and tokenizers as before

# Load the label encoder
label_encoder = joblib.load('label_encoder.joblib')

# Load the trained models and tokenizers
models = []
tokenizers = []
n_splits = 5  # Ensure this matches your cross-validation setting

for fold in range(n_splits):
    model_save_path = f'model_fold_{fold}'
    tokenizer_save_path = f'tokenizer_fold_{fold}'
    
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
print("Test Data Performance:")
print(classification_report(test_df['label'], df_predictions))

# Step C: Use the trained model to predict on unknown data

# Load the label encoder
label_encoder = joblib.load('label_encoder.joblib')

# Load the trained models and tokenizers
models = []
tokenizers = []
n_splits = 5  # Ensure this matches your cross-validation setting

for fold in range(n_splits):
    model_save_path = f'model_fold_{fold}'
    tokenizer_save_path = f'tokenizer_fold_{fold}'
    
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    model.eval()  # Set model to evaluation mode
    
    models.append(model)
    tokenizers.append(tokenizer)
    
# Make predictions on the dataset 
column_name = 'own column name'
all_texts = df[column_name].tolist()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
