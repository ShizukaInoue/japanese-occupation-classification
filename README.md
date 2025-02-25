# Japanese Occupation Sector Classification System

## Overview
A machine learning system that automatically classifies Japanese occupations into sectors (Primary, Secondary, Tertiary, and Unemployed) using NLP techniques.

## 🌟 Key Features
- **Multi-level Classification**: Hierarchical categorization into main sectors, subcategories, and detailed occupation groups
- **Japanese Text Processing**: Specialized handling of Japanese characters, including:
  - Old-to-new character form conversion
  - Text normalization
  - Multiple character encoding support
- **ML Pipeline**:
  - Japanese BERT model (cl-tohoku/bert-base-japanese-v3)
  - 5-fold cross-validation
  - Ensemble learning with majority voting
  - Confidence scoring
- **Robust Data Augmentation**: Comprehensive Japanese text augmentation including:
  - Character variations (新字体/旧字体)
  - Kana conversions (hiragana/katakana)
  - Common abbreviations and business terms

## 🛠 Technical Architecture

### 1. Data Preprocessing
- Text cleaning and normalization
- Japanese character standardization
- Pattern-based initial classification
- Data augmentation for training enhancement

### 2. Pattern Matching System
Hierarchical pattern structure:
```
Sector
└── Category
    └── Subcategory
        └── Occupation Patterns
```

### 3. BERT Model Training
- **Model**: cl-tohoku/bert-base-japanese-v3
- **Validation**: 5-fold cross-validation
- **Features**:
  - Early stopping
  - Learning rate scheduling
  - Dropout regularization
  - Confidence thresholding

### 4. Prediction System
- Ensemble prediction with majority voting
- Confidence score calculation
- Multi-model aggregation

## 📊 Performance Metrics
- Cross-validated accuracy
- Per-category precision and recall
- Confidence score distribution
- Model ensemble performance

## 🔧 Requirements

### Python Dependencies
\`\`\`
pandas
numpy
torch
transformers
fugashi
jageocoder
jaconv
kyujipy
ja-cvu-normalizer
scikit-learn
joblib
\`\`\`

## 📝 Usage

### Basic Usage
1. Data Preparation:
\`\`\`python
df = pd.read_excel("your_data.xlsx")
df['cleaned_occupation'] = df['occupation_column'].apply(preprocess_text)
\`\`\`

2. Model Training:
\`\`\`python
predictions, label_encoder = cross_validate_model_with_majority_vote(train_df, n_splits=5)
\`\`\`

3. Making Predictions:
\`\`\`python
categories, confidences = predict_batch(texts, model, tokenizer, label_encoder, device)
\`\`\`

## 🔍 Project Structure
```
.
├── occupation_classification_generalized.ipynb  # Main notebook
├── models/                                     # Saved model files
│   ├── model_fold_0/
│   ├── model_fold_1/
│   └── ...
├── tokenizers/                                 # Saved tokenizer files
└── data/                                      # Data directory
```

## 📚 References
- BERT Japanese: https://github.com/cl-tohoku/bert-japanese
- Fugashi: https://github.com/polm/fugashi
- JaConverter: https://github.com/ikegami-yukino/jaconv

## 🔧 Function Reference

### Data Processing Functions

#### `preprocess_text(text: str) -> str`
Preprocesses Japanese occupation text with the following steps:
- Strips whitespace
- Normalizes characters using jaconv
- Converts half-width kana to full-width
- Removes parentheses and their contents
- Removes digits and punctuation
- Standardizes spaces

#### `create_tensor_dataset(texts, labels, tokenizer, max_length)`
Creates PyTorch datasets for BERT training:
- Tokenizes text with specified max length
- Adds special tokens ([CLS], [SEP])
- Creates attention masks
- Converts to PyTorch tensors
- Returns TensorDataset object

#### `augment_data(texts: List[str], labels: List[Any]) -> Tuple[List[str], List[Any]]`
Performs comprehensive Japanese text augmentation:
- Handles multiple character variations (新字体/旧字体)
- Converts between hiragana/katakana
- Manages special characters and business terms
- Includes common abbreviations and title variations
- Maintains label consistency
- Returns augmented texts and labels

### Training Functions

#### `train_epoch(model, train_loader, optimizer, scheduler, device)`
Handles single training epoch:
- Manages model training mode
- Processes batches of data
- Computes and backpropagates loss
- Updates model parameters
- Tracks metrics (loss, accuracy)
- Returns average loss and accuracy

#### `evaluate(model, val_loader, device, confidence_threshold=0.8)`
Evaluates model performance:
- Runs model in evaluation mode
- Computes predictions and confidence scores
- Applies confidence thresholding
- Calculates metrics:
  - Average loss
  - Accuracy
  - Average confidence
- Returns evaluation metrics

#### `train_model_fold(train_texts, train_labels, val_texts, val_labels, label_encoder, fold)`
Manages training for a single cross-validation fold:
- Initializes BERT model and tokenizer
- Creates datasets and dataloaders
- Sets up optimization components
- Implements early stopping
- Saves best model state
- Returns trained model and tokenizer

### Prediction Functions

#### `predict_batch(texts, model, tokenizer, label_encoder, device)`
Handles batch prediction:
- Validates and cleans input texts
- Processes texts in manageable batches
- Generates predictions and confidence scores
- Handles missing values
- Returns predictions and confidence scores

#### `cross_validate_model_with_majority_vote(df, n_splits=5)`
Implements complete cross-validation pipeline:
- Prepares data with label encoding
- Performs data augmentation
- Manages k-fold cross-validation
- Trains models for each fold
- Implements majority voting
- Returns aggregated predictions

### Model Constants
```python
MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.3
MAX_LENGTH = 256
```

### Training Parameters
- **Batch Size**: 32 samples per batch
- **Epochs**: 5 training cycles
- **Learning Rate**: 1e-5 with linear warmup
- **Early Stopping**: Patience of 3 epochs
- **Dropout Rate**: 0.3 for regularization
- **Max Sequence Length**: 256 tokens

