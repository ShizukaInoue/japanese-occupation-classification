# Japanese Occupation Classifier

A machine learning project for classifying Japanese occupations into sectors using BERT and pattern matching.

## Project Structure

```
Code/
├── japanese_occupation_classifier/
│   ├── __init__.py
│   ├── main.py
│   └── utils/
│       ├── data_processing.py
│       ├── pattern_matching.py
│       ├── prediction.py
│       ├── preprocessing.py
│       └── training.py
```

## Features

- Text preprocessing optimized for Japanese occupation names
- Pattern-based initial classification system
- BERT-based deep learning model (using cl-tohoku/bert-base-japanese-v3)
- Cross-validation with majority voting
- Confidence scoring for predictions
- Hierarchical classification (Primary/Secondary/Tertiary sectors)

## Classification Categories

### Primary Sector
- Agriculture (farming, horticulture, livestock, management)
- Fishing (marine, aquaculture, processing)
- Forestry (logging, management)
- Mining (extraction, quarrying)

### Secondary Sector
- Manufacturing (general, metal, machinery, textile, food, chemical)
- Construction (general, specialized, civil)

### Tertiary Sector
- Office Work (general, clerical, management)
- Commerce (retail, wholesale, business)
- Professional Services (medical, education, technical)

### Other
- Unemployed (no work, unable to work)
- Domestic (homemaker, family work)
- Student (general, training)

## Usage

1. **Data Preparation**
   ```python
   from japanese_occupation_classifier import create_training_data
   
   # Load and preprocess data
   training_df, unknown_df = create_training_data(df, "occupation_column")
   ```

2. **Training**
   ```python
   from japanese_occupation_classifier import train_and_evaluate
   
   # Train model and evaluate performance
   train_and_evaluate(
       input_file="data/your_data.xlsx",
       text_column="occupation",
       label_column="sector"
   )
   ```

3. **Prediction**
   ```python
   from japanese_occupation_classifier import predict_batch
   
   # Make predictions on new data
   predictions, confidences = predict_batch(
       texts=new_texts,
       model=loaded_model,
       tokenizer=loaded_tokenizer,
       label_encoder=loaded_encoder,
       device=device
   )
   ```

## Model Details

- Base Model: cl-tohoku/bert-base-japanese-v3
- Fine-tuning: Cross-validation with 5 folds
- Early Stopping: Based on validation loss
- Confidence Scoring: Ensemble averaging across folds

## Data Processing Pipeline

1. Text Preprocessing
   - Character normalization
   - Noise removal
   - Japanese-specific text cleaning

2. Pattern Matching
   - Initial classification using predefined patterns
   - Hierarchical category assignment

3. BERT Training
   - Data augmentation
   - Cross-validation
   - Majority voting

4. Prediction
   - Ensemble prediction
   - Confidence calculation
   - Unknown category handling

## Directory Structure Details

- `main.py`: Main training and evaluation pipeline
- `utils/`:
  - `preprocessing.py`: Text preprocessing functions
  - `pattern_matching.py`: Rule-based classification patterns
  - `data_processing.py`: Dataset creation and augmentation
  - `training.py`: Model training functions
  - `prediction.py`: Prediction utilities


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

