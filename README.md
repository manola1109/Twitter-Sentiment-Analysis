# Twitter Hate Speech Detection

## Project Overview
This project aims to detect hate speech in tweets by classifying them as either containing racist/sexist content (label 1) or not (label 0). The implementation includes three different approaches with varying levels of complexity and performance.

## Repository Structure
```
.
├── README.md                    # Project documentation
├── requirements.txt             # Required packages
├── hate_speech_detection.py     # Basic ML approach
├── improved_model.py           # Advanced features model
├── bert_model.py               # BERT-based model
├── ensemble_approach.py        # Ensemble model (recommended)
├── train.csv                   # Training data (provided)
├── test_tweets.csv            # Test data (provided)
└── test_predictions.csv       # Output predictions
```

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Basic Model
```bash
python hate_speech_detection.py
```

### Running the Improved Model
```bash
python improved_model.py
```

### Running the BERT Model
```bash
python bert_model.py
```

### Running the Ensemble Model (Recommended)
```bash
python ensemble_approach.py
```

## Approach

### Data Preprocessing
- Lowercase conversion
- URL and mention removal
- Hashtag handling
- Character normalization
- Special character removal
- Stopword removal
- Lemmatization

### Feature Engineering
- Text vectorization (TF-IDF, Count vectors)
- Tweet length and word count
- Hate term detection
- Profanity detection
- Uppercase ratio analysis

### Models
1. Basic ML Pipeline: XGBoost with GridSearchCV optimization
2. Improved Model: Enhanced preprocessing and feature engineering
3. BERT Model: Fine-tuned BERT for sequence classification
4. Ensemble Model: Combination of XGBoost, Logistic Regression, SVM, Random Forest, and Naive Bayes

## Model Performance
The models were evaluated using the F1-Score, with the following approximate results on validation data:

| Model           | F1-Score    |
|----------------|-------------|
| Basic XGBoost  | ~0.75-0.78  |
| Improved Model | ~0.78-0.80  |
| BERT           | ~0.82-0.85  |
| Ensemble       | ~0.79-0.82  |

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- nltk
- transformers (for BERT model)
- torch (for BERT model)

## Notes
- The ensemble model offers the best balance between performance and resource requirements
- The BERT model provides the highest accuracy but requires more computational resources
- For quick testing, the basic model is recommended
- For production deployment, the ensemble model is recommended

## Contact
For any questions or feedback regarding this project, please open an issue in the repository.
