import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

print("Loading and exploring the data...")
# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_tweets.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Check for missing values and handle them
print("\nChecking for missing values...")
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Analyze class distribution in training data
print("\nClass distribution in training data:")
class_distribution = train_df['label'].value_counts()
print(class_distribution)
print(f"Class balance: {train_df['label'].value_counts(normalize=True)}")

# Text preprocessing with additional cleanup steps
def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Replace emojis with text
    text = re.sub(r':\)', ' happy ', text)
    text = re.sub(r':\(', ' sad ', text)
    
    # Remove mentions and replace with 'user'
    text = re.sub(r'@\w+', 'user', text)
    
    # Replace hashtags but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove repeating characters (e.g., "soooo" -> "so")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("\nPreprocessing text data...")
# Apply preprocessing to the text columns
train_df['processed_tweet'] = train_df['tweet'].apply(preprocess_text)
test_df['processed_tweet'] = test_df['tweet'].apply(preprocess_text)

# Function to remove stopwords and lemmatize the text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    # Remove some stopwords that might be relevant for hate speech detection
    for word in ['not', 'no', 'nor', 'against']:
        if word in stop_words:
            stop_words.remove(word)
    
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply stopword removal and lemmatization
train_df['processed_tweet'] = train_df['processed_tweet'].apply(clean_text)
test_df['processed_tweet'] = test_df['processed_tweet'].apply(clean_text)

# Feature engineering - add length and other features
train_df['tweet_length'] = train_df['processed_tweet'].apply(len)
test_df['tweet_length'] = test_df['processed_tweet'].apply(len)

train_df['word_count'] = train_df['processed_tweet'].apply(lambda x: len(x.split()))
test_df['word_count'] = test_df['processed_tweet'].apply(lambda x: len(x.split()))

# Check for common hate speech terms/patterns
def contains_hate_terms(text):
    hate_patterns = ['hate', 'kill', 'die', 'racist', 'sexist', 'stupid', 'dumb', 'idiot']
    return sum([1 for term in hate_patterns if term in text.lower()])

train_df['hate_term_count'] = train_df['processed_tweet'].apply(contains_hate_terms)
test_df['hate_term_count'] = test_df['processed_tweet'].apply(contains_hate_terms)

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_df[['processed_tweet', 'tweet_length', 'word_count', 'hate_term_count']], 
    train_df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['label']
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# Custom transformation for the pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=10000, ngram_range=(1, 3)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        
    def fit(self, X, y=None):
        self.vectorizer.fit(X['processed_tweet'])
        return self
    
    def transform(self, X):
        text_features = self.vectorizer.transform(X['processed_tweet'])
        
        # Convert additional features to numpy array
        additional_features = X[['tweet_length', 'word_count', 'hate_term_count']].values
        
        # Combine sparse matrix with dense array
        from scipy.sparse import hstack
        return hstack([text_features, additional_features])

print("\nBuilding and training models...")
# Create individual models for ensemble
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lr_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

# Create ensemble model
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lr', lr_model)
    ],
    voting='soft'
)

# Create pipeline
pipeline = Pipeline([
    ('features', TextFeatureExtractor(max_features=15000, ngram_range=(1, 3))),
    ('classifier', ensemble)
])

# Use stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

print("\nPerforming cross-validation...")
for train_idx, val_idx in skf.split(train_df[['processed_tweet', 'tweet_length', 'word_count', 'hate_term_count']], train_df['label']):
    # Get training and validation data for this fold
    X_fold_train = train_df.iloc[train_idx][['processed_tweet', 'tweet_length', 'word_count', 'hate_term_count']]
    y_fold_train = train_df.iloc[train_idx]['label']
    
    X_fold_val = train_df.iloc[val_idx][['processed_tweet', 'tweet_length', 'word_count', 'hate_term_count']]
    y_fold_val = train_df.iloc[val_idx]['label']
    
    # Train model
    pipeline.fit(X_fold_train, y_fold_train)
    
    # Get predictions
    y_fold_pred = pipeline.predict(X_fold_val)
    
    # Calculate F1 score
    fold_f1 = f1_score(y_fold_val, y_fold_pred)
    cv_scores.append(fold_f1)
    print(f"Fold F1 score: {fold_f1:.4f}")

print(f"\nCross-validation F1 score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Train the final model on the entire training data
print("\nTraining final model on all training data...")
pipeline.fit(train_df[['processed_tweet', 'tweet_length', 'word_count', 'hate_term_count']], train_df['label'])

# Make predictions on the test set
print("\nGenerating predictions on test data...")
test_predictions = pipeline.predict(test_df[['processed_tweet', 'tweet_length', 'word_count', 'hate_term_count']])

# Save predictions to a CSV file
with open('test_predictions.csv', 'w') as f:
    for pred in test_predictions:
        f.write(f"{pred}\n")

print("\nPredictions saved to 'test_predictions.csv'")

# Calculate additional statistics
print("\nTest predictions statistics:")
positive_tweets = np.sum(test_predictions == 1)
negative_tweets = np.sum(test_predictions == 0)
print(f"Predicted hate speech tweets (label 1): {positive_tweets} ({positive_tweets/len(test_predictions)*100:.2f}%)")
print(f"Predicted non-hate speech tweets (label 0): {negative_tweets} ({negative_tweets/len(test_predictions)*100:.2f}%)")

print("\nTraining complete!")
