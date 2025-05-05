import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the datasets
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_tweets.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Handle missing values
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Analyze class distribution
print("\nClass distribution in training data:")
print(train_df['label'].value_counts())
print(f"Class balance: {train_df['label'].value_counts(normalize=True)}")

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Replace emojis with text
    text = re.sub(r':\)', ' happy ', text)
    text = re.sub(r':\(', ' sad ', text)
    text = re.sub(r';+\)', ' wink ', text)
    
    # Replace mentions with [USER]
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Replace hashtags but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Handle repeating characters (e.g., "soooo" -> "soo")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s!?.]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("Preprocessing text...")
# Apply preprocessing
train_df['processed_tweet'] = train_df['tweet'].apply(preprocess_text)
test_df['processed_tweet'] = test_df['tweet'].apply(preprocess_text)

# Function to remove stopwords and lemmatize
def clean_text(text):
    # Create a custom stopwords list (removing negation words that might be important for sentiment)
    stop_words = set(stopwords.words('english'))
    for word in ['not', 'no', 'nor', 'against', 'hate', 'don', 'doesn', 'didn']:
        if word in stop_words:
            stop_words.remove(word)
    
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply stopword removal and lemmatization
train_df['clean_tweet'] = train_df['processed_tweet'].apply(clean_text)
test_df['clean_tweet'] = test_df['processed_tweet'].apply(clean_text)

# Feature engineering
print("Extracting features...")

# Text length features
train_df['tweet_length'] = train_df['processed_tweet'].apply(len)
test_df['tweet_length'] = test_df['processed_tweet'].apply(len)

train_df['word_count'] = train_df['processed_tweet'].apply(lambda x: len(x.split()))
test_df['word_count'] = test_df['processed_tweet'].apply(lambda x: len(x.split()))

# Check for potential hate speech indicators
def contains_hate_terms(text):
    hate_patterns = ['hate', 'kill', 'die', 'racist', 'sexist', 'stupid', 'dumb', 'idiot', 'ugly']
    return sum([1 for term in hate_patterns if term in text.lower()])

def contains_profanity(text):
    profanity = ['fuck', 'shit', 'bitch', 'ass', 'damn', 'cunt', 'whore']
    return sum([1 for term in profanity if term in text.lower()])

def contains_uppercase(text):
    return sum(1 for c in text if c.isupper()) / max(len(text), 1)

# Apply feature extraction
train_df['hate_term_count'] = train_df['processed_tweet'].apply(contains_hate_terms)
test_df['hate_term_count'] = test_df['processed_tweet'].apply(contains_hate_terms)

train_df['profanity_count'] = train_df['processed_tweet'].apply(contains_profanity)
test_df['profanity_count'] = test_df['processed_tweet'].apply(contains_profanity)

train_df['uppercase_ratio'] = train_df['tweet'].apply(contains_uppercase)
test_df['uppercase_ratio'] = test_df['tweet'].apply(contains_uppercase)

# Split the data for validation
print("Splitting data for validation...")
X_train, X_val, y_train, y_val = train_test_split(
    train_df[['clean_tweet', 'tweet_length', 'word_count', 'hate_term_count', 
             'profanity_count', 'uppercase_ratio']], 
    train_df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['label']
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# Create feature extractors
class TextFeaturePipeline:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=10000, 
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.9
        )
        self.count_vec = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
        
    def fit(self, X):
        self.tfidf.fit(X['clean_tweet'])
        self.count_vec.fit(X['clean_tweet'])
        return self
    
    def transform(self, X):
        tfidf_features = self.tfidf.transform(X['clean_tweet'])
        count_features = self.count_vec.transform(X['clean_tweet'])
        
        # Get the additional features
        additional_features = X[['tweet_length', 'word_count', 'hate_term_count', 
                               'profanity_count', 'uppercase_ratio']].values
        
        # Combine all features
        from scipy.sparse import hstack
        return hstack([tfidf_features, count_features, additional_features])

# Create our pipeline
feature_pipeline = TextFeaturePipeline()

# Fit the feature extraction on the training data
print("Extracting text features...")
feature_pipeline.fit(X_train)

# Transform the data
X_train_features = feature_pipeline.transform(X_train)
X_val_features = feature_pipeline.transform(X_val)

# Create individual models
print("Building ensemble of models...")
# 1. XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.5,  # Handle class imbalance
    random_state=42
)

# 2. Logistic Regression
lr_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    random_state=42,
    max_iter=1000,
    solver='liblinear'
)

# 3. SVM
svm_model = LinearSVC(
    C=1.0,
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

# 4. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

# 5. Naive Bayes
nb_model = MultinomialNB(alpha=0.1)

# Train individual models
print("Training individual models...")
models = {
    'XGBoost': xgb_model,
    'Logistic Regression': lr_model,
    'SVM': svm_model,
    'Random Forest': rf_model,
    'Naive Bayes': nb_model
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_features, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val_features)
    f1 = f1_score(y_val, y_val_pred)
    
    print(f"{name} F1 score: {f1:.4f}")
    print(classification_report(y_val, y_val_pred))

# Create and train voting ensemble
print("\nTraining ensemble model...")
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lr', lr_model),
        ('svm', svm_model),
        ('rf', rf_model),
        ('nb', nb_model)
    ],
    voting='hard'  # Use majority vote
)

ensemble.fit(X_train_features, y_train)

# Evaluate ensemble on validation set
y_val_pred_ensemble = ensemble.predict(X_val_features)
ensemble_f1 = f1_score(y_val, y_val_pred_ensemble)

print(f"Ensemble F1 score: {ensemble_f1:.4f}")
print(classification_report(y_val, y_val_pred_ensemble))

# Train final model on entire training dataset
print("\nTraining final model on entire training data...")
# Combine training and validation data
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([pd.Series(y_train), pd.Series(y_val)])

# Extract features
X_full_features = feature_pipeline.transform(X_full)

# Train the ensemble on the full dataset
ensemble.fit(X_full_features, y_full)

# Transform test data
print("\nGenerating predictions on test data...")
X_test_features = feature_pipeline.transform(test_df[['clean_tweet', 'tweet_length', 'word_count', 
                                                    'hate_term_count', 'profanity_count', 'uppercase_ratio']])

# Make predictions on the test set
test_predictions = ensemble.predict(X_test_features)

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
