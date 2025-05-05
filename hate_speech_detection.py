import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_tweets.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Check the first few rows of the training data
print("\nFirst few rows of training data:")
print(train_df.head())

# Check the first few rows of the test data
print("\nFirst few rows of test data:")
print(test_df.head())

# Check for missing values
print("\nMissing values in training data:")
print(train_df.isnull().sum())

print("\nMissing values in test data:")
print(test_df.isnull().sum())

# Analyze class distribution in training data
print("\nClass distribution in training data:")
print(train_df['label'].value_counts())
print(f"Class balance: {train_df['label'].value_counts(normalize=True)}")

# Preprocess text function
def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply preprocessing to the text columns
train_df['processed_tweet'] = train_df['tweet'].apply(preprocess_text)
test_df['processed_tweet'] = test_df['tweet'].apply(preprocess_text)

# Function to remove stopwords and stem the text
def remove_stopwords_and_stem(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = nltk.word_tokenize(text)
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply stopword removal and stemming
train_df['processed_tweet'] = train_df['processed_tweet'].apply(remove_stopwords_and_stem)
test_df['processed_tweet'] = test_df['processed_tweet'].apply(remove_stopwords_and_stem)

# Print a sample of processed tweets
print("\nSample of processed tweets:")
for i in range(5):
    print(f"Original: {train_df['tweet'].iloc[i]}")
    print(f"Processed: {train_df['processed_tweet'].iloc[i]}")
    print("---")

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_df['processed_tweet'], 
    train_df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['label']
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# Feature extraction and model training
# Create a pipeline with TF-IDF and XGBoost classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('classifier', xgb.XGBClassifier(random_state=42))
])

# Set up parameters for GridSearchCV
params = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.1, 0.01],
    'classifier__n_estimators': [100, 200]
}

# Find the best parameters using GridSearchCV
grid_search = GridSearchCV(pipeline, params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and score
print("\nBest parameters:")
print(grid_search.best_params_)
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Evaluate on validation set
y_val_pred = grid_search.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred)
print(f"\nValidation F1 score: {val_f1:.4f}")

# Classification report on validation set
print("\nClassification report on validation set:")
print(classification_report(y_val, y_val_pred))

# Confusion matrix on validation set
cm = confusion_matrix(y_val, y_val_pred)
print("\nConfusion Matrix:")
print(cm)

# Train the final model on the entire training data
final_model = grid_search.best_estimator_
final_model.fit(train_df['processed_tweet'], train_df['label'])

# Make predictions on the test set
test_predictions = final_model.predict(test_df['processed_tweet'])

# Create DataFrame with id and label columns
predictions_df = pd.DataFrame({
    'id': test_df['id'],
    'label': test_predictions
})

# Save predictions to a CSV file
predictions_df.to_csv('test_predictions.csv', index=False)

print("\nPredictions saved to 'test_predictions.csv'")

# Calculate additional statistics
print("\nAdditional model statistics:")
positive_tweets = len(test_predictions[test_predictions == 1])
negative_tweets = len(test_predictions[test_predictions == 0])
print(f"Predicted hate speech tweets (label 1): {positive_tweets} ({positive_tweets/len(test_predictions)*100:.2f}%)")
print(f"Predicted non-hate speech tweets (label 0): {negative_tweets} ({negative_tweets/len(test_predictions)*100:.2f}%)")
