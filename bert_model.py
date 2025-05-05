import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import re
import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the datasets
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_tweets.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Drop any rows with missing values
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Analyze class distribution
print("\nClass distribution in training data:")
print(train_df['label'].value_counts())
print(f"Class balance: {train_df['label'].value_counts(normalize=True)}")

# Simple preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Replace user mentions with [USER]
    text = re.sub(r'@\w+', '[USER]', text)
    # Replace hashtags with the text only
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing
print("Preprocessing text...")
train_df['processed_tweet'] = train_df['tweet'].apply(preprocess_text)
test_df['processed_tweet'] = test_df['tweet'].apply(preprocess_text)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    train_df['processed_tweet'].values, 
    train_df['label'].values, 
    test_size=0.1,  # Using a smaller validation set to speed up training
    random_state=42, 
    stratify=train_df['label']
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# Load the BERT tokenizer
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128  # Maximum sequence length

# Create a PyTorch Dataset
class TweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Get features
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

# Create datasets
print("Creating datasets...")
train_dataset = TweetDataset(X_train, y_train, tokenizer, max_length)
val_dataset = TweetDataset(X_val, y_val, tokenizer, max_length)
test_dataset = TweetDataset(test_df['processed_tweet'].values, tokenizer=tokenizer, max_length=max_length)

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load the BERT model
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = model.to(device)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training function
def train_model(model, train_loader, val_loader, optimizer, epochs=3):
    best_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print('-' * 40)
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        print(f"Training loss: {train_loss:.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                true_labels = labels.cpu().numpy()
                
                val_preds.extend(preds)
                val_true.extend(true_labels)
        
        # Calculate F1 score
        val_f1 = f1_score(val_true, val_preds)
        print(f"Validation F1 score: {val_f1:.4f}")
        
        print(classification_report(val_true, val_preds))
        
        # Save the best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_bert_model.pt')
            print("Model saved!")
    
    return best_f1

# Train the model
print("\nTraining BERT model...")
epochs = 2  # Reduced for demonstration, use 3-4 for better results
best_f1 = train_model(model, train_loader, val_loader, optimizer, epochs=epochs)
print(f"Best validation F1 score: {best_f1:.4f}")

# Load the best model
model.load_state_dict(torch.load('best_bert_model.pt'))
model.eval()

# Make predictions on the test set
print("\nGenerating predictions on test data...")
test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)

# Save predictions to a CSV file
with open('test_predictions.csv', 'w') as f:
    for pred in test_preds:
        f.write(f"{pred}\n")

print("\nPredictions saved to 'test_predictions.csv'")

# Calculate additional statistics
print("\nTest predictions statistics:")
positive_tweets = sum(test_preds)
negative_tweets = len(test_preds) - positive_tweets
print(f"Predicted hate speech tweets (label 1): {positive_tweets} ({positive_tweets/len(test_preds)*100:.2f}%)")
print(f"Predicted non-hate speech tweets (label 0): {negative_tweets} ({negative_tweets/len(test_preds)*100:.2f}%)")

print("\nBERT model training complete!")
