import pandas as pd
import numpy as np
import nltk
import re
import string
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Stemmer
ps = PorterStemmer()

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Remove duplicates
df.drop_duplicates(inplace=True)

# Text preprocessing function
def transform_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Tokenization
    words = text.split()

    # Remove stopwords and stemming
    filtered_words = []

    for word in words:

        if word not in stopwords.words('english'):

            stemmed_word = ps.stem(word)

            filtered_words.append(stemmed_word)

    return " ".join(filtered_words)

# Apply preprocessing
df['transformed_message'] = df['message'].apply(transform_text)

# Features and labels
X = df['transformed_message']

y = df['label']

# TF-IDF Vectorizer with bigrams
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

# Transform text
X = tfidf.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearSVC()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Classification Report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Save vectorizer
joblib.dump(tfidf, 'vectorizer.pkl')

# Save model
joblib.dump(model, 'spam_model.pkl')

print("\nModel and vectorizer saved successfully!")