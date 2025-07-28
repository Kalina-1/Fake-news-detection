# pre_post.py

import pandas as pd
import re
import nltk
import string  # <-- Import the string module
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# NLTK downloads
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load datasets
fake_path = "data/Fake.csv"
real_path = "data/True.csv"
fake = pd.read_csv(fake_path)
real = pd.read_csv(real_path)

# Add labels: 1 for fake, 0 for real
fake['label'] = 1
real['label'] = 0

# Combine the datasets
df = pd.concat([fake[['title', 'text', 'label']], real[['title', 'text', 'label']]])

# Function to clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Tokenization and Lemmatization
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

df["lemmatized_text"] = df["clean_text"].apply(lemmatize_text)

# Save cleaned and lemmatized text
df.to_csv("data/cleaned_data.csv", index=False)
