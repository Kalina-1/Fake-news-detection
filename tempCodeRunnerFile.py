import pandas as pd
import nltk

# Download all essential tokenizers
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')        # For lemmatizer
nltk.download('stopwords')     # Optional but useful


# Load datasets using full paths
fake_df = pd.read_csv("/Users/kalinashrestha/Downloads/fake_and_real_news_dataset.csv/Fake.csv")
real_df = pd.read_csv("/Users/kalinashrestha/Downloads/fake_and_real_news_dataset.csv/True.csv")

# Add labels: 1 = Fake, 0 = Real
fake_df['label'] = 1
real_df['label'] = 0

# Combine the datasets
df = pd.concat([fake_df, real_df], ignore_index=True)

# Optional: show top rows
print(df.head())




import re
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>+', '', text)  # remove HTML
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Preview result
print(df[["text", "clean_text"]].head())


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

# Apply to cleaned text
df["lemmatized_text"] = df["clean_text"].apply(lemmatize_text)

# Preview a few rows
print(df[["clean_text", "lemmatized_text"]].head())



from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize the lemmatized text
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df["lemmatized_text"])

# Convert to DataFrame for visualization
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Preview first few rows
print(tfidf_df.head())

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df["lemmatized_text"])

# Convert to previewable format
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(tfidf_df.head())
