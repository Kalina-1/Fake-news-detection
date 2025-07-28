# evaluate_logistic_regression.py

import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download stopwords (only the first time)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load datasets
fake_path = "data/Fake.csv"
real_path = "data/True.csv"

fake = pd.read_csv(fake_path)
real = pd.read_csv(real_path)

# Add labels
fake['label'] = 1
real['label'] = 0

# Combine and shuffle datasets
data = pd.concat([fake, real], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Apply text cleaning
data['text'] = data['text'].apply(clean_text)

# Function to evaluate model with different max_features in TF-IDF
def evaluate_tfidf_features(max_features_list):
    results = []
    for max_feat in max_features_list:
        print(f"\nEvaluating for max_features = {max_feat}")
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(max_features=max_feat, stop_words='english')
        X = tfidf.fit_transform(data['text'])
        y = data['label']

        # Split into Train+Validation and Test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        # Split again: Train and Validation
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)

        # Train Logistic Regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Predict and Evaluate on Validation Set
        y_val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)

        # Save results
        results.append({
            'max_features': max_feat,
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        })

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

    return results

# Try different TF-IDF feature sizes
max_features_list = [2000, 5000, 10000]
experiment_results = evaluate_tfidf_features(max_features_list)

# Print Summary Table
print("\n--- Comparison Summary ---")
for res in experiment_results:
    print(res)
