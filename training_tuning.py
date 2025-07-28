import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download stopwords once
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load datasets
fake_path = "/Users/kalinashrestha/Downloads/fake_and_real_news_dataset.csv/Fake.csv"
real_path = "/Users/kalinashrestha/Downloads/fake_and_real_news_dataset.csv/True.csv"

fake = pd.read_csv(fake_path)
real = pd.read_csv(real_path)

# Add labels: fake=1, real=0
fake['label'] = 1
real['label'] = 0

# Combine and shuffle dataset
data = pd.concat([fake, real], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)        # Keep only letters and spaces
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

data['text'] = data['text'].apply(clean_text)

# Split dataset: train+val (80%) and test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, stratify=data['label'], random_state=42
)

# Further split train+val into train (70%) and val (10%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
)

# Hyperparameters to tune
max_features_options = [2000, 5000, 10000]
max_iter_options = [100, 500, 1000]

results = []

for max_feat in max_features_options:
    # Fit TF-IDF vectorizer on training + validation data to avoid data leakage
    tfidf = TfidfVectorizer(max_features=max_feat, stop_words='english')
    X_train_val = tfidf.fit_transform(pd.concat([X_train, X_val]))

    # Split vectorized data back into train and val
    train_len = len(X_train)
    X_train_vec = X_train_val[:train_len]
    X_val_vec = X_train_val[train_len:]

    for max_iter_val in max_iter_options:
        # Train Logistic Regression
        model = LogisticRegression(max_iter=max_iter_val)
        model.fit(X_train_vec, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val_vec)

        # Evaluate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)

        results.append({
            'max_features': max_feat,
            'max_iter': max_iter_val,
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        })

# Print tuning results
print("Model Training and Hyperparameter Tuning Results (Validation Set):")
for res in results:
    print(res)

# Select best parameters based on highest F1 score
best_config = max(results, key=lambda x: x['f1_score'])
print("\nBest Hyperparameters Found:")
print(best_config)

# Train final model on full train+val with best params, evaluate on test set
tfidf_final = TfidfVectorizer(max_features=best_config['max_features'], stop_words='english')
X_train_val_final = tfidf_final.fit_transform(pd.concat([X_train, X_val]))
X_test_final = tfidf_final.transform(X_test)

model_final = LogisticRegression(max_iter=best_config['max_iter'])
model_final.fit(X_train_val_final, pd.concat([y_train, y_val]))

y_test_pred = model_final.predict(X_test_final)

print("\nFinal Model Evaluation on Test Set:")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
