import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === LOAD CLEANED AND LEMMATIZED DATA ===
df = pd.read_csv("data/cleaned_data.csv")


# === HANDLE MISSING DATA ===
df.dropna(subset=['lemmatized_text', 'label'], inplace=True)

# === TF-IDF VECTORIZATION ===
tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(df['lemmatized_text']).toarray()
y = df['label']

# === SPLIT DATA ===
# First split: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Second split: from train+val → 70% train, 10% val
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)

# === TRAIN LOGISTIC REGRESSION MODEL ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === VALIDATION EVALUATION ===
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
val_prec = precision_score(y_val, y_val_pred)
val_rec = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

print("=== VALIDATION SET METRICS ===")
print(f"Accuracy : {val_acc:.4f}")
print(f"Precision: {val_prec:.4f}")
print(f"Recall   : {val_rec:.4f}")
print(f"F1 Score : {val_f1:.4f}")

# === TEST SET EVALUATION ===
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\n=== TEST SET METRICS ===")
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall   : {test_rec:.4f}")
print(f"F1 Score : {test_f1:.4f}")
