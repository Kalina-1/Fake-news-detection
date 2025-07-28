import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)

# Load cleaned and lemmatized dataset
df = pd.read_csv("data/cleaned_data.csv")

# Drop any rows with missing text or label
df.dropna(subset=['lemmatized_text', 'label'], inplace=True)

# Feature and target
X = df['lemmatized_text']
y = df['label']

# Split into training (70%), validation (10%), and test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)
X_test_vec = tfidf.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predictions
y_val_pred = model.predict(X_val_vec)
y_test_pred = model.predict(X_test_vec)
y_val_proba = model.predict_proba(X_val_vec)[:, 1]
y_test_proba = model.predict_proba(X_test_vec)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)

# Feature Importance
feature_names = np.array(tfidf.get_feature_names_out())
coefficients = model.coef_.flatten()
top_indices = np.argsort(np.abs(coefficients))[-20:]
top_features = feature_names[top_indices]
top_importances = coefficients[top_indices]

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index=["Real (0)", "Fake (1)"], columns=["Predicted Real", "Predicted Fake"])

# ---------------------- PLOTS ----------------------

# 1. ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 2. Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, color='blue')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()

# 3. Top 20 Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=top_importances, y=top_features)
plt.title("Top 20 Important Features (TF-IDF Coefficients)")
plt.tight_layout()
plt.show()

# 4. Confusion Matrix Heatmap
plt.figure()
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# 5. Accuracy and F1 Score Comparison
metrics_df = pd.DataFrame({
    "Set": ["Validation", "Test"],
    "Accuracy": [accuracy_score(y_val, y_val_pred), accuracy_score(y_test, y_test_pred)],
    "F1 Score": [f1_score(y_val, y_val_pred), f1_score(y_test, y_test_pred)]
})
metrics_df.set_index("Set", inplace=True)
metrics_df.plot(kind="bar", figsize=(8, 5))
plt.title("Accuracy and F1 Score Comparison (Validation vs Test)")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# 6. Pie Chart of Predicted Labels
unique, counts = np.unique(y_test_pred, return_counts=True)
plt.figure()
plt.pie(counts, labels=[f"Real ({counts[0]})", f"Fake ({counts[1]})"], autopct='%1.1f%%', colors=['green', 'red'])
plt.title("Distribution of Predicted Labels on Test Set")
plt.tight_layout()
plt.show()
