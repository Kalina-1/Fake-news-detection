# results_and_evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# Load preprocessed data
df = pd.read_csv("data/cleaned_data.csv")

# Drop any missing values
df.dropna(subset=['lemmatized_text', 'label'], inplace=True)

X = df['lemmatized_text']
y = df['label']

# Split: 80% temp (train+val) and 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Further split: 70% train and 10% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
)

# Hyperparameter tuning
max_features_options = [2000, 5000, 10000]
max_iter_options = [100, 500, 1000]
results = []

for max_feat in max_features_options:
    tfidf = TfidfVectorizer(max_features=max_feat, stop_words='english')
    X_train_val = tfidf.fit_transform(pd.concat([X_train, X_val]))

    X_train_vec = X_train_val[:len(X_train)]
    X_val_vec = X_train_val[len(X_train):]

    for max_iter_val in max_iter_options:
        model = LogisticRegression(max_iter=max_iter_val)
        model.fit(X_train_vec, y_train)
        y_val_pred = model.predict(X_val_vec)

        results.append({
            'max_features': max_feat,
            'max_iter': max_iter_val,
            'accuracy': round(accuracy_score(y_val, y_val_pred), 4),
            'precision': round(precision_score(y_val, y_val_pred), 4),
            'recall': round(recall_score(y_val, y_val_pred), 4),
            'f1_score': round(f1_score(y_val, y_val_pred), 4)
        })

# Display tuning results
print("Model Training and Hyperparameter Tuning Results (Validation Set):")
for res in results:
    print(res)

# Best configuration
best_config = max(results, key=lambda x: x['f1_score'])
print("\nBest Hyperparameters Found:")
print(best_config)

# Train final model and evaluate on test set
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

# Plot confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real (0)', 'Fake (1)'],
            yticklabels=['Real (0)', 'Fake (1)'])
plt.title("Confusion Matrix for Final Model on Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
# results_and_evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# Load preprocessed data
df = pd.read_csv("data/cleaned_data.csv")

# Drop any missing values
df.dropna(subset=['lemmatized_text', 'label'], inplace=True)

X = df['lemmatized_text']
y = df['label']

# Split: 80% temp (train+val) and 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Further split: 70% train and 10% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
)

# Hyperparameter tuning
max_features_options = [2000, 5000, 10000]
max_iter_options = [100, 500, 1000]
results = []

for max_feat in max_features_options:
    tfidf = TfidfVectorizer(max_features=max_feat, stop_words='english')
    X_train_val = tfidf.fit_transform(pd.concat([X_train, X_val]))

    X_train_vec = X_train_val[:len(X_train)]
    X_val_vec = X_train_val[len(X_train):]

    for max_iter_val in max_iter_options:
        model = LogisticRegression(max_iter=max_iter_val)
        model.fit(X_train_vec, y_train)
        y_val_pred = model.predict(X_val_vec)

        results.append({
            'max_features': max_feat,
            'max_iter': max_iter_val,
            'accuracy': round(accuracy_score(y_val, y_val_pred), 4),
            'precision': round(precision_score(y_val, y_val_pred), 4),
            'recall': round(recall_score(y_val, y_val_pred), 4),
            'f1_score': round(f1_score(y_val, y_val_pred), 4)
        })

# Display tuning results
print("Model Training and Hyperparameter Tuning Results (Validation Set):")
for res in results:
    print(res)

# Best configuration
best_config = max(results, key=lambda x: x['f1_score'])
print("\nBest Hyperparameters Found:")
print(best_config)

# Train final model and evaluate on test set
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

# Plot confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real (0)', 'Fake (1)'],
            yticklabels=['Real (0)', 'Fake (1)'])
plt.title("Confusion Matrix for Final Model on Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
