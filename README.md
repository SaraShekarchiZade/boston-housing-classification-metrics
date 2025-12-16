# boston-housing-classification-metrics
This project explores the evaluation of machine learning models using classification metrics on the Boston Housing dataset
# Boston Housing Price Evaluation Using Classification Metrics

## 1. Project Overview

This project evaluates housing prices using the **Boston Housing dataset**. Although the original problem is a **regression task** (predicting continuous house prices), the objective of this assignment is to evaluate **classification metrics** such as **Accuracy, Precision, Recall, F1-score, ROC Curve, and AUC**.

To align the dataset with these metrics, the regression problem is **redefined as a binary classification task** by categorizing houses into *low-priced* and *high-priced* classes based on the median house price.

---

## 2. Dataset Description

* Dataset: Boston Housing
* Features: 13 numerical attributes
* Target variable: `MEDV` (Median value of owner-occupied homes)

---

## 3. Problem Reformulation

Since metrics like Recall and ROC-AUC are defined for classification tasks, the continuous target variable (`MEDV`) is converted into a binary label:

* **Class 1 (High Price):** MEDV ≥ median(MEDV)
* **Class 0 (Low Price):** MEDV < median(MEDV)

The median is used to ensure balanced class distribution and avoid bias.

---

## 4. Methodology

### 4.1 Preprocessing

* Train/Test split (80% / 20%)
* Feature standardization using `StandardScaler`

### 4.2 Model Selection

* **Logistic Regression** is used due to its interpretability and probabilistic output, which is required for ROC and AUC analysis.

---

## 5. Evaluation Metrics

The following metrics are computed:

* Accuracy
* Precision
* Recall
* F1-score
* ROC Curve
* AUC Score

---

## 6. Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# Load Boston Housing dataset
boston = fetch_openml(name='boston', version=1, as_frame=True)
X = boston.data
y = boston.target.astype(float)

# Convert regression target to binary classification
y_binary = (y >= y.median()).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

---

## 7. Results and Analysis

* **Accuracy** provides an overall performance measure but may be misleading alone.
* **Precision** indicates how many predicted high-price houses are truly high-priced.
* **Recall** measures the model's ability to identify all high-price houses.
* **F1-score** balances Precision and Recall.
* **ROC Curve** visualizes the trade-off between True Positive Rate and False Positive Rate.
* **AUC** quantifies the model's discriminative ability; values closer to 1 indicate better performance.

---

## 8. Conclusion

By reformulating the Boston Housing regression problem into a classification task, it becomes possible to evaluate commonly used classification metrics. Logistic Regression demonstrates reasonable performance and provides interpretable results suitable for academic evaluation.

This approach ensures alignment with course requirements while preserving methodological correctness.
