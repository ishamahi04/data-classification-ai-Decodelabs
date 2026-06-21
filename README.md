# 🤖 Data Classification Using AI
### Project 2 | DecodeLabs AI Internship | Batch 2026

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)](https://github.com/ishamahi04)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Project Overview

This project is **Project 2** of the DecodeLabs Industrial AI Training Program. It demonstrates the complete **Supervised Learning pipeline** — from raw data to intelligent classification — using the classic **Iris Benchmark Dataset** and the **K-Nearest Neighbors (KNN)** algorithm.

> *"We do not write the rules. We provide history, and the machine derives the logic."*
> — DecodeLabs, Project 2

---

## 🎯 Goal

Build a basic but complete classification model that:
- Loads and explores a real-world dataset
- Applies feature scaling (StandardScaler)
- Splits data into training and testing sets
- Trains a KNN classifier
- Evaluates model performance using Confusion Matrix and F1 Score

---

## 🏗️ IPO Framework (Full Architecture)

```
INPUT               →     PROCESS              →     OUTPUT
─────────────────────────────────────────────────────────────
Iris Dataset              Train-Test Split          Confusion Matrix
Feature Scaling     →     KNN Algorithm        →    F1 Score
(StandardScaler)          (Optimal K)               Classification Report
```

---

## 📂 Project Structure

```
data-classification-ai/
│
├── data_classification.py     # Main Python script (full pipeline)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
└── outputs/                   # Auto-generated on running the script
    ├── confusion_matrix.png
    ├── elbow_curve.png
    └── feature_distributions.png
```

---

## 🌸 Dataset: The Iris Benchmark

| Property    | Value                        |
|-------------|------------------------------|
| Samples     | 150 (Balanced — 50 per class)|
| Classes     | 3 (Setosa, Versicolor, Virginica) |
| Features    | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| Source      | `sklearn.datasets.load_iris()`|

---

## 🔬 Algorithm: K-Nearest Neighbors (KNN)

**Proximity Principle:** Similar things exist in close proximity.

- A new data point is classified based on the majority vote of its **K nearest neighbors** in the feature space.
- The optimal K is determined using the **Elbow Method** (minimizing the error rate).

```
K=1  →  Overfitting (Noise-sensitive)
K=Optimal  →  Best balance (The Elbow)
K=100  →  Underfitting (Too generic)
```

---

## ⚙️ Scikit-Learn Workflow

```python
# INSTANTIATE → FIT → PREDICT
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/ishamahi04/data-classification-ai.git
cd data-classification-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Script
```bash
python data_classification.py
```

The script will:
- Print dataset info and statistics to the console
- Display and save the Elbow Curve, Confusion Matrix, and Feature Distribution plots
- Print the final Accuracy and F1 Score

---

## 📊 Expected Results

| Metric       | Expected Value |
|--------------|----------------|
| Accuracy     | ~96–100%       |
| F1 Score     | ~0.96–1.00     |
| Train Split  | 80%            |
| Test Split   | 20%            |

> **Note:** In imbalanced data, accuracy alone is a mirage. This project uses **F1 Score** (the harmonic mean of Precision and Recall) as the primary evaluation metric.

---

## 📈 Output Visualizations

### 1. Elbow Curve — Choosing Optimal K
Plots error rate vs K value. The "elbow" (minimum error) gives the best K.

### 2. Confusion Matrix
Shows True Positives, False Positives, True Negatives, and False Negatives per class.

### 3. Feature Distribution
Histogram of all 4 features across the 3 Iris species.

---

## 🧠 Key Concepts Covered

| Concept | Description |
|---|---|
| Supervised Learning | Learning from labeled training data |
| Feature Scaling | StandardScaler (Mean=0, Variance=1) removes bias |
| Train-Test Split | 80/20 with shuffle to remove order bias |
| KNN Algorithm | Distance-based proximity classification |
| Elbow Method | Finding the optimal hyperparameter K |
| Confusion Matrix | Diagnostic tool: TP, FP, FN, TN |
| F1 Score | Harmonic mean of Precision & Recall |

---

## 🔭 Next Steps (Emerging Horizons)

- Deep Learning & CNNs for Computer Vision tasks
- Comparing KNN with Decision Tree, SVM, and Random Forest
- Testing the model on completely new/unseen data samples

---

## 👩‍💻 Intern Details

| Field | Info |
|---|---|
| Name | Isha Khatri |
| Program | B.Tech — Artificial Intelligence & Machine Learning |
| College | Sree Chaitanya Institute of Technological Sciences |
| Internship | DecodeLabs Industrial AI Training, Batch 2026 |
| GitHub | [@ishamahi04](https://github.com/ishamahi04) |

---

## 📜 License

This project is licensed under the MIT License.

---

*Built with 🤍 as part of the DecodeLabs Batch 2026 Industrial Training Program.*
