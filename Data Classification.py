# ============================================================
# Project 2: Data Classification Using AI
# Intern: Isha Khatri | DecodeLabs | Batch 2026
# Algorithm: K-Nearest Neighbors (KNN)
# Dataset: Iris Benchmark Dataset
# ============================================================

# --- Step 1: Import Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

print("=" * 60)
print("  PROJECT 2: DATA CLASSIFICATION USING AI")
print("  Intern: Isha Khatri | DecodeLabs | Batch 2026")
print("=" * 60)

# -------------------------------------------------------
# Step 2: Load and Understand the Dataset (IPO - INPUT)
# -------------------------------------------------------
print("\n[1] Loading Iris Dataset...")
iris = load_iris()

# Convert to DataFrame for easy exploration
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

print(f"\nDataset Shape   : {df.shape}")
print(f"Samples         : {df.shape[0]} (Balanced)")
print(f"Features        : {df.shape[1] - 2}")
print(f"Classes         : {df['species'].nunique()} -> {list(df['species'].unique())}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nBasic Statistics:")
print(df.describe())

print(f"\nClass Distribution:")
print(df['species'].value_counts())

# -------------------------------------------------------
# Step 3: Feature Scaling - The Gatekeeper Rule (INPUT)
# -------------------------------------------------------
print("\n[2] Applying Feature Scaling (StandardScaler)...")

X = iris.data          # Features: sepal_length, sepal_width, petal_length, petal_width
y = iris.target        # Labels: 0=Setosa, 1=Versicolor, 2=Virginica

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("   Scaling complete: Mean=0, Variance=1 for all features.")

# -------------------------------------------------------
# Step 4: Train-Test Split - Structural Integrity (PROCESS)
# -------------------------------------------------------
print("\n[3] Splitting Data into Training and Testing Sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,       # 80% train, 20% test (as per project architecture)
    random_state=42,      # Reproducibility
    shuffle=True          # Remove order bias
)

print(f"   Training Set : {X_train.shape[0]} samples (80%)")
print(f"   Testing Set  : {X_test.shape[0]} samples (20%)")

# -------------------------------------------------------
# Step 5: Find Optimal K using Elbow Method (PROCESS)
# -------------------------------------------------------
print("\n[4] Finding Optimal K using Elbow Method...")

error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

optimal_k = k_range[error_rates.index(min(error_rates))]
print(f"   Optimal K found: {optimal_k}")

# Plot Elbow Curve
plt.figure(figsize=(10, 5))
plt.plot(list(k_range), error_rates, color='#1B3A6B', linewidth=2, marker='o',
         markerfacecolor='#E8622A', markersize=8)
plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
plt.title('Tuning the Engine: Choosing Optimal K\n(The Elbow Method)', fontsize=14, fontweight='bold')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: elbow_curve.png")

# -------------------------------------------------------
# Step 6: Train the KNN Model (PROCESS)
# -------------------------------------------------------
print(f"\n[5] Training KNN Model with K={optimal_k}...")

# IPO Framework - THE WORKFLOW: SCIKIT-LEARN
model = KNeighborsClassifier(n_neighbors=optimal_k)   # INSTANTIATE
model.fit(X_train, y_train)                            # FIT
predictions = model.predict(X_test)                   # PREDICT

print("   Model training complete.")

# -------------------------------------------------------
# Step 7: Evaluate Model - Output Validation (OUTPUT)
# -------------------------------------------------------
print("\n[6] Evaluating Model Performance...")

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

print(f"\n{'='*40}")
print(f"  ACCURACY  : {accuracy * 100:.2f}%")
print(f"  F1 SCORE  : {f1:.4f}")
print(f"{'='*40}")

print(f"\nDetailed Classification Report:")
target_names = ['Setosa', 'Versicolor', 'Virginica']
print(classification_report(y_test, predictions, target_names=target_names))

# -------------------------------------------------------
# Step 8: Confusion Matrix - The Diagnostic Tool (OUTPUT)
# -------------------------------------------------------
print("[7] Generating Confusion Matrix...")

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            linewidths=0.5, linecolor='gray',
            annot_kws={"size": 14, "weight": "bold"})
plt.title('The Diagnostic Tool: Confusion Matrix\n(Project 2 - KNN Classification)',
          fontsize=13, fontweight='bold')
plt.ylabel('Actual Label', fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: confusion_matrix.png")

# -------------------------------------------------------
# Step 9: Feature Distribution Visualization
# -------------------------------------------------------
print("\n[8] Generating Feature Distribution Plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = iris.feature_names
colors = {'Setosa': '#1B3A6B', 'Versicolor': '#4A90D9', 'Virginica': '#E8622A'}

for idx, feature in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    for species in ['Setosa', 'Versicolor', 'Virginica']:
        subset = df[df['species'] == species][feature]
        ax.hist(subset, alpha=0.6, label=species,
                color=colors[species], bins=15, edgecolor='white')
    ax.set_title(feature.replace('(cm)', '').strip(), fontweight='bold')
    ax.set_xlabel('cm')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Raw Material: Iris Dataset Feature Distributions',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: feature_distributions.png")

# -------------------------------------------------------
# Final Summary
# -------------------------------------------------------
print("\n" + "=" * 60)
print("  PROJECT 2 COMPLETE - FULL ARCHITECTURE SUMMARY")
print("=" * 60)
print(f"  Dataset       : Iris Benchmark (150 samples, 3 classes, 4 features)")
print(f"  Algorithm     : K-Nearest Neighbors (KNN)")
print(f"  Optimal K     : {optimal_k}")
print(f"  Train/Test    : 80% / 20% (Shuffled)")
print(f"  Preprocessing : StandardScaler (Mean=0, Variance=1)")
print(f"  Accuracy      : {accuracy * 100:.2f}%")
print(f"  F1 Score      : {f1:.4f}")
print(f"  Outputs       : confusion_matrix.png, elbow_curve.png, feature_distributions.png")
print("=" * 60)
print("\n  Intern: Isha Khatri | DecodeLabs | Batch 2026")
print("  GitHub: github.com/ishamahi04")
print("=" * 60)
