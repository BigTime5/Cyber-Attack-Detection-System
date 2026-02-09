"""
Cyber Attack Detection System - Fast Version
Efficient Neural Network Implementation
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CYBER ATTACK DETECTION - BETH Dataset Analysis")
print("="*60)

np.random.seed(42)

# Load datasets
print("\n[1] Loading datasets...")
train_df = pd.read_csv('data/labelled_train.csv')
test_df = pd.read_csv('data/labelled_test.csv')
val_df = pd.read_csv('data/labelled_validation.csv')

print(f"Train: {train_df.shape[0]:,} samples")
print(f"Test: {test_df.shape[0]:,} samples")
print(f"Validation: {val_df.shape[0]:,} samples")

feature_cols = ['processId', 'threadId', 'parentProcessId', 'userId', 'mountNamespace', 'argsNum', 'returnValue']
target_col = 'sus_label'

X_train = train_df[feature_cols].values.astype(np.float64)
y_train = train_df[target_col].values.astype(np.float64)
X_test = test_df[feature_cols].values.astype(np.float64)
y_test = test_df[target_col].values.astype(np.float64)

# Class distribution
print("\n[2] Class Distribution...")
train_malicious = int(y_train.sum())
train_benign = len(y_train) - train_malicious
imbalance_ratio = train_benign / train_malicious if train_malicious > 0 else 0

print(f"Benign: {train_benign:,} ({train_benign/len(y_train)*100:.2f}%)")
print(f"Malicious: {train_malicious:,} ({train_malicious/len(y_train)*100:.2f}%)")
print(f"Imbalance Ratio: {imbalance_ratio:.1f}:1")

# Scale features
print("\n[3] Preprocessing...")
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std == 0] = 1
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# Build a simple but effective model using logistic regression as baseline
# Then enhance with neural network concepts
print("\n[4] Training Enhanced Neural Network...")

# Create augmented features for better learning
def create_features(X):
    X_new = X.copy()
    n = X.shape[0]
    
    # Add polynomial features for top features
    user_idx = feature_cols.index('userId')
    proc_idx = feature_cols.index('processId')
    
    # Interactions
    user_proc = (X[:, user_idx] * X[:, proc_idx]).reshape(-1, 1)
    user_sq = (X[:, user_idx] ** 2).reshape(-1, 1)
    
    return np.hstack([X_new, user_sq, user_proc])

X_train_enh = create_features(X_train_scaled)
X_test_enh = create_features(X_test_scaled)

print(f"Enhanced features: {X_train_enh.shape[1]}")

# Fast neural network using matrix operations
input_dim = X_train_enh.shape[1]

# Initialize weights
W1 = np.random.randn(input_dim, 64) * 0.1
b1 = np.zeros((1, 64))
W2 = np.random.randn(64, 32) * 0.1
b2 = np.zeros((1, 32))
W3 = np.random.randn(32, 16) * 0.1
b3 = np.zeros((1, 16))
W4 = np.random.randn(16, 1) * 0.1
b4 = np.zeros((1, 1))

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Training with mini-batches
print("Training neural network...")
batch_size = 4096
epochs = 30
lr = 0.01

# Balance training data
n_malicious = int(y_train.sum())
n_benign = len(y_train) - n_malicious
indices_mal = np.where(y_train == 1)[0]
indices_ben = np.where(y_train == 0)[0]

for epoch in range(epochs):
    # Sample balanced batches
    batch_mal = np.random.choice(indices_mal, min(n_malicious, batch_size // 2), replace=False)
    batch_ben = np.random.choice(indices_ben, batch_size - len(batch_mal), replace=False)
    batch_idx = np.concatenate([batch_mal, batch_ben])
    np.random.shuffle(batch_idx)
    
    X_batch = X_train_enh[batch_idx]
    y_batch = y_train[batch_idx]
    
    # Forward
    z1 = X_batch @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    a3 = relu(z3)
    z4 = a3 @ W4 + b4
    a4 = sigmoid(z4)
    
    # Backward
    m = len(y_batch)
    dz4 = a4 - y_batch.reshape(-1, 1)
    dW4 = a3.T @ dz4 / m
    db4 = np.sum(dz4, axis=0, keepdims=True) / m
    
    da3 = dz4 @ W4.T
    dz3 = da3 * (a3 > 0)
    dW3 = a2.T @ dz3 / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m
    
    da2 = dz3 @ W3.T
    dz2 = da2 * (a2 > 0)
    dW2 = a1.T @ dz2 / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    
    da1 = dz2 @ W2.T
    dz1 = da1 * (a1 > 0)
    dW1 = X_batch.T @ dz1 / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    
    # Update
    W4 -= lr * dW4
    b4 -= lr * db4
    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    
    if (epoch + 1) % 10 == 0:
        pred = (a4 >= 0.5).flatten()
        acc = (pred == y_batch).mean()
        print(f"Epoch {epoch+1}/{epochs}: Training Accuracy: {acc:.4f}")

# Evaluation
print("\n[5] Evaluating Model...")
z1 = X_test_enh @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
a2 = relu(z2)
z3 = a2 @ W3 + b3
a3 = relu(z3)
z4 = a3 @ W4 + b4
y_pred_proba = sigmoid(z4).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

# Metrics
tn = int(np.sum((y_test == 0) & (y_pred == 0)))
fp = int(np.sum((y_test == 0) & (y_pred == 1)))
fn = int(np.sum((y_test == 1) & (y_pred == 0)))
tp = int(np.sum((y_test == 1) & (y_pred == 1)))

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# ROC-AUC
def fast_roc_auc(y_true, y_proba):
    pos_scores = y_proba[y_true == 1]
    neg_scores = y_proba[y_true == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    correct = 0
    for p in pos_scores:
        correct += np.sum(neg_scores < p) + 0.5 * np.sum(neg_scores == p)
    return correct / (n_pos * n_neg)

roc_auc = fast_roc_auc(y_test, y_pred_proba)
avg_precision = np.mean([np.sum((y_test == 1) & (y_pred_proba >= t)) / max(1, np.sum(y_pred_proba >= t)) 
                         for t in np.linspace(0.1, 0.9, 20)])

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)

print("\n--- Classification Report ---")
print(f"                 Precision    Recall    F1-Score")
print(f"Benign (0)        {tn/(tn+fp):.4f}       {tn/(tn+fn):.4f}     {(2*tn/(tn+fp)*tn/(tn+fn))/(tn/(tn+fp)+tn/(tn+fn)) if (tn/(tn+fp)+tn/(tn+fn))>0 else 0:.4f}")
print(f"Malicious (1)     {tp/(tp+fp):.4f}       {tp/(tp+fn):.4f}     {f1:.4f}")

print("\n--- Confusion Matrix ---")
print(f"                  Predicted")
print(f"                  Benign    Malicious")
print(f"Actual Benign     {tn:8,}  {fp:8,}")
print(f"Actual Malicious  {fn:8,}  {tp:8,}")

print("\n--- Detailed Performance Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Average Precision Score: {avg_precision:.4f}")

print(f"\n--- Attack Detection Analysis ---")
detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"Detection Rate: {detection_rate:.4f}")
print(f"False Alarm Rate: {false_alarm_rate:.4f}")
print(f"Malicious Events Detected: {tp:,} out of {tp+fn:,}")

# Feature importance
print("\n--- Feature Importance Analysis ---")
feature_importance = {}
for i, col in enumerate(feature_cols):
    corr = abs(np.corrcoef(train_df[col].values, train_df[target_col].values)[0,1])
    feature_importance[col] = round(float(corr), 4)
    print(f"{col:20s}: {corr:.4f}")

most_important = max(feature_importance, key=feature_importance.get)
print(f"\nMost Predictive Feature: {most_important}")

# Save results
results = {
    'dataset_info': {
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'val_samples': int(len(val_df)),
        'train_malicious_pct': round(train_malicious/len(y_train)*100, 4),
        'test_malicious_pct': round(float(y_test.sum()/len(y_test)*100), 4),
        'class_imbalance_ratio': round(float(imbalance_ratio), 2),
        'num_features': int(X_train.shape[1])
    },
    'model_architecture': {
        'type': 'Deep Neural Network',
        'hidden_layers': [64, 32, 16],
        'activation': 'ReLU',
        'optimizer': 'Gradient Descent',
        'epochs': epochs
    },
    'performance_metrics': {
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'specificity': round(float(specificity), 4),
        'f1_score': round(float(f1), 4),
        'roc_auc': round(float(roc_auc), 4),
        'avg_precision': round(float(avg_precision), 4),
        'detection_rate': round(float(detection_rate), 4),
        'false_alarm_rate': round(float(false_alarm_rate), 4)
    },
    'confusion_matrix': {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    },
    'feature_importance': feature_importance
}

with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)