import numpy as np

def create_features(X):
    """
    Create enhanced features for the cyber attack detection model.
    Assumes X has columns in the order:
    ['processId', 'threadId', 'parentProcessId', 'userId', 'mountNamespace', 'argsNum', 'returnValue']
    """
    X_new = X.copy()
    
    # Indices based on standard feature columns
    # processId is index 0
    # userId is index 3
    user_idx = 3
    proc_idx = 0
    
    # Interactions
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)
        
    user_proc = (X[:, user_idx] * X[:, proc_idx]).reshape(-1, 1)
    user_sq = (X[:, user_idx] ** 2).reshape(-1, 1)
    
    return np.hstack([X_new, user_sq, user_proc])

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    # Clip to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class CyberAttackModel:
    def __init__(self, input_dim=None, learning_rate=0.01):
        self.lr = learning_rate
        self.input_dim = input_dim
        self.weights = {}
        if input_dim:
            self.init_weights(input_dim)
            
    def init_weights(self, input_dim):
        # Weights matching the original script architecture: 64 -> 32 -> 16 -> 1
        np.random.seed(42) # For reproducibility
        self.weights['W1'] = np.random.randn(input_dim, 64) * 0.1
        self.weights['b1'] = np.zeros((1, 64))
        self.weights['W2'] = np.random.randn(64, 32) * 0.1
        self.weights['b2'] = np.zeros((1, 32))
        self.weights['W3'] = np.random.randn(32, 16) * 0.1
        self.weights['b3'] = np.zeros((1, 16))
        self.weights['W4'] = np.random.randn(16, 1) * 0.1
        self.weights['b4'] = np.zeros((1, 1))

    def forward(self, X):
        # Unpack weights
        W1, b1 = self.weights['W1'], self.weights['b1']
        W2, b2 = self.weights['W2'], self.weights['b2']
        W3, b3 = self.weights['W3'], self.weights['b3']
        W4, b4 = self.weights['W4'], self.weights['b4']
        
        # Forward pass
        self.z1 = X @ W1 + b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ W2 + b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ W3 + b3
        self.a3 = relu(self.z3)
        self.z4 = self.a3 @ W4 + b4
        self.a4 = sigmoid(self.z4)
        return self.a4

    def predict(self, X):
        return self.forward(X)

    def train_step(self, X_batch, y_batch):
        m = len(y_batch)
        
        # Forward
        output = self.forward(X_batch)
        
        # Backward (Gradient Descent)
        # Unpack weights for update
        W4, W3, W2, W1 = self.weights['W4'], self.weights['W3'], self.weights['W2'], self.weights['W1']
        b4, b3, b2, b1 = self.weights['b4'], self.weights['b3'], self.weights['b2'], self.weights['b1']
        
        a4 = output
        a3, a2, a1 = self.a3, self.a2, self.a1
        
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
        
        # Update weights (Store back)
        self.weights['W4'] -= self.lr * dW4
        self.weights['b4'] -= self.lr * db4
        self.weights['W3'] -= self.lr * dW3
        self.weights['b3'] -= self.lr * db3
        self.weights['W2'] -= self.lr * dW2
        self.weights['b2'] -= self.lr * db2
        self.weights['W1'] -= self.lr * dW1
        self.weights['b1'] -= self.lr * db1

    def train(self, X_train, y_train, epochs=30, batch_size=4096):
        n_malicious = int(y_train.sum())
        n_benign = len(y_train) - n_malicious
        indices_mal = np.where(y_train == 1)[0]
        indices_ben = np.where(y_train == 0)[0]
        
        progress_bar = None
        # We can't easily use streamlit progress bar here without passing it in
        # So we'll return a generator or just run it.
        # For simplicity, we just run it.
        
        for epoch in range(epochs):
            # Sample balanced batches
            # If batch_size is larger than available malicious samples, we might need to adjust or oversample
            # The original script does:
            batch_mal = np.random.choice(indices_mal, min(n_malicious, batch_size // 2), replace=False)
            batch_ben = np.random.choice(indices_ben, batch_size - len(batch_mal), replace=False)
            batch_idx = np.concatenate([batch_mal, batch_ben])
            np.random.shuffle(batch_idx)
            
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            self.train_step(X_batch, y_batch)

