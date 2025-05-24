import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ot import sinkhorn2
import torch
import torch.optim as optim

# --- Step 1: Load and Preprocess Data (Adult Dataset) ---
adult = fetch_openml(name='adult', version=2, as_frame=True)
X = adult.data[['age', 'education-num', 'hours-per-week']].astype(float)
y = (adult.target == '>50K').astype(int)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step 2: Pretrain Target Model (m) ---
m = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(f"Target model accuracy: {accuracy_score(y_test, m.predict(X_test)):.3f}")

# --- Step 3: Generate Counterfactuals (D_cf) ---
def generate_counterfactuals(X, m, n_samples=400):
    X_cf = []
    for x in X[m.predict(X) == 0][:n_samples]:
        x_cf = x.copy()
        x_cf[0] += 1.0  # Perturb 'age' to flip prediction
        if m.predict([x_cf])[0] == 1:
            X_cf.append(x_cf)
    return np.array(X_cf)

X_cf = generate_counterfactuals(X_train, m)
y_cf = np.full(len(X_cf), 0.5)

# --- Step 4: Compute Wasserstein Barycenters (Q_0, Q_1) ---
def compute_barycenter(P_c, P_cf, lambda_c=0.5, gamma=0.1, n_iter=100, lr=0.01):
    """Optimize barycenter Q_c using gradient descent."""
    # Initialize Q_c as mean of P_c and P_cf
    Q_c = torch.tensor((np.mean(P_c, axis=0) + lambda_c * np.mean(P_cf, axis=0)) / (1 + lambda_c)
    Q_c.requires_grad_(True)
    
    # Convert numpy arrays to PyTorch tensors
    P_c_tensor = torch.tensor(P_c, dtype=torch.float32)
    P_cf_tensor = torch.tensor(P_cf, dtype=torch.float32)
    
    optimizer = optim.Adam([Q_c], lr=lr)
    
    for _ in range(n_iter):
        # Compute W2 distances (Sinkhorn approximation)
        M_c = torch.cdist(Q_c.unsqueeze(0), P_c_tensor)**2
        W_c = sinkhorn2(torch.ones(1)/1, torch.ones(len(P_c))/len(P_c), M_c, reg=0.1)
        
        M_cf = torch.cdist(Q_c.unsqueeze(0), P_cf_tensor)**2
        W_cf = sinkhorn2(torch.ones(1)/1, torch.ones(len(P_cf))/len(P_cf), M_cf, reg=0.1)
        
        # Symmetry regularization (if optimizing Q_0 and Q_1 jointly)
        if 'Q_1' in globals():
            M_cf_Q0 = torch.cdist(Q_c.unsqueeze(0), P_cf_tensor)**2
            M_cf_Q1 = torch.cdist(Q_1.unsqueeze(0), P_cf_tensor)**2
            W_cf_Q0 = sinkhorn2(torch.ones(1)/1, torch.ones(len(P_cf))/len(P_cf), M_cf_Q0, reg=0.1)
            W_cf_Q1 = sinkhorn2(torch.ones(1)/1, torch.ones(len(P_cf))/len(P_cf), M_cf_Q1, reg=0.1)
            R = (W_cf_Q0 - W_cf_Q1)**2
        else:
            R = torch.tensor(0.0)
        
        # Total loss
        loss = W_c + lambda_c * W_cf + gamma * R
        
        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return Q_c.detach().numpy()

# Subsample to 400 points per distribution
X_0 = X_train[m.predict(X_train) == 0][:400]
X_1 = X_train[m.predict(X_train) == 1][:400]
X_cf = X_cf[:400]

# Compute barycenters
Q_0 = compute_barycenter(X_0, X_cf, lambda_c=0.5, gamma=0.1)
Q_1 = compute_barycenter(X_1, X_cf, lambda_c=0.5, gamma=0.1)

# --- Step 5: Train Surrogate Model ---
def wasserstein_distance(x, Q, P_ref, reg=0.1):
    """Compute W2(x, Q) using Sinkhorn, where Q is the barycenter."""
    M = torch.cdist(torch.tensor(x).unsqueeze(0), torch.tensor(P_ref))**2
    return sinkhorn2(torch.ones(1)/1, torch.ones(len(P_ref))/len(P_ref), M, reg=reg)

# Prepare training data
X_surrogate = np.vstack([X_0, X_1, X_cf])
y_surrogate = []

for x in X_surrogate:
    d0 = wasserstein_distance(x, Q_0, X_0)
    d1 = wasserstein_distance(x, Q_1, X_1)
    if d0 < d1 - 0.2:  # Threshold tau=0.2
        y_surrogate.append(0)
    elif d1 < d0 - 0.2:
        y_surrogate.append(1)
    else:
        y_surrogate.append(0.5)

# Train logistic regression
logreg = LogisticRegression(class_weight={0:1, 0.5:0.5, 1:1})
logreg.fit(X_surrogate, y_surrogate)

# --- Step 6: Evaluate Fidelity ---
y_pred_m = m.predict(X_test)
y_pred_logreg = logreg.predict(X_test)
fidelity = accuracy_score(y_pred_m, y_pred_logreg)
print(f"Fidelity: {fidelity:.3f}")
