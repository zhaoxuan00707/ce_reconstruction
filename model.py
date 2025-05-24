import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from ot import sinkhorn2
from scipy.stats import multivariate_normal

# --- Step 1: Load and Preprocess Adult Dataset ---
adult = fetch_openml(name='adult', version=2, as_frame=True)
X = adult.data
y = (adult.target == '>50K').astype(int)  # Binary classification

# Select numerical features and scale
num_features = ['age', 'education-num', 'hours-per-week']
X = X[num_features].astype(float)
X = StandardScaler().fit_transform(X)

# Split into target model training (m) and evaluation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step 2: Pretrain Target Model (m) ---
m = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(f"Target model accuracy: {accuracy_score(y_test, m.predict(X_test)):.3f}")

# --- Step 3: Generate Counterfactuals (D_cf) ---
def generate_counterfactuals(X, m, n_samples=400):
    """Simple counterfactual generator: perturb samples to cross decision boundary."""
    X_cf = []
    for x in X[m.predict(X) == 0][:n_samples]:  # Only class 0 samples
        x_cf = x.copy()
        x_cf[0] += 1.0  # Perturb 'age' to flip prediction (simplistic)
        if m.predict([x_cf])[0] == 1:  # Ensure it flips
            X_cf.append(x_cf)
    return np.array(X_cf)

X_cf = generate_counterfactuals(X_train, m, n_samples=400)
y_cf = np.full(len(X_cf), 0.5)  # Label counterfactuals as 0.5

# --- Step 4: Compute Wasserstein Barycenters (Q_0, Q_1) ---
def compute_barycenter(X_class, X_cf, lambda_c=0.5, reg=0.1):
    """Compute barycenter between class and counterfactual distributions."""
    # Empirical distributions (subset to 400 samples)
    X_class = X_class[:400]
    X_cf = X_cf[:400]
    
    # Mean and covariance for barycenter initialization
    mean_class = np.mean(X_class, axis=0)
    mean_cf = np.mean(X_cf, axis=0)
    Q_mean = (mean_class + lambda_c * mean_cf) / (1 + lambda_c)
    
    # Approximate covariance (simplified)
    Q_cov = np.cov(X_class, rowvar=False) + np.cov(X_cf, rowvar=False)
    return Q_mean, Q_cov

# Get class subsets
X_0 = X_train[m.predict(X_train) == 0][:400]
X_1 = X_train[m.predict(X_train) == 1][:400]

# Compute barycenters
Q0_mean, Q0_cov = compute_barycenter(X_0, X_cf, lambda_c=0.5)
Q1_mean, Q1_cov = compute_barycenter(X_1, X_cf, lambda_c=0.5)

# --- Step 5: Train Surrogate Model (logreg) ---
def wasserstein_distance(x, mean, cov, reg=0.1):
    """Approximate W2 distance using Sinkhorn."""
    x_dist = multivariate_normal(mean=x, cov=np.eye(len(x)) * 1e-3)
    q_dist = multivariate_normal(mean=mean, cov=cov)
    samples_x = x_dist.rvs(100).reshape(-1, len(x))
    samples_q = q_dist.rvs(100)
    M = np.linalg.norm(samples_x[:, None] - samples_q[None], axis=2)**2
    return sinkhorn2(np.ones(100)/100, np.ones(100)/100, M, reg=reg)

# Prepare training data for surrogate
X_surrogate = np.vstack([X_0, X_1, X_cf])
y_surrogate = np.hstack([np.zeros(len(X_0)), np.ones(len(X_1)), y_cf])

# Assign soft labels based on W2 distances
y_soft = []
for x in X_surrogate:
    d0 = wasserstein_distance(x, Q0_mean, Q0_cov)
    d1 = wasserstein_distance(x, Q1_mean, Q1_cov)
    if d0 < d1 - 0.2:  # Threshold tau=0.2
        y_soft.append(0)
    elif d1 < d0 - 0.2:
        y_soft.append(1)
    else:
        y_soft.append(0.5)

# Train logistic regression
logreg = LogisticRegression(class_weight={0:1, 0.5:0.5, 1:1})
logreg.fit(X_surrogate, y_soft)

# --- Step 6: Evaluate Fidelity ---
y_pred_m = m.predict(X_test)
y_pred_logreg = logreg.predict(X_test)
fidelity = accuracy_score(y_pred_m, y_pred_logreg)
print(f"Fidelity (agreement with m): {fidelity:.3f}")
