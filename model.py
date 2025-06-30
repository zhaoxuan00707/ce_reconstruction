import pandas as pd
import numpy as np
from torch.nn.functional import log_softmax
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import dice_ml
from dice_ml import Dice

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = pd.read_csv(
    "./adult.data",
    names=COLUMN_NAMES, na_values=" ?", skipinitialspace=True
)

df.drop(columns=['native-country'])
df.dropna(inplace=True)
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# -------------------------------
# 2. Define features
# -------------------------------
numerical_cols = [
    'age', 'fnlwgt', 'education-num', 'capital-gain',
    'capital-loss', 'hours-per-week'
]

categorical_cols = [
    'workclass', 'education', 'marital-status',
    'occupation', 'relationship', 'race', 'sex'
]

X = df.drop("income", axis=1)
y = df["income"]

# -------------------------------
# 3. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# -------------------------------
# 4. Preprocessing + model pipeline
# -------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

clf.fit(X_train, y_train)

data_dice = dice_ml.Data(
    dataframe=df,
    continuous_features=numerical_cols,
    outcome_name='income'
)

model_dice = dice_ml.Model(model=clf, backend="sklearn")

dice_exp = Dice(data_dice, model_dice, method="genetic")

# -------------------------------
# 6. Generate counterfactuals
# -------------------------------
# Select a test instance (from raw data)
query_instance = X_test.iloc[[0]]
#query_instance['income'] = clf.predict(query_instance)[0]  # Add prediction for DiCE

# Generate counterfactuals
cf = dice_exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")
cf.visualize_as_dataframe()

# 1. Predict on the entire test set
y_pred = clf.predict(X_test)

# 2. Select 400 samples predicted as class 0
pred_0_indices = X_test[y_pred == 0].index[:20]
samples_0 = X_test.loc[pred_0_indices]

print(f"Selected {len(samples_0)} samples with predicted class 0.")

# 3. Select 400 samples predicted as class 1
pred_1_indices = X_test[y_pred == 1].index[:20]
samples_1 = X_test.loc[pred_1_indices]

print(f"Selected {len(samples_1)} samples with predicted class 1.")


# 4. generate 400 samples predicted as class 0 to counterfactuals
all_cfs_dfs = []

for i, instance in samples_0.iterrows():
    instance_df = instance.to_frame().T
    try:
        cf_obj = dice_exp.generate_counterfactuals(instance_df, total_CFs=1, desired_class=1)  # flip to class 1
        cf_df = cf_obj.cf_examples_list[0].final_cfs_df
        cf_df['query_index'] = i
        all_cfs_dfs.append(cf_df)
    except Exception as e:
        print(f"Failed to generate CF for instance {i}: {e}")

# 4. Concatenate all counterfactuals into one DataFrame
counterfactuals_df = pd.concat(all_cfs_dfs, ignore_index=True)

print(f"Generated {len(counterfactuals_df)} counterfactual samples predicted as class 1.")

print(counterfactuals_df)

import ot  # pip install POT


#X0 = samples_0.to_numpy()
#X1 = counterfactuals_df.to_numpy()






common_cols = samples_0.columns.intersection(samples_1.columns).intersection(counterfactuals_df.columns)

X0_vals = samples_0[common_cols]
X1_vals = samples_1[common_cols]
Xcf_vals = counterfactuals_df[common_cols]

# Concatenate all three
combined = pd.concat([X0_vals, X1_vals, Xcf_vals], axis=0)

# One-hot encode categoricals (pandas get_dummies will handle all categorical columns)
combined_encoded = pd.get_dummies(combined)

# Standardize numeric columns after encoding
scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined_encoded)

# Split back to individual arrays
X0_scaled = combined_scaled[:len(samples_0)]
X1_scaled = combined_scaled[len(samples_0):len(samples_0)+len(samples_1)]
Xcf_scaled = combined_scaled[len(samples_0)+len(samples_1):]
    
print("X0_scaled:", X0_scaled.shape)
print("Q0:", samples_0.shape)


def compute_barycenter(original_df, counterfactual_df, n_bary=30, weight_original=1.0, weight_cf=0.5):
    """
    Compute the Wasserstein barycenter between original and counterfactual samples.

    Parameters:
    - original_df (pd.DataFrame): Original instances (X0 or X1)
    - counterfactual_df (pd.DataFrame): Corresponding counterfactuals
    - n_bary (int): Number of support points in the barycenter
    - weight_original (float): Weight for the original distribution
    - weight_cf (float): Weight for the counterfactual distribution

    Returns:
    - barycenter_support (np.ndarray): Coordinates of barycenter support points
    """
    # Keep only common columns
    #common_cols = original_df.columns.intersection(counterfactual_df.columns)
    #X0 = original_df[common_cols].to_numpy()
    #X1 = counterfactual_df[common_cols].to_numpy()

    # Encode categoricals
    #combined = pd.concat([pd.DataFrame(X0), pd.DataFrame(X1)], axis=0)
    #combined_encoded = pd.get_dummies(combined)

    #X0_encoded = combined_encoded.iloc[:len(X0)].copy()
    #X1_encoded = combined_encoded.iloc[len(X0):].copy()

    # Standardize
    #scaler = StandardScaler()
    #X_all_scaled = scaler.fit_transform(np.vstack([X0_encoded, X1_encoded]))
    #X0_scaled = X_all_scaled[:len(X0)]
    #X1_scaled = X_all_scaled[len(X0):]

    # Normalize distribution weights
    weights = np.array([weight_original, weight_cf])
    weights = weights / weights.sum()

    # Uniform weights over samples
    measures_weights = [
        np.ones(len(X0_scaled)) / len(X0_scaled),
        np.ones(len(X1_scaled)) / len(X1_scaled)
    ]

    # Initialize barycenter support randomly
    np.random.seed(42)
    d =original_df.shape[1]
    print('d',X0_scaled.shape)
    X_init = np.random.randn(n_bary, d)

    barycenter_support = ot.lp.free_support_barycenter(
        measures_locations=[X0_scaled, X1_scaled],
        measures_weights=measures_weights,
        X_init=X_init,
        weights=weights,
        numItermax=100,
        stopThr=1e-7,
        verbose=True
    )

    return barycenter_support


print(compute_barycenter(X0_scaled, Xcf_scaled).shape)


def predict_wasserstein_label(x, barycenter_0, barycenter_1, tau=0.0):
    """
    Predict label based on Wasserstein-2 distance to barycenters.

    Args:
        x (np.ndarray): Test sample of shape (d,)
        barycenter_0 (np.ndarray): Support of class 0 barycenter
        barycenter_1 (np.ndarray): Support of class 1 barycenter
        tau (float): Margin threshold

    Returns:
        int: Predicted label (0 or 1)
    """
    w2_0 = np.min(np.linalg.norm(barycenter_0 - x, axis=1) ** 2)
    w2_1 = np.min(np.linalg.norm(barycenter_1 - x, axis=1) ** 2)

    if w2_0 < w2_1 - tau:
        return 0
    elif w2_1 < w2_0 - tau:
        return 1
    else:
        return 0  # or return None for ambiguous
    


#Wasserstein-2 distance squared between support points and empirical samples ---
def wasserstein2_squared(X, Y):
        # Convert pandas DataFrame to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    M = ot.dist(X, Y, metric='euclidean') ** 2
    return ot.emd2(a, b, M)

# --- 4. Full objective ---
def compute_total_loss(Q0, Q1, P0, P1, Pcf, lambda_0=0.5, lambda_1=0.5, gamma=0.3):

    w2_Q0_P0 = wasserstein2_squared(Q0, X0_scaled)
    w2_Q0_Pcf = wasserstein2_squared(Q0, Xcf_scaled)

    w2_Q1_P1 = wasserstein2_squared(Q1, X1_scaled)
    w2_Q1_Pcf = wasserstein2_squared(Q1, Xcf_scaled)

    # Symmetry regularization term
    sym_reg = (np.sqrt(w2_Q0_Pcf) - np.sqrt(w2_Q1_Pcf)) ** 2

    total_loss = (
        w2_Q0_P0 +
        lambda_0 * w2_Q0_Pcf +
        w2_Q1_P1 +
        lambda_1 * w2_Q1_Pcf +
        gamma * sym_reg
    )
    return total_loss


max_iters = 10
lambda_0, lambda_1, gamma = 1.0, 1.0, 10.0

Q0 = None
Q1 = None

for iter in range(max_iters):
    # Update Q0 barycenter with weights incorporating counterfactuals
    Q0 = compute_barycenter(X0_scaled, Xcf_scaled)
    print("q0 shape:", Q0.shape)

    # Update Q1 barycenter similarly
    Q1 = compute_barycenter(X1_scaled, Xcf_scaled)

    # Compute loss with current Q0, Q1
    loss = compute_total_loss(Q0, Q1,X0_scaled, X1_scaled, Xcf_scaled)

    print(f"Iteration {iter + 1}/{max_iters} - Loss: {loss:.4f}")
