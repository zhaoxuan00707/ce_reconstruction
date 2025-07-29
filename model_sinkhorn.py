import pandas as pd
import random
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



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For deterministic behavior in some torch operations (e.g., convolutions)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call it before any randomness
set_seed(42)

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

# Fit only on training data
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('dummy', 'passthrough')  # no classifier
])

#pipeline.fit(X_train)

# Transform train and test
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

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
pred_0_indices = X_test[y_pred == 0].index[:400]
samples_0 = X_test.loc[pred_0_indices]

print(f"Selected {len(samples_0)} samples with predicted class 0.")

# 3. Select 400 samples predicted as class 1
pred_1_indices = X_test[y_pred == 1].index[:400]
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






#common_cols = samples_0.columns.intersection(samples_1.columns).intersection(counterfactuals_df.columns)

#X0_vals = samples_0[common_cols]
#X1_vals = samples_1[common_cols]
#Xcf_vals = counterfactuals_df[common_cols]

# Concatenate all three
#combined = pd.concat([X0_vals, X1_vals, Xcf_vals], axis=0)

# One-hot encode categoricals (pandas get_dummies will handle all categorical columns)
#combined_encoded = pd.get_dummies(combined)

# Standardize numeric columns after encoding
#scaler = StandardScaler()
#combined_scaled = scaler.fit_transform(combined_encoded)

# Split back to individual arrays
X0_scaled = pipeline.transform(samples_0)
X1_scaled = pipeline.transform(samples_1)


Xcf_scaled = pipeline.transform(counterfactuals_df)
    
print("X0_scaled:", X0_scaled.shape)
print("X1_scaled:", X1_scaled.shape)
print("Xcf_scaled:", Xcf_scaled.shape)

from geomloss import SamplesLoss
from itertools import product


sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)

def wasserstein2_squared(X, Y):

    return sinkhorn_loss(X, Y)

if hasattr(X0_scaled, "toarray"):
    X0_scaled = X0_scaled.toarray()

if hasattr(X1_scaled, "toarray"):
    X1_scaled = X1_scaled.toarray()

if hasattr(Xcf_scaled, "toarray"):
    Xcf_scaled = Xcf_scaled.toarray()


X0_scaled = torch.tensor(X0_scaled, dtype=torch.float32)
X1_scaled = torch.tensor(X1_scaled, dtype=torch.float32)
Xcf_scaled = torch.tensor(Xcf_scaled, dtype=torch.float32)

def compute_lambda_c(P_cf, P_0, P_1,c): #this def is from euqation 2 from method section


    P_c = P_0 if c == 0 else P_1
    P_1_c = P_1 if c == 0 else P_0

    w_cf_c = wasserstein2_squared(P_cf, P_c)
    w_cf_1_c = wasserstein2_squared(P_cf, P_1_c)

    denominator = w_cf_c + w_cf_1_c
    if denominator == 0:
        return 0.5  # avoid divide-by-zero, assume uniform confidence
    lambda_c = w_cf_1_c / denominator
    return lambda_c

print("lambda_0:", compute_lambda_c(Xcf_scaled,X0_scaled,X1_scaled,0 ))

lambda_0=compute_lambda_c(Xcf_scaled,X0_scaled,X1_scaled,0 )
lambda_1=compute_lambda_c(Xcf_scaled,X0_scaled,X1_scaled,1 )



def compute_total_loss(Q0, Q1, lambda_0=lambda_0, lambda_1=lambda_1, gamma=0.1): #this def is from euqation 5 from method section


    w2_Q0_P0 = wasserstein2_squared(Q0, X0_scaled)
    w2_Q0_Pcf = wasserstein2_squared(Q0, Xcf_scaled)

    w2_Q1_P1 = wasserstein2_squared(Q1, X1_scaled)
    w2_Q1_Pcf = wasserstein2_squared(Q1, Xcf_scaled)

    # Symmetry regularization term
    sym_reg = (torch.sqrt(w2_Q0_Pcf) - torch.sqrt(w2_Q1_Pcf)) ** 2

    total_loss = (
        w2_Q0_P0 +
        lambda_0 * w2_Q0_Pcf +
        w2_Q1_P1 +
        lambda_1 * w2_Q1_Pcf +
        gamma * sym_reg
    )
    return total_loss

d=X0_scaled.shape[1]

# Initialize Q0, Q1 as tensors with gradients enabled
Q0 = torch.randn(400, d, requires_grad=True)  # N0 points in d dimensions
Q1 = torch.randn(400, d, requires_grad=True)

# Assume P0, P1, Pcf are fixed tensors (data)
# Define your regularizer, e.g. symmetry_reg(Q0, Q1)

optimizer = torch.optim.Adam([Q0, Q1], lr=1e-2)


import matplotlib.pyplot as plt

losses = []

for epoch in range(200):
    optimizer.zero_grad()
    loss = compute_total_loss(Q0, Q1)
    loss.backward()
    optimizer.step()




    losses.append(loss.item())  # Store loss as float

    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

# Plot after training
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Loss vs. Epoch")
plt.grid(True)
plt.show()

from scipy.sparse import csr_matrix

def predict_wasserstein_label(x, barycenter_0, barycenter_1, tau=0.0): #this def is from euqation 3 from method section
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
    if isinstance(x, csr_matrix):
       x = x.toarray()

    # Convert to torch tensor
    x = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
    #x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32)  # (1, d)
    #q0_tensor = torch.tensor(barycenter_0, dtype=torch.float32)
    #q1_tensor = torch.tensor(barycenter_1, dtype=torch.float32)
    #print(x.shape)
    #print(barycenter_0.shape)
    w2_0 = sinkhorn_loss(x, barycenter_0).item()
    w2_1 = sinkhorn_loss(x, barycenter_1).item()

    if w2_0 < w2_1 - tau:
        return 0
    elif w2_1 < w2_0 - tau:
        return 1
    else:
        return 'ambiguous'
    

    
def evaluate_accuracy(X_test_processed, Y_test, barycenter_0, barycenter_1):

    predictions = [
        predict_wasserstein_label(x, barycenter_0, barycenter_1)
        for x in X_test_processed
    ]


    predictions = np.array(predictions)
    accuracy = np.mean(predictions == Y_test)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


print(evaluate_accuracy(X_test_processed, y_test,Q0, Q1))

y_orig = clf.predict(X_test)

y_surrogate = np.array([
    predict_wasserstein_label(x, Q0, Q1)
    for x in X_test_processed
])

def fidelity_score(y_orig, y_surrogate):
    """
    Compute fidelity between original and surrogate predictions.

    Args:
        y_orig (np.ndarray): Predictions from original model
        y_surrogate (np.ndarray): Predictions from surrogate model

    Returns:
        float: Fidelity score
    """
    return np.mean(y_orig == y_surrogate)

# Compute and print fidelity
fidelity = fidelity_score(y_orig, y_surrogate)
print(f"Fidelity Score: {fidelity:.4f}")
