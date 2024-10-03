import os
import pickle

import dill
import matplotlib.pyplot as plt
import numpy as np
import shap  # https://github.com/slundberg/shap
import shapreg  # https://github.com/iancovert/shapley-regression
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fastshap import Surrogate

# Load and split data
X_train, X_test, Y_train, Y_test = train_test_split(
    *shap.datasets.adult(), test_size=0.2, random_state=42
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=0
)

# Data scaling
num_features = X_train.shape[1]
feature_names = X_train.columns.tolist()
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train.values)
X_val = ss.transform(X_val.values)
X_test = ss.transform(X_test.values)


if os.path.isfile("census model.pkl"):
    print("Loading saved model")
    with open("census model.pkl", "rb") as f:
        model = pickle.load(f)


# Select device
device = torch.device("cpu")

# Check for model
if os.path.isfile("census surrogate.pt"):
    print("Loading saved surrogate model")
    surr = torch.load("census surrogate.pt").to(device)
    surrogate = Surrogate(surr, num_features)


# Setup for KernelSHAP
def imputer(x, S):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.float32, device=device)
    pred = surrogate(x, S).softmax(dim=-1)
    return pred.cpu().data.numpy()


all_samples_values = []
all_samples_results = []
explained_indexes = []
# compute all the shap Values
for index in range(len(X_test)):
    print(f"Explaining {index} of {len(X_test)}")
    x = X_test[index]
    y = int(Y_test[index])

    try:
        game = shapreg.games.PredictionGame(imputer, x)
        shap_values, all_results = shapreg.shapley.ShapleyRegression(
            game,
            batch_size=32,
            paired_sampling=False,
            detect_convergence=True,
            bar=True,
            return_all=True,
        )
        explained_indexes.append(index)
        all_samples_values.append(shap_values)
        all_samples_results.append(all_results)
    except:
        continue


# Save results
with open("shap_values.pkl", "wb") as f:
    dill.dump(all_samples_values, f)

with open("results.pkl", "wb") as f:
    dill.dump(all_samples_results, f)

with open("indexes.pkl", "wb") as f:
    dill.dump(explained_indexes, f)
