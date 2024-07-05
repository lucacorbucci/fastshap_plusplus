import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import shap  # https://github.com/slundberg/shap
import shapreg  # https://github.com/iancovert/shapley-regression
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_model(model_name):
    if os.path.isfile(model_name):
        model = torch.load(model_name)
    else:
        raise FileNotFoundError("Model not found")
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# surrogate_DP = load_model("./train_explainer/surrogate_dp.pt").to(device)
surrogate = load_model("./train_explainer/census surrogate.pt").to(device)

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


# Setup for KernelSHAP
def imputer(x, S):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.float32, device=device)
    pred = surrogate((x, S)).softmax(dim=-1)
    return pred.cpu().data.numpy()

    # Select example


all_shap_values = []
all_results = []
all_indexes = []

for i in range(len(X_test) - 1):
    # ind = np.random.choice(len(X_test))
    x = X_test[i : i + 1]
    y = int(Y_test[i])

    # Run KernelSHAP to convergence
    try:
        game = shapreg.games.PredictionGame(imputer, x)
        shap_values, results = shapreg.shapley.ShapleyRegression(
            game,
            batch_size=32,
            paired_sampling=False,
            detect_convergence=True,
            bar=True,
            return_all=True,
        )
        all_shap_values.append(shap_values)
        all_results.append(results)
        all_indexes.append(i)
    except:
        continue


# Save results
with open("shap_values.pkl", "wb") as f:
    dill.dump(all_shap_values, f)

with open("results.pkl", "wb") as f:
    dill.dump(all_results, f)

with open("indexes.pkl", "wb") as f:
    dill.dump(all_indexes, f)
