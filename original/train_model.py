import os.path
import pickle

import lightgbm as lgb
import numpy as np
import shap  # https://github.com/slundberg/shap
import shapreg  # https://github.com/iancovert/shapley-regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


if os.path.isfile("census_model.pkl"):
    print("Loading saved model")
    with open("census_model.pkl", "rb") as f:
        model = pickle.load(f)

else:
    # Setup
    params = {
        "max_bin": 512,
        "learning_rate": 0.05,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 10,
        "verbose": -1,
        "min_data": 100,
        "boost_from_average": True,
    }

    # More setup
    d_train = lgb.Dataset(X_train, label=Y_train)
    d_val = lgb.Dataset(X_val, label=Y_val)

    # Train model
    model = lgb.train(
        params,
        d_train,
        10000,
        valid_sets=[d_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(1000)],
    )

    # Save model
    with open("census_model.pkl", "wb") as f:
        pickle.dump(model, f)
