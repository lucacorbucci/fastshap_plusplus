import argparse
import os.path
import pickle
import time

import lightgbm as lgb
import numpy as np
import shap  # https://github.com/slundberg/shap  # https://github.com/slundberg/shap
import shapreg  # https://github.com/iancovert/shapley-regression  # https://github.com/iancovert/shapley-regression
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fastshap import FastSHAP, KLDivLoss, Surrogate
from fastshap.utils import MaskLayer1d

parser = argparse.ArgumentParser(description="Training Adult")
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument("--project_name", type=str, default=None)
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--rnd", type=int, default=None)

# arguments for the explainer
parser.add_argument("--validation_samples", type=int, default=None)
parser.add_argument("--normalization", type=str, default="additive")
parser.add_argument("--num_samples", type=int, default=32)
parser.add_argument("--surrogate", type=str, default="")
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--paired_sampling", type=bool, default=True)
parser.add_argument("--eff_lambda", type=bool, default=True)


def setup_wandb(args, rnd):
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project=args.project_name,
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "rnd": rnd,
        },
    )
    return wandb_run


args = parser.parse_args()

# based on current time generate a random seed
if args.rnd is None:
    rnd = int(str(time.time()).split(".")[1]) * 42
else:
    rnd = args.rnd

# Load and split data
X_train, X_test, Y_train, Y_test = train_test_split(
    *shap.datasets.adult(), test_size=0.2, random_state=42
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=rnd
)

# Data scaling
num_features = X_train.shape[1]
feature_names = X_train.columns.tolist()
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train.values)
X_val = ss.transform(X_val.values)
X_test = ss.transform(X_test.values)

# Select device
device = torch.device("cuda")

if os.path.isfile("census_model.pkl"):
    print("Loading saved model")
    with open("census_model.pkl", "rb") as f:
        model = pickle.load(f)


# Check for model
if os.path.isfile("surrogato_nuovo.pt"):
    print("Loading saved surrogate model")
    surr = torch.load("surrogato_nuovo.pt").to(device)
    surrogate = Surrogate(surr, num_features)


# Create explainer model
explainer = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 2 * num_features),
).to(device)

# Set up FastSHAP object
fastshap = FastSHAP(
    explainer, surrogate, normalization=args.normalization, link=nn.Softmax(dim=-1)
)
wandb_run = setup_wandb(args, rnd)

# Train
fastshap.train(
    X_train,
    X_val,
    batch_size=args.batch_size,
    num_samples=args.num_samples,
    max_epochs=args.epochs,
    verbose=True,
    validation_samples=args.validation_samples,
    wandb=wandb,
    optimizer_name=args.optimizer,
    paired_sampling=args.paired_sampling,
    eff_lambda=args.eff_lambda,
)

if args.save_model:
    # Save explainer
    explainer.cpu()
    torch.save(explainer, f"{args.model_name}.pt")
    explainer.to(device)
