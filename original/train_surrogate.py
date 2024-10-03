import argparse
import os.path
import pickle
import time

import lightgbm as lgb
import numpy as np
import shap  # https://github.com/slundberg/shap
import shapreg  # https://github.com/iancovert/shapley-regression
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import wandb
from fastshap.surrogate import Surrogate
from fastshap.utils import KLDivLoss, MaskLayer1d

parser = argparse.ArgumentParser(description="Training Adult")
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument("--project_name", type=str, default=None)
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--original_model", type=str, default="")

# arguments for the surrogate
parser.add_argument("--validation_samples", type=int, default=None)
parser.add_argument("--validation_batch_size", type=int, default=None)
parser.add_argument("--rnd", type=int, default=None)
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--test", type=bool, default=False)


def setup_wandb(args, rnd):
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project=args.project_name,
        dir="/raid/lcorbucci/wandb_tmp",
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
if args.sweep:
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=rnd
    )
else:
    X_val = None
    Y_val = None

# Data scaling
num_features = X_train.shape[1]
feature_names = X_train.columns.tolist()
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train.values)
if X_val:
    X_val = ss.transform(X_val.values)
X_test = ss.transform(X_test.values)

if os.path.isfile("census model.pkl"):
    print("Loading saved model")
    with open("census model.pkl", "rb") as f:
        model = pickle.load(f)

# Select device
device = torch.device("cuda")


# Create surrogate model
surr = nn.Sequential(
    MaskLayer1d(value=0, append=True),
    nn.Linear(2 * num_features, 128),
    nn.ELU(inplace=True),
    nn.Linear(128, 128),
    nn.ELU(inplace=True),
    nn.Linear(128, 2),
).to(device)

# Set up surrogate object
surrogate = Surrogate(surr, num_features)


# Set up original model
def original_model(x):
    pred = model.predict(x.cpu().numpy())
    pred = np.stack([1 - pred, pred]).T
    return torch.tensor(pred, dtype=torch.float32, device=x.device)


wandb_run = setup_wandb(args, rnd)

if args.sweep:
    # Train
    surrogate.train_original_model(
        X_train,
        X_val,
        original_model,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        loss_fn=KLDivLoss(),
        validation_samples=args.validation_samples,  # this number is multiplied with the length of the validation dataset
        validation_batch_size=args.validation_batch_size,  # size of the mini batch
        verbose=True,
        lr=args.lr,
        wandb=wandb_run,
        optimizer_name=args.optimizer,
    )
else:
    # Train
    surrogate.train_original_model(
        train_data=X_train,
        val_data=None,
        test_data=X_test,
        original_model=original_model,
        batch_size=1000,
        max_epochs=100,
        loss_fn=KLDivLoss(),
        test_samples=args.validation_samples,
        test_batch_size=args.validation_batch_size,
        test=True,
        verbose=True,
        lr=0.001,
        wandb=wandb_run,
        optimizer_name="adam",
    )
if args.save_model:
    # Save surrogate
    surr.cpu()
    torch.save(surr, f"{args.model_name}.pt")
    surr.to(device)
