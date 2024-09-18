import os.path
import pickle
import warnings
from copy import deepcopy

import dill
import lightgbm as lgb

# import shapreg  # https://github.com/iancovert/shapley-regression
import numpy as np
import shap  # https://github.com/slundberg/shap
import torch
import torch.nn as nn

# import Functional torch
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import (
    DataLoader,
    Dataset,
    TensorDataset,
)

from fastshap.utils import DatasetRepeat, ShapleySampler
from utils import prepare_data

warnings.simplefilter("ignore")
import argparse

import wandb
from fastshap.fastshap_dp import (
    FastSHAP,
    calculate_grand_coalition,
    generate_validation_data,
)
from fastshap.surrogate_dp import SurrogateDP
from fastshap.utils import setup_data

parser = argparse.ArgumentParser(description="Training Adult")
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument("--clipping", type=float, default=None)
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--project_name", type=str, default=None)
parser.add_argument("--save_model", type=bool, default=False)

# arguments for the explainer
parser.add_argument("--validation_samples", type=int, default=None)
parser.add_argument("--normalization", type=str, default="additive")
parser.add_argument("--num_samples", type=int, default=32)
parser.add_argument("--surrogate", type=str, default="")
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--dataset_name", type=str, default="adult")


def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not found")


# def prepare_data():
#     # Load and split data
#     X_train, X_test, Y_train, Y_test = train_test_split(
#         *shap.datasets.adult(), test_size=0.2, random_state=42
#     )
#     X_train, X_val, Y_train, Y_val = train_test_split(
#         X_train, Y_train, test_size=0.2, random_state=0
#     )

#     # Data scaling
#     num_features = X_train.shape[1]
#     feature_names = X_train.columns.tolist()
#     ss = StandardScaler()
#     ss.fit(X_train)
#     X_train = ss.transform(X_train.values)
#     X_val = ss.transform(X_val.values)
#     X_test = ss.transform(X_test.values)

#     return X_train, X_val, X_test, Y_train, Y_val, Y_test, num_features, feature_names


def prepare_dataset_for_explainer(
    train_data,
    val_data,
    batch_size,
    imputer,
    num_samples,
    link,
    device,
    validation_samples,
    num_players,
    validation_seed=None,
):
    # Set up train dataset.
    if isinstance(train_data, np.ndarray):
        x_train = torch.tensor(train_data, dtype=torch.float32)
        train_set = TensorDataset(x_train)
    elif isinstance(train_data, torch.Tensor):
        train_set = TensorDataset(train_data)
    elif isinstance(train_data, Dataset):
        train_set = train_data
    else:
        raise ValueError("train_data must be np.ndarray, torch.Tensor or " "Dataset")

    # Set up validation dataset.
    if isinstance(val_data, np.ndarray):
        x_val = torch.tensor(val_data, dtype=torch.float32)
        val_set = TensorDataset(x_val)
    elif isinstance(val_data, torch.Tensor):
        val_set = TensorDataset(val_data)
    elif isinstance(val_data, Dataset):
        val_set = val_data
    else:
        raise ValueError("train_data must be np.ndarray, torch.Tensor or " "Dataset")

    num_workers = 0

    imputer = imputer.to(device)

    # Grand coalition value.
    grand_train = calculate_grand_coalition(
        train_set,
        imputer,
        batch_size * num_samples,
        link,
        device,
        num_workers,
        num_features,
    ).cpu()
    grand_val = calculate_grand_coalition(
        val_set,
        imputer,
        batch_size * num_samples,
        link,
        device,
        num_workers,
        num_features,
    ).cpu()

    # Null coalition.
    with torch.no_grad():
        zeros = torch.zeros(1, num_players, dtype=torch.float32, device=device)
        null = link(imputer((train_set[0][0].unsqueeze(0).to(device), zeros)))
        if len(null.shape) == 1:
            null = null.reshape(1, 1)

    # Set up train loader.
    train_set = DatasetRepeat([train_set, TensorDataset(grand_train)])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
    )

    # Generate validation data.
    sampler = ShapleySampler(num_players)
    if validation_seed is not None:
        torch.manual_seed(validation_seed)
    val_S, val_values = generate_validation_data(
        val_set,
        imputer,
        validation_samples,
        sampler,
        batch_size * num_samples,
        link,
        device,
        num_workers,
        num_players,
    )

    # Set up val loader.
    val_set = DatasetRepeat([val_set, TensorDataset(grand_val, val_S, val_values)])
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * num_samples,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, grand_train, grand_val, null, sampler


def load_model(model_name):
    if os.path.isfile(model_name):
        model = torch.load(model_name)
    else:
        raise FileNotFoundError("Model not found")
    return model


def setup_wandb(args):
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project=args.project_name,
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "epsilon": args.epsilon,
            "gradnorm": args.clipping,
            "optimizer": args.optimizer,
        },
    )
    return wandb_run


args = parser.parse_args()

wandb_run = setup_wandb(args)


surrogate = load_model(f"{args.surrogate}")

imputer = surrogate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


(
    _,
    _,
    _,
    X_train,
    X_val,
    X_test,
    Y_train,
    Y_val,
    Y_test,
    num_features,
    feature_names,
) = prepare_data(args)


train_loader, val_loader, grand_train, grand_val, null, sampler = (
    prepare_dataset_for_explainer(
        X_train,
        X_val,
        args.batch_size,
        imputer,
        num_samples=args.num_samples,
        link=nn.Softmax(dim=-1),
        device=device,
        validation_samples=args.validation_samples,
        num_players=num_features,
    )
)

# else:
# Create explainer model
explainer = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 2 * num_features),
).to(device)

explainer = ModuleValidator.fix(explainer)
ModuleValidator.validate(explainer, strict=False)
privacy_engine = PrivacyEngine()
optimizer = get_optimizer(args.optimizer, explainer, args.lr)

if args.epsilon:
    explainer, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=explainer,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=args.clipping,
        target_epsilon=args.epsilon,
        target_delta=1e-5,
        epochs=args.epochs,
    )
else:
    explainer, optimizer, train_loader = privacy_engine.make_private(
        module=explainer,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=10000000000,
        noise_multiplier=0,
    )
print("Created private model")


# Set up FastSHAP object
fastshap = FastSHAP(
    explainer,
    surrogate,
    num_features=num_features,
    normalization=args.normalization,
    link=nn.Softmax(dim=-1),
)

# Train
fastshap.train(
    train_loader,
    val_loader,
    grand_train,
    grand_val,
    null,
    batch_size=args.batch_size,
    num_samples=args.num_samples,
    lr=args.lr,
    max_epochs=args.epochs,
    validation_samples=args.validation_samples,
    verbose=True,
    optimizer=optimizer,
    wandb=wandb,
    sampler=sampler,
    bar=True,
)

if args.save_model:
    torch.save(explainer, f"./{args.model_name}.pt")
    # dill.dump(fastshap, open(f"./{args.model_name}.pkl", "wb"))
