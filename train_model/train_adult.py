import argparse
import os.path
import pickle
import warnings
from copy import deepcopy

# import shapreg  # https://github.com/iancovert/shapley-regression
import numpy as np
import shap  # https://github.com/slundberg/shap
import torch
import torch.nn as nn

# import Functional torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    TensorDataset,
)
from tqdm.auto import tqdm

warnings.simplefilter("ignore")
import wandb

parser = argparse.ArgumentParser(description="Training Adult")
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument("--clipping", type=float, default=None)
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--node_shuffle_seed", type=int, default=None)
parser.add_argument("--project_name", type=str, default=None)
parser.add_argument("--save_model", type=bool, default=False)


def create_dataset(data, labels):
    # convert Y to tensor
    labels = torch.tensor([0 if item == False else 1 for item in labels])

    # convert data to tensor
    data = torch.tensor(data, dtype=torch.float32)
    return TensorDataset(data, labels)


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
            "node_shuffle_seed": args.node_shuffle_seed,
            "optimizer": args.optimizer,
        },
    )
    return wandb_run


def get_optimizer(optimizer, model, lr):
    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not recognized")


def eval_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            predicted = outputs.argmax(dim=1, keepdim=True)
            total += target.size(0)
            correct += (predicted == target.view_as(predicted)).sum().item()
    acc = correct / total
    return acc


def train_model(
    model,
    optimizer,
    train_loader,
    epochs,
    val_loader=None,
    test_loader=None,
    args=None,
    device=None,
):
    wandb_run = setup_wandb(args)

    for epoch in tqdm(range(epochs)):
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb_run.log({"train_loss": loss.item()})

        # test on the training set
        train_acc = eval_model(model, train_loader, device)
        wandb_run.log({"train_accuracy": train_acc, "epoch": epoch})

        # validate on the validation set
        if val_loader is not None:
            val_acc = eval_model(model, val_loader, device)
            wandb_run.log({"validation_accuracy": val_acc, "epoch": epoch})

        # test on the test set
        if test_loader is not None:
            test_acc = eval_model(model, test_loader, device)
            wandb_run.log({"test_accuracy": test_acc, "epoch": epoch})


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def data_scaling(X_train, X_val, X_test):
    # Data scaling
    num_features = X_train.shape[1]
    feature_names = X_train.columns.tolist()
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    if X_val is not None:
        X_val = ss.transform(X_val)
    if X_test is not None:
        X_test = ss.transform(X_test)

    return X_train, X_val, X_test


def prepare_data(args):
    # Load and split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        *shap.datasets.adult(), test_size=0.2, random_state=42
    )
    X_val, Y_val = None, None
    if args.sweep:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=0.2, random_state=args.node_shuffle_seed
        )
        X_test = None
        Y_test = None
    X_train, X_val, X_test = data_scaling(X_train, X_val, X_test)

    # create a dataloader for training using X_train and Y_train
    train_set = create_dataset(X_train, Y_train)
    if X_val is not None:
        val_set = create_dataset(X_val, Y_val)
    else:
        val_set = None

    if X_test is not None:
        test_set = create_dataset(X_test, Y_test)
    else:
        test_set = None

    return train_set, val_set, test_set, X_train.shape[1]


if __name__ == "__main__":
    args = parser.parse_args()

    train_set, val_set, test_set, feature_size = prepare_data(args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        if val_set is not None
        else None
    )
    test_loader = (
        DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        if test_set is not None
        else None
    )

    model = Model(feature_size, 128, 2)

    optimizer = get_optimizer(args.optimizer, model, args.lr)

    privacy_engine = PrivacyEngine()

    if args.epsilon is not None:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=args.clipping,
            target_epsilon=args.epsilon,
            target_delta=1e-5,
            epochs=args.epochs,
        )
    else:
        surr, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=10000000000,
            noise_multiplier=0,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_model(
        model,
        optimizer,
        train_loader,
        args.epochs,
        val_loader,
        test_loader,
        args,
        device,
    )

    if args.save_model:
        # save model with and without state dict
        torch.save(model, "model.pth")
        torch.save(model.state_dict(), "model_state_dict.pth")
