import argparse
import random
import time
import warnings

import numpy as np

# import shapreg  # https://github.com/iancovert/shapley-regression
import torch
import torch.nn as nn

# import Functional torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from opacus import PrivacyEngine
from torch.utils.data import (
    DataLoader,
)
from tqdm.auto import tqdm

from utils import prepare_data

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
parser.add_argument("--validation_seed", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--project_name", type=str, default=None)
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--model_name", type=str, default="model.pth")
parser.add_argument("--dataset_name", type=str, default="adult")


def setup_wandb(args):
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project=args.project_name,
        dir="/raid/lcorbucci/wandb_tmp",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "epsilon": args.epsilon,
            "gradnorm": args.clipping,
            "validation_seed": args.validation_seed,
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
            labels = labels.type(torch.LongTensor)
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return output


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.validation_seed is None:
        validation_seed = int(str(time.time()).split(".")[1]) * args.seed

        args.validation_seed = validation_seed

    if args.dataset_name == "adult" or args.dataset_name == "dutch":
        (
            train_set,
            val_set,
            test_set,
            _,
            _,
            _,
            _,
            _,
            _,
            feature_size,
            _,
        ) = prepare_data(args)
    elif args.dataset_name == "mnist":
        (
            train_set,
            val_set,
            test_set,
        ) = prepare_data(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    if args.dataset_name == "adult" or args.dataset_name == "dutch":
        model = Model(feature_size, 128, 2)
    elif args.dataset_name == "mnist":
        model = Net()

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

    print("Training model with Validation set: ", val_set)
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
        torch.save(model, f"{args.model_name}.pth")
