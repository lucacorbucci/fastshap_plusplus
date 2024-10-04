import os.path
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
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    TensorDataset,
)
from tqdm.auto import tqdm

from fastshap.image_surrogate_DP import ImageSurrogate_DP

# from fastshap import ImageSurrogate_DP, KLDivLoss
from fastshap.utils import (
    DatasetInputOnly,
    KLDivLoss,
    MaskLayer1d,
    MaskLayer2d,
    UniformSampler,
)
from utils import prepare_data

warnings.simplefilter("ignore")
import argparse

import wandb
from fastshap.surrogate_dp import SurrogateDP
from fastshap.utils import setup_data


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--surrogate_name", type=str, default="")
parser.add_argument("--dataset_name", type=str, default="adult")

# arguments for the surrogate
parser.add_argument("--validation_samples", type=int, default=None)
parser.add_argument("--validation_batch_size", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--validation_seed", type=int, default=None)


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
            "optimizer": args.optimizer,
            "validation_seed": args.validation_seed,
        },
    )
    return wandb_run


def load_model(model_name):
    if os.path.isfile(model_name):
        model = torch.load(model_name)
    else:
        raise FileNotFoundError("Model not found")
    return model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1)  # Change the input channels from 1 to 2
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
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x


def get_surrogate_model(args):
    # Create surrogate model
    if args.dataset_name == "adult" or args.dataset_name == "dutch":
        return nn.Sequential(
            MaskLayer1d(value=0, append=True),
            nn.Linear(2 * num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
    else:
        return nn.Sequential(MaskLayer2d(value=0, append=True), Net())


def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not found")


args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.validation_seed is None:
    validation_seed = int(str(time.time()).split(".")[1]) * args.seed
    args.validation_seed = validation_seed

if args.dataset_name == "adult" or args.dataset_name == "dutch":
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
elif args.dataset_name == "mnist":
    (
        train_set,
        val_set,
        test_set,
    ) = prepare_data(args)

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

model = load_model(args.model_name)
if args.dataset_name == "mnist":
    train_surr = DatasetInputOnly(train_set)
    # Set up train data loader.
    if isinstance(train_surr, torch.Tensor):
        train_set = TensorDataset(train_surr)
    elif isinstance(train_surr, Dataset):
        train_set = train_surr
    else:
        raise ValueError("train_data must be either tensor or a " "PyTorch Dataset")

    random_sampler = RandomSampler(
        train_set,
        replacement=True,
        num_samples=int(np.ceil(len(train_set) / args.batch_size)) * args.batch_size,
    )
    print(
        "Random sampler: ",
        int(np.ceil(len(train_set) / args.batch_size)) * args.batch_size,
    )
    batch_sampler = BatchSampler(
        random_sampler, batch_size=args.batch_size, drop_last=True
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        pin_memory=True,
        num_workers=0,
    )
    val_surr = None
    test_surr = None
    if val_set:
        val_surr = DatasetInputOnly(val_set)
    if test_set:
        test_surr = DatasetInputOnly(test_set)
elif args.dataset_name == "adult" or args.dataset_name == "dutch":
    train_loader, random_sampler, batch_sampler = setup_data(
        train_data=X_train, batch_size=args.batch_size
    )

surr = get_surrogate_model(args)
surr = ModuleValidator.fix(surr)
ModuleValidator.validate(surr, strict=False)

privacy_engine = PrivacyEngine()
optimizer = get_optimizer(args.optimizer, surr, args.lr)

if args.epsilon:
    surr, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=surr,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=args.clipping,
        target_epsilon=args.epsilon,
        target_delta=1e-5,
        epochs=args.epochs,
    )
else:
    surr, optimizer, train_loader = privacy_engine.make_private(
        module=surr,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=10000000000,
        noise_multiplier=0,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
surr = surr.to(device)
model = model.to(device)
# Set up surrogate object: we pass the model that we have defined as
# surrogate and the number of input features of the training dataset
# we used to train the black box.
if args.dataset_name == "mnist":
    surrogate = ImageSurrogate_DP(surr, width=28, height=28, superpixel_size=2)
else:
    surrogate = SurrogateDP(surr, num_features)
print("Created surrogate")

if args.dataset_name == "mnist":
    # Set up original model
    original_model = nn.Sequential(model, nn.Softmax(dim=1))
else:
    original_model = nn.Sequential(model, nn.Softmax(dim=1))

wandb_run = setup_wandb(args)

if args.dataset_name == "mnist":
    surrogate.train_original_model(
        train_surr,  # We pass the training dataset of the black box to the surrogate object
        val_surr,  # We pass the validation dataset of the black box to the surrogate object
        test_surr,  # We pass the test dataset of the black box to the surrogate object
        original_model,  # black box we want to explain
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        loss_fn=KLDivLoss(),
        validation_samples=args.validation_samples,  # this number is multiplied with the length of the validation dataset
        validation_batch_size=args.validation_batch_size,  # size of the mini batch
        verbose=True,
        lr=args.lr,
        optimizer=optimizer,
        train_loader=train_loader,
        random_sampler=random_sampler,
        batch_sampler=batch_sampler,
        bar=True,
        wandb=wandb,
        training_seed=args.seed,
        validation_seed=args.validation_seed,
    )

else:
    surrogate.train_original_model(
        X_train,  # We pass the training dataset of the black box to the surrogate object
        X_val,  # We pass the validation dataset of the black box to the surrogate object
        X_test,  # We pass the test dataset of the black box to the surrogate object
        original_model,  # black box we want to explain
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        loss_fn=KLDivLoss(),
        validation_samples=args.validation_samples,  # this number is multiplied with the length of the validation dataset
        validation_batch_size=args.validation_batch_size,  # size of the mini batch
        verbose=True,
        lr=args.lr,
        optimizer=optimizer,
        train_loader=train_loader,
        random_sampler=random_sampler,
        batch_sampler=batch_sampler,
        bar=True,
        wandb=wandb,
        training_seed=args.seed,
        validation_seed=args.validation_seed,
    )

if args.save_model:
    surr.cpu()
    torch.save(surr, f"{args.surrogate_name}.pt")
