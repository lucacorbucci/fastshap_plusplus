import os.path
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
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

from bb_architecture import Model, SimpleCNN, TabularModel
from fastshap.image_surrogate_DP import ImageSurrogate_DP
from fastshap.utils import (
    DatasetInputOnly,
    KLDivLoss,
    MaskLayer1d,
    MaskLayer2d,
)
from surrogate_architecture import get_image_surrogate, get_tabular_surrogate
from utils import (
    is_dataset_supported,
    is_image_dataset,
    prepare_data,
    setup_data_images,
)

warnings.simplefilter("ignore")
import argparse

import wandb

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


def get_surrogate_model(args):
    # Create surrogate model
    if args.dataset_name == "adult" or args.dataset_name == "dutch":
        return get_tabular_surrogate(num_features)
    else:
        return get_image_surrogate(args)


def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not found")


if __name__ == "__main__":
    args = parser.parse_args()

    if not is_dataset_supported(args.dataset_name):
        raise ValueError("Dataset not supported")
    image_dataset = is_image_dataset(args.dataset_name)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.validation_seed is None:
        validation_seed = int(str(time.time()).split(".")[1]) * args.seed
        args.validation_seed = validation_seed

    if not image_dataset:
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
    else:
        (
            train_set,
            val_set,
            test_set,
        ) = prepare_data(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = load_model(args.model_name)
    if image_dataset:
        train_loader, train_surr, val_surr, test_surr = setup_data_images()
    elif not image_dataset:
        train_loader, random_sampler, batch_sampler = setup_data(
            train_data=X_train, batch_size=args.batch_size
        )

    # create the model corresponding to this dataset to be used as a surrogate
    # and check if it has an architecture that is compatible with Opacus.
    surrogate_model = get_surrogate_model(args)
    surrogate_model = ModuleValidator.fix(surrogate_model)
    ModuleValidator.validate(surrogate_model, strict=False)

    # This is the privacy engine needed for training the surrogate model
    # using DP. Even if we do not want to use DP we still want to use the
    # PrivacyEngine and the "private" model without noise because this makes
    # easier to switch between private and non-private training.
    privacy_engine = PrivacyEngine()
    optimizer = get_optimizer(args.optimizer, surrogate_model, args.lr)

    if args.epsilon:
        surrogate_model, optimizer, train_loader = (
            privacy_engine.make_private_with_epsilon(
                module=surrogate_model,
                optimizer=optimizer,
                data_loader=train_loader,
                max_grad_norm=args.clipping,
                target_epsilon=args.epsilon,
                target_delta=1e-5,
                epochs=args.epochs,
            )
        )
    else:
        # If we do not want to use DP we set the noise multiplier to 0
        # the clipping to a very high value and we do not use poisson sampling.
        surrogate_model, optimizer, train_loader = privacy_engine.make_private(
            module=surrogate_model,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=10000000000,
            noise_multiplier=0,
            poisson_sampling=False,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate_model = surrogate_model.to(device)
    model = model.to(device)

    # Set up surrogate object: we pass the model that we have defined as
    # surrogate and the number of input features of the training dataset
    # we used to train the black box.
    if args.dataset_name == "mnist":
        surrogate = ImageSurrogate_DP(
            surrogate_model, width=28, height=28, superpixel_size=2
        )
        original_model = nn.Sequential(model, nn.Softmax(dim=1))
    else:
        surrogate = SurrogateDP(surrogate_model, num_features)
        original_model = nn.Sequential(model, nn.Softmax(dim=1))

    wandb_run = setup_wandb(args)

    if args.dataset_name == "mnist":
        surrogate.train_original_model(
            train_surr,
            val_surr,
            test_surr,
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
            X_train,
            X_val,
            X_test,
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
        surrogate_model.cpu()
        torch.save(surrogate_model, f"{args.surrogate_name}.pt")
