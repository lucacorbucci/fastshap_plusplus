import argparse
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from opacus import PrivacyEngine
from torch.utils.data import (
    DataLoader,
)
from tqdm.auto import tqdm

from bb_architecture import SimpleCNN, TabularModel
from utils import get_optimizer, is_dataset_supported, is_image_dataset, prepare_data

warnings.simplefilter("ignore")


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


def get_model(args):
    if args.dataset_name == "adult" or args.dataset_name == "dutch":
        model = TabularModel(feature_size, 128, 2)
    elif args.dataset_name == "mnist":
        model = SimpleCNN()
    return model


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

        train_acc = eval_model(model, train_loader, device)
        wandb_run.log({"train_accuracy": train_acc, "epoch": epoch})

        if val_loader is not None:
            val_acc = eval_model(model, val_loader, device)
            wandb_run.log({"validation_accuracy": val_acc, "epoch": epoch})

        if test_loader is not None:
            test_acc = eval_model(model, test_loader, device)
            wandb_run.log({"test_accuracy": test_acc, "epoch": epoch})

    return model


if __name__ == "__main__":
    args = parser.parse_args()

    if not is_dataset_supported(args.dataset_name):
        raise ValueError(f"Dataset {args.dataset_name} is not supported")

    # Don't remove the seed setting
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.validation_seed is None:
        validation_seed = int(str(time.time()).split(".")[1]) * args.seed
        args.validation_seed = validation_seed

    image_dataset = is_image_dataset(args.dataset_name)

    if not image_dataset:
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
    elif image_dataset:
        (
            train_set,
            val_set,
            test_set,
        ) = prepare_data(args)

    # Don't remove the seed setting
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

    model = get_model(args)

    optimizer = get_optimizer(args.optimizer, model, args.lr)

    privacy_engine = PrivacyEngine()

    # If we do not want to train with privacy we set the noise multiplier to 0 and
    # the clipping to a very high value. Otherwise w train with DP.
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

    model = train_model(
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
        torch.save(
            model, f"../../artifacts/{args.dataset_name}/bb/{args.model_name}.pth"
        )
