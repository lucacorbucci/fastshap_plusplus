import argparse
import os.path
import random
import time
import warnings

# import shapreg  # https://github.com/iancovert/shapley-regression
import numpy as np
import torch
import torch.nn as nn
import wandb

# import Functional torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from bb_architecture import SimpleCNN, TabularModel
from explainer_architecture import (
    SimpleConvLinearNetMNIST,
    get_tabular_explainer,
)
from fastshap.fastshap_dp import (
    FastSHAP,
)
from fastshap.image_surrogate_DP import ImageSurrogate_DP
from fastshap.surrogate_dp import SurrogateDP
from fastshap.utils import DatasetInputOnly
from utils import (
    get_optimizer,
    is_dataset_supported,
    is_image_dataset,
    prepare_data,
    prepare_dataset_for_explainer,
)

warnings.simplefilter("ignore")

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
parser.add_argument("--normalization", type=str, default="none")
parser.add_argument("--num_samples", type=int, default=32)
parser.add_argument("--surrogate", type=str, default="")
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--dataset_name", type=str, default="adult")

parser.add_argument("--paired_sampling", type=bool, default=True)
parser.add_argument("--eff_lambda", type=float, default=0)

parser.add_argument("--validation_seed", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)


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


if __name__ == "__main__":
    args = parser.parse_args()

    if not is_dataset_supported(args.dataset_name):
        raise ValueError("Dataset not supported")
    image_dataset = is_image_dataset(args.dataset_name)

    wandb_run = setup_wandb(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    surrogate = load_model(f"{args.surrogate}")

    imputer = surrogate.to(device)

    imputer = surrogate

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

    if image_dataset:
        surrogate = ImageSurrogate_DP(surrogate, width=28, height=28, superpixel_size=2)
        fastshap_train = DatasetInputOnly(train_set)
        if val_set is not None:
            fastshap_val = DatasetInputOnly(val_set)
            fastshap_test = None
        else:
            fastshap_val = None
            fastshap_test = DatasetInputOnly(test_set)
    else:
        surrogate = SurrogateDP(surrogate, num_features)

    if args.sweep:
        train_loader, val_loader, grand_train, grand_val, null, sampler = (
            prepare_dataset_for_explainer(
                args,
                X_train if not image_dataset else fastshap_train,
                X_val if not image_dataset else fastshap_val,
                args.batch_size,
                surrogate,
                num_samples=args.num_samples,
                link=nn.Softmax(dim=-1),
                device=device,
                validation_samples=args.validation_samples,
                num_players=num_features
                if not image_dataset
                else surrogate.num_players,
                validation_seed=args.validation_seed,
                image_dataset=image_dataset,
            )
        )
    else:
        train_loader, test_loader, grand_train, grand_test, null, sampler = (
            prepare_dataset_for_explainer(
                args,
                X_train if not image_dataset else fastshap_train,
                X_test if not image_dataset else fastshap_test,
                args.batch_size,
                surrogate,
                num_samples=args.num_samples,
                link=nn.Softmax(dim=-1),
                device=device,
                validation_samples=args.validation_samples,
                num_players=num_features
                if not image_dataset
                else surrogate.num_players,
                validation_seed=args.validation_seed,
                image_dataset=image_dataset,
            )
        )
        val_loader = None
        grand_val = None

    if not image_dataset:
        explainer = get_tabular_explainer(num_features).to(device)
    else:
        # explainer = SimpleCNN(n_classes=10, in_channels=1, base_channels=16).to(device)
        # explainer = UNet(
        #     n_classes=10,
        #     num_down=2,
        #     num_up=1,
        #     # num_convs=2,
        #     in_channels=1,
        #     base_channels=16,
        #     bilinear=False,
        # ).to(device)
        explainer = SimpleConvLinearNetMNIST().to(device)
        # print model details summary for the model
        print(explainer)

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
        num_features=num_features if not image_dataset else surrogate.num_players,
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
        eff_lambda=args.eff_lambda,
        paired_sampling=args.paired_sampling,
        image_dataset=image_dataset,
    )

    if args.save_model:
        torch.save(
            fastshap.explainer,
            f"../../artifacts/{args.dataset_name}/explainer/{args.model_name}.pt",
        )
