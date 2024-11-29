import os
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from Models.celeba import CelebaNet
from Models.logistic_regression_net import (
    LinearClassificationNet,
    LinearClassificationNetValerio,
)
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from Utils.preferences import Preferences

from fastshap.utils import (
    MaskLayer1d,
)


class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data=None,
        data=None,
        targets=None,
        transform: Optional[Callable] = None,
    ) -> None:
        path = path_to_data.parent if path_to_data else None
        self.dataset_path = path.parent.parent.parent if path_to_data else None

        super(TorchVision_FL, self).__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.sensitive_features, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets


class Utils:
    @staticmethod
    def seed_everything(seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def split_train(X, y, z, size):
        combined_data = list(zip(X, y, z))
        random.shuffle(combined_data)
        X, y, z = zip(*combined_data)
        X_split = X[: int(size * len(X))]
        y_split = y[: int(size * len(y))]
        z_split = z[: int(size * len(z))]
        X = X[int(size * len(X)) :]
        y = y[int(size * len(y)) :]
        z = z[int(size * len(z)) :]

        return X, y, z, X_split, y_split, z_split

    @staticmethod
    def get_tabular_surrogate(num_features):
        return nn.Sequential(
            MaskLayer1d(value=0, append=True),
            nn.Linear(2 * num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    @staticmethod
    def get_surrogate_model(dataset_name, num_features):
        # Create surrogate model
        if dataset_name == "dutch":
            return Utils.get_tabular_surrogate(num_features)

    @staticmethod
    def get_explainer_model(dataset_name, num_features):
        # Create surrogate model
        if dataset_name == "dutch":
            return Utils.get_tabular_explainer(num_features)

    @staticmethod
    def get_tabular_explainer(num_features):
        return nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * num_features),
        )

    @staticmethod
    def get_model(
        dataset: str,
        device: torch.device,
        input_size: int = None,
        output_size: int = None,
    ) -> torch.nn.Module:
        """This function returns the model to train.

        Args:
            dataset (str): the name of the dataset
            device (torch.device): the device where the model will be trained

        Raises:
            ValueError: if the dataset is not supported

        Returns:
            torch.nn.Module: the model to train
        """
        if dataset == "celeba":
            return CelebaNet()
        elif dataset == "dutch":
            return LinearClassificationNetValerio(input_size=12, output_size=2)
        elif dataset == "income":
            return LinearClassificationNet(input_size=54, output_size=2)
        elif dataset == "adult":
            return LinearClassificationNet(input_size=103, output_size=2)
        else:
            raise ValueError(f"Dataset {dataset} not supported")

    @staticmethod
    def get_optimizer(model, preferences):
        if preferences.optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=preferences.learning_rate,
            )
        elif preferences.optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=preferences.learning_rate,
            )
        elif preferences.optimizer == "adamW":
            return torch.optim.AdamW(
                model.parameters(),
                lr=preferences.learning_rate,
            )
        else:
            raise ValueError("Optimizer not recognized")

    @staticmethod
    def create_private_model(
        model: torch.nn.Module,
        preferences: Preferences,
        original_optimizer,
        train_loader,
        delta: float,
        noise_multiplier: float = 0,
        accountant=None,
    ) -> Tuple[GradSampleModule, DPOptimizer, DataLoader]:
        """

        Args:
            model (torch.nn.Module): the model to wrap
            epsilon (float): the target epsilon for the privacy budget
            original_optimizer (_type_): the optimizer of the model before
                wrapping it with Privacy Engine
            train_loader (_type_): the train dataloader used to train the model
            epochs (_type_): for how many epochs the model will be trained
            delta (float): the delta for the privacy budget
            MAX_GRAD_NORM (float): the clipping value for the gradients
            batch_size (int): batch size

        Returns:
            Tuple[GradSampleModule, DPOptimizer, DataLoader]: the wrapped model,
                the wrapped optimizer and the train dataloader
        """
        privacy_engine = PrivacyEngine(accountant="rdp")
        if accountant:
            privacy_engine.accountant = accountant

        # We can wrap the model with Privacy Engine using the
        # method .make_private(). This doesn't require you to
        # specify a epsilon. In this case we need to specify a
        # noise multiplier.
        # make_private_with_epsilon() instead requires you to
        # provide a target epsilon and a target delta. In this
        # case you don't need to specify a noise multiplier.
        private_model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=original_optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=preferences.clipping,
        )
        print(f"Created private model with noise {optimizer.noise_multiplier}")

        return private_model, optimizer, train_loader, privacy_engine

    @staticmethod
    def setup_wandb(preferences):
        wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            project=preferences.project_name,
            name="experiment" if not preferences.run_name else preferences.run_name,
            # track hyperparameters and run metadata
            config={
                "learning_rate": preferences.learning_rate,
                "batch_size": preferences.batch_size,
                "dataset": preferences.dataset,
                "num_rounds": preferences.fl_rounds,
                "num_nodes": preferences.num_nodes,
                "epochs": preferences.epochs,
                "epsilon": preferences.epsilon if preferences.epsilon else None,
                "gradnorm": preferences.clipping,
                "ratio_unfair_nodes": preferences.ratio_unfair_nodes,
                "node_shuffle_seed": preferences.node_shuffle_seed,
                "ratio_unfairness": preferences.ratio_unfairness,
            },
        )

        return wandb_run

    @staticmethod
    def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def get_data(
        path_to_data: str,
        cid: str,
        batch_size: int,
        workers: int,
        dataset: str,
        partition: str = "train",
    ):
        """Generates trainset/valset object and returns appropiate dataloader."""

        partition = (
            "train"
            if partition == "train"
            else "test"
            if partition == "test"
            else "validation"
        )
        return Utils.get_dataset(Path(path_to_data), cid, partition, dataset)

    @staticmethod
    def get_dataloader(
        path_to_data: str,
        cid: str,
        # is_train: bool,
        batch_size: int,
        workers: int,
        dataset: str,
        partition: str = "train",
    ):
        """Generates trainset/valset object and returns appropiate dataloader."""

        partition = (
            "train"
            if partition == "train"
            else "test"
            if partition == "test"
            else "validation"
        )
        dataset = Utils.get_dataset(Path(path_to_data), cid, partition, dataset)

        # we use as number of workers all the cpu cores assigned to this actor
        kwargs = {"num_workers": 0, "pin_memory": True, "drop_last": False}
        return DataLoader(dataset, batch_size=batch_size, **kwargs)

    @staticmethod
    def get_dataset(path_to_data: Path, cid: str, partition: str, dataset: str):
        # generate path to cid's data
        path_to_data = path_to_data / cid / (partition + ".pt")
        if dataset == "dutch":
            return torch.load(path_to_data)
        elif dataset == "adult":
            return torch.load(path_to_data)
        elif dataset == "dutch_unprivileged":
            return torch.load(path_to_data)
        elif dataset == "german":
            return torch.load(path_to_data)
        elif dataset == "compas":
            return torch.load(path_to_data)
        elif dataset == "income":
            return torch.load(path_to_data)
        else:
            return TorchVision_FL(
                path_to_data,
                transform=Utils.get_transformation(dataset),
            )

    @staticmethod
    def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
