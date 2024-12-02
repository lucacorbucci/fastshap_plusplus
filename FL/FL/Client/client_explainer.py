import gc
import logging
import os
import warnings

import dill
import flwr as fl
import numpy as np
import ray
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import (
    DataLoader,
    Dataset,
    TensorDataset,
)
from Utils.utils import Utils

from fastshap.fastshap_dp import (
    FastSHAP,
    calculate_grand_coalition_FL,
    generate_validation_data_FL,
    validate_FL,
)
from fastshap.surrogate_dp import SurrogateDP
from fastshap.utils import (
    DatasetRepeat,
    ShapleySampler,
)


def load_model(model_name):
    if os.path.isfile(model_name):
        model = torch.load(model_name)
    else:
        raise FileNotFoundError("Model not found")
    return model


def prepare_dataset_for_explainer_FL(
    train_data,
    batch_size,
    imputer,
    num_samples,
    link,
    device,
    num_players,
    image_dataset=False,
    validation=False,
    validation_seed=None,
    validation_samples=None,
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

    num_workers = 0

    # Grand coalition value.
    grand_train = calculate_grand_coalition_FL(
        train_set,
        imputer,
        batch_size * num_samples,
        link,
        device,
        num_workers,
        num_players,
        image_dataset,
    ).cpu()

    # Null coalition.
    with torch.no_grad():
        zeros = torch.zeros(1, num_players, dtype=torch.float32, device=device)

        input_data = torch.tensor(train_set[0][0]).unsqueeze(0).to(device)

        # null = link(imputer(train_set[0][0].unsqueeze(0).to(device), zeros))
        null = link(imputer(input_data, zeros))
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

    if validation:
        # Generate validation data.
        sampler = ShapleySampler(num_players)
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        train_S, train_values = generate_validation_data_FL(
            train_set,
            imputer,
            validation_samples,
            sampler,
            batch_size * num_samples,
            link,
            device,
            num_workers,
            num_players,
            image_dataset,
        )

        # Set up val loader.
        train_set = DatasetRepeat(
            [train_set, TensorDataset(grand_train, train_S, train_values)]
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size * num_samples,
            pin_memory=True,
            num_workers=num_workers,
        )
        return train_loader, grand_train, null, sampler

    print("Dataset ready for the explainer")
    return train_loader, grand_train, null, sampler


class FlowerExplainerClient(fl.client.NumPyClient):
    def __init__(
        self,
        preferences,
        cid: str,
        client_generator,
    ):
        logging.info(f"Node {cid} is initializing...")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.client_generator = client_generator
        self.cid = cid
        self.preferences = preferences
        self.fed_dir = preferences.fed_dir
        self.lr = preferences.learning_rate

    def get_parameters(self, config):
        return Utils.get_params(self.net)

    def fit(self, parameters, config):
        current_fl_round = config["server_round"]
        random_generator = np.random.default_rng(
            seed=[int(self.client_generator.random() * 2**32), current_fl_round]
        )
        seed = int(random_generator.random() * 2**32)
        Utils.seed_everything(seed)

        with open(f"{self.fed_dir}/counter_sampling.pkl", "rb") as f:
            counter_sampling = dill.load(f)
            self.sampling_frequency = counter_sampling[str(self.cid)]

        num_features = 12  # bb_model.num_features

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        surrogate_model = load_model(self.preferences.surrogate_name)
        surrogate_model = surrogate_model.to(self.preferences.device)

        surrogate = SurrogateDP(surrogate_model, num_features)

        dataset = Utils.get_data(
            self.fed_dir,
            self.cid,
            batch_size=config["batch_size"],
            workers=num_workers,
            dataset=self.preferences.dataset,
            partition="train",
        )

        original_train_loader, grand_train, null, sampler = (
            prepare_dataset_for_explainer_FL(
                train_data=dataset,
                batch_size=self.preferences.batch_size,
                imputer=surrogate,
                num_samples=len(dataset),
                link=nn.Softmax(dim=-1),
                device=self.preferences.device,
                num_players=num_features,
            )
        )

        self.delta = 1 / len(original_train_loader.dataset)

        loaded_privacy_engine = None

        # If we already used this client we need to load the state regarding
        # the privacy engine both for the classic model and for the model
        # used for the regularization
        if os.path.exists(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl"):
            with open(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl", "rb") as file:
                loaded_privacy_engine = dill.load(file)

        if self.preferences.epsilon is None:
            self.noise_multiplier = 0
            self.original_epsilon = None
        else:
            if os.path.exists(f"{self.fed_dir}/noise_level_{self.cid}.pkl"):
                with open(f"{self.fed_dir}/noise_level_{self.cid}.pkl", "rb") as file:
                    self.noise_multiplier = dill.load(file)
                    self.original_epsilon = self.preferences.epsilon
                    self.preferences.epsilon = None
            else:
                noise = self.get_noise(dataset=original_train_loader)
                with open(f"{self.fed_dir}/noise_level_{self.cid}.pkl", "wb") as file:
                    dill.dump(noise, file)
                self.noise_multiplier = noise
                self.original_epsilon = self.preferences.epsilon
                self.preferences.epsilon = None

        self.net = Utils.get_explainer_model(
            dataset_name=self.preferences.dataset, num_features=num_features
        )

        Utils.set_params(self.net, parameters)

        self.optimizer = Utils.get_optimizer(
            model=self.net, preferences=self.preferences
        )

        (
            private_explainer,
            private_optimizer,
            train_loader,
            privacy_engine,
        ) = Utils.create_private_model(
            model=self.net,
            preferences=self.preferences,
            original_optimizer=self.optimizer,
            train_loader=original_train_loader,
            delta=self.delta,
            noise_multiplier=self.noise_multiplier,
            accountant=loaded_privacy_engine,
        )

        private_explainer = private_explainer.to(self.preferences.device)
        self.net = self.net.to(self.preferences.device)

        fastshap = FastSHAP(
            private_explainer,
            surrogate,
            num_features=num_features,
            normalization="none",
            link=nn.Softmax(dim=-1),
        )

        # gc.collect()

        train_loss = fastshap.train_FL(
            train_loader,
            grand_train,
            null,
            batch_size=self.preferences.batch_size,
            num_samples=len(dataset),
            lr=self.preferences.learning_rate,
            max_epochs=self.preferences.epochs,
            verbose=True,
            optimizer=private_optimizer,
            sampler=sampler,
            bar=True,
            eff_lambda=self.preferences.eff_lambda,
            paired_sampling=self.preferences.paired_sampling,
            device=self.preferences.device,
        )

        with open(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl", "wb") as f:
            dill.dump(privacy_engine.accountant, f)

        # del private_explainer
        # gc.collect()

        # Return local model and statistics
        return (
            Utils.get_params(fastshap.explainer),
            len(train_loader.dataset),
            {
                # "train_losses": all_losses,
                "train_loss": train_loss,
                "cid": self.cid,
                "explainer": True,
            },
        )

    def evaluate(self, parameters, config):
        surrogate_model = load_model(self.preferences.surrogate_name)
        surrogate_model = surrogate_model.to(self.preferences.device)
        num_features = 12  # bb_model.num_features

        surrogate = SurrogateDP(surrogate_model, num_features)

        self.net = Utils.get_explainer_model(
            dataset_name=self.preferences.dataset, num_features=num_features
        )

        Utils.set_params(self.net, parameters)
        print("Evaluating... on client", self.cid)
        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        partition = config["phase"]

        dataset = Utils.get_data(
            self.fed_dir,
            self.cid,
            batch_size=config["batch_size"],
            workers=num_workers,
            dataset=self.preferences.dataset,
            partition="train",
        )

        train_loader, grand_train, null, sampler = prepare_dataset_for_explainer_FL(
            train_data=dataset,
            batch_size=self.preferences.batch_size,
            imputer=surrogate,
            num_samples=len(dataset),
            link=nn.Softmax(dim=-1),
            device=self.preferences.device,
            num_players=num_features,
            validation=True,
            validation_seed=self.preferences.seed,
            validation_samples=self.preferences.validation_samples,
        )

        print(
            "Evaluating... on client",
            self.cid,
            "Batch size",
            config["batch_size"],
            self.preferences.batch_size,
        )

        # # Send model to device
        self.net = self.net.to(self.preferences.device)
        # surrogate = SurrogateDP(self.net, num_features)

        fastshap_model = FastSHAP(
            explainer=self.net,
            imputer=surrogate,
            num_features=num_features,
            normalization=None,
            link=nn.Softmax(dim=-1),
        )

        loss = validate_FL(
            val_loader=train_loader,
            imputer=surrogate,
            explainer=fastshap_model.explainer,
            null=null,
            link=nn.Softmax(dim=-1),
            normalization=None,
            num_players=num_features,
            device=self.preferences.device,
            FL_evaluation=True,
            num_samples=len(dataset),
            eff_lambda=self.preferences.eff_lambda,
        )

        if partition == "validation":
            metrics = {
                "validation_loss": loss,
                "cid": self.cid,
                "explainer": True,
            }
        elif partition == "test":
            metrics = {
                "test_loss": loss,
                "cid": self.cid,
                "explainer": True,
            }
        else:
            raise ValueError("Partition not found")

        # Return statistics

        return (
            loss,
            len(train_loader.dataset),
            metrics,
        )

    def get_noise(self, dataset, target_epsilon=None):
        model_noise = Utils.get_model(
            self.preferences.dataset, device=self.preferences.device
        )
        privacy_engine = PrivacyEngine(accountant="rdp")
        optimizer_noise = Utils.get_optimizer(model_noise, self.preferences)
        (
            _,
            private_optimizer,
            _,
        ) = privacy_engine.make_private_with_epsilon(
            module=model_noise,
            optimizer=optimizer_noise,
            data_loader=dataset,
            epochs=self.sampling_frequency * self.preferences.epochs,
            target_epsilon=self.preferences.epsilon
            if target_epsilon is None
            else target_epsilon,
            target_delta=self.delta,
            max_grad_norm=self.preferences.clipping,
        )

        return private_optimizer.noise_multiplier
