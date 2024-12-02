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
from Utils.utils import Utils

from fastshap.surrogate_dp import SurrogateDP, validate_FL
from fastshap.utils import KLDivLoss, setup_data, setup_data_FL


def load_model(model_name):
    if os.path.isfile(model_name):
        model = torch.load(model_name)
    else:
        raise FileNotFoundError("Model not found")
    return model


class FlowerSurrogateClient(fl.client.NumPyClient):
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
        self.clipping = preferences.clipping
        self.fed_dir = preferences.fed_dir

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

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        bb_model = load_model(self.preferences.bb_name)
        dataset = Utils.get_data(
            self.fed_dir,
            self.cid,
            batch_size=config["batch_size"],
            workers=num_workers,
            dataset=self.preferences.dataset,
            partition="train",
        )

        train_loader, random_sampler, batch_sampler = setup_data(
            train_data=dataset, batch_size=self.preferences.batch_size
        )

        self.delta = 1 / len(train_loader.dataset)

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
                noise = self.get_noise(dataset=train_loader)
                with open(f"{self.fed_dir}/noise_level_{self.cid}.pkl", "wb") as file:
                    dill.dump(noise, file)
                self.noise_multiplier = noise
                self.original_epsilon = self.preferences.epsilon
                self.preferences.epsilon = None

        num_features = 12  # bb_model.num_features
        self.num_features = num_features
        self.net = Utils.get_surrogate_model(
            self.preferences.dataset, num_features=num_features
        )
        Utils.set_params(self.net, parameters)

        self.optimizer = Utils.get_optimizer(
            model=self.net, preferences=self.preferences
        )

        bb_model = nn.Sequential(bb_model, nn.Softmax(dim=1))

        (
            private_net,
            private_optimizer,
            train_loader,
            privacy_engine,
        ) = Utils.create_private_model(
            model=self.net,
            preferences=self.preferences,
            original_optimizer=self.optimizer,
            train_loader=train_loader,
            delta=self.delta,
            noise_multiplier=self.noise_multiplier,
            accountant=loaded_privacy_engine,
        )
        # private_net.to(self.preferences.device)

        surrogate = SurrogateDP(private_net, num_features)
        private_net = private_net.to(self.preferences.device)
        self.net = self.net.to(self.preferences.device)

        bb_model = bb_model.to(self.preferences.device)

        gc.collect()

        surrogate.train_original_model_FL(
            original_model=bb_model,  # black box we want to explain
            batch_size=self.preferences.batch_size,
            max_epochs=self.preferences.epochs,
            loss_fn=KLDivLoss(),
            train_loader=train_loader,
            random_sampler=random_sampler,
            batch_sampler=batch_sampler,
            lr=self.preferences.learning_rate,
            training_seed=self.preferences.seed,
            bar=True,
            verbose=True,
            optimizer=private_optimizer,
            # validation_seed=args.validation_seed,
        )

        Utils.set_params(self.net, Utils.get_params(surrogate.surrogate))

        train_loss, train_fidelity = validate_FL(
            surrogate=surrogate,
            loss_fn=KLDivLoss(),
            data_loader=train_loader.dataset,
            num_players=num_features,
            preferences=self.preferences,
            original_model=bb_model,
        )

        with open(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl", "wb") as f:
            dill.dump(privacy_engine.accountant, f)

        del private_net
        gc.collect()

        # Return local model and statistics
        return (
            Utils.get_params(surrogate.surrogate),
            len(train_loader.dataset),
            {
                # "train_losses": all_losses,
                "train_loss": train_loss,
                "train_fidelity": train_fidelity,
                "cid": self.cid,
                "surrogate": True,
            },
        )

    def evaluate(self, parameters, config):
        bb_model = load_model(self.preferences.bb_name)
        bb_model = nn.Sequential(bb_model, nn.Softmax(dim=1))
        bb_model = bb_model.to(self.preferences.device)

        num_features = 12  # bb_model.num_features
        self.net = Utils.get_surrogate_model(
            self.preferences.dataset, num_features=num_features
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

        train_loader, random_sampler, batch_sampler = setup_data_FL(
            train_data=dataset, batch_size=self.preferences.batch_size
        )

        print(
            "Evaluating... on client",
            self.cid,
            "Batch size",
            config["batch_size"],
            self.preferences.batch_size,
        )

        # Send model to device
        self.net.to(self.preferences.device)
        surrogate = SurrogateDP(self.net, num_features)

        loss, fidelity = validate_FL(
            surrogate=surrogate,
            loss_fn=KLDivLoss(),
            data_loader=train_loader.dataset,
            num_players=num_features,
            preferences=self.preferences,
            original_model=bb_model,
        )

        if partition == "validation":
            metrics = {
                "validation_loss": loss,
                "validation_fidelity": fidelity,
                "cid": self.cid,
                "surrogate": True,
            }
        elif partition == "test":
            metrics = {
                "test_loss": loss,
                "test_fidelity": fidelity,
                "cid": self.cid,
                "surrogate": True,
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
        model_noise = Utils.get_surrogate_model(
            self.preferences.dataset, num_features=self.num_features
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
            max_grad_norm=self.clipping,
        )

        return private_optimizer.noise_multiplier
