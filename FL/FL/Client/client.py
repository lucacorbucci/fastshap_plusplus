import gc
import logging
import os
import warnings
from collections import OrderedDict
from logging import INFO

import dill
import flwr as fl
import numpy as np
import ray
import torch
from flwr.common.logger import log
from Learning.learning import Learning
from opacus import PrivacyEngine
from Utils.utils import Utils


class FlowerClient(fl.client.NumPyClient):
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
        self.lr = preferences.learning_rate
        self.net = Utils.get_model(preferences.dataset, device=self.preferences.device)
        self.optimizer = Utils.get_optimizer(model=self.net, preferences=preferences)
        self.clipping = preferences.clipping
        self.fed_dir = preferences.fed_dir

    def get_parameters(self, config):
        return Utils.get_params(self.net)

    def fit(self, parameters, config):
        Utils.set_params(self.net, parameters)

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
        train_loader = Utils.get_dataloader(
            self.fed_dir,
            self.cid,
            batch_size=config["batch_size"],
            workers=num_workers,
            dataset=self.preferences.dataset,
            partition="train",
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
        private_net.to(self.preferences.device)

        gc.collect()

        all_metrics = []
        all_losses = []
        for epoch in range(0, self.preferences.epochs):
            metrics = Learning.train(
                model=private_net,
                train_loader=train_loader,
                optimizer=private_optimizer,
                device=self.preferences.device,
            )

            all_metrics.append(metrics)
            all_losses.append(metrics["Train Loss"])

        Utils.set_params(self.net, Utils.get_params(private_net))

        final_metrics = Learning.test(
            model=private_net,
            test_loader=train_loader,
            device=self.preferences.device,
        )

        if self.preferences.save_local_models:
            log(INFO, f"Saving local model of client {self.cid}")

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(
                private_net._module.state_dict().keys(), Utils.get_params(private_net)
            )
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model = Utils.get_model(
                self.preferences.dataset, device=self.preferences.device
            )
            model.load_state_dict(state_dict, strict=True)

            if not os.path.exists(f"{self.preferences.dataset_path}/models/{self.cid}"):
                os.makedirs(
                    f"{self.preferences.dataset_path}/models/{self.cid}", exist_ok=True
                )
            # Save the model
            torch.save(
                model,
                f"{self.preferences.dataset_path}/models/{self.cid}/model_node_{self.cid}_round_{current_fl_round}.pth",
            )

            torch.save(
                model.state_dict(),
                f"{self.preferences.dataset_path}/models/{self.cid}/model_node_{self.cid}_round_{current_fl_round}_state_dict.pth",
            )

        with open(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl", "wb") as f:
            dill.dump(privacy_engine.accountant, f)

        del private_net
        gc.collect()

        # Return local model and statistics
        return (
            Utils.get_params(self.net),
            len(train_loader.dataset),
            {
                # "train_losses": all_losses,
                "train_loss": float(final_metrics["Loss"]),
                "train_f1": float(final_metrics["F1 Score"]),
                "train_accuracy": float(final_metrics["Accuracy"]),
                "cid": self.cid,
            },
        )

    def evaluate(self, parameters, config):
        Utils.set_params(self.net, parameters)
        print("Evaluating... on client", self.cid)
        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])

        if self.preferences.cross_device:
            partition = "train"
        else:
            partition = config["phase"]

        dataset = Utils.get_dataloader(
            self.fed_dir,
            self.cid,
            batch_size=self.preferences.batch_size,
            workers=num_workers,
            dataset=self.preferences.dataset,
            partition=partition,
        )

        # Send model to device
        self.net.to(self.preferences.device)

        # Evaluate
        metrics = Learning.test(
            model=self.net,
            test_loader=dataset,
            device=self.preferences.device,
        )

        metrics = {
            "accuracy": float(metrics["Accuracy"]),
            "loss": float(metrics["Loss"]),
            "cid": self.cid,
            "f1_score": metrics["F1 Score"],
        }

        # Return statistics
        return (
            metrics["loss"],
            len(dataset.dataset),
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
            max_grad_norm=self.clipping,
        )

        return private_optimizer.noise_multiplier
