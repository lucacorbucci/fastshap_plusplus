import argparse
import logging
import os
import signal
import sys
import time
from typing import Dict

import flwr as fl
import numpy as np
import torch
from Client.client import FlowerClient
from Client.client_explainer import FlowerExplainerClient
from Client.client_surrogate import FlowerSurrogateClient
from ClientManager.client_manager import SimpleClientManager
from Dataset.federated_dataset import FederatedDataset
from Dataset.load_data import LoadDataset
from Dataset.tabular_dataset import TabularDataset
from flwr.common.typing import Scalar
from Server.server import Server
from Strategy.fed_avg import FedAvg
from Utils.aggregation import Aggregation
from Utils.preferences import Preferences
from Utils.utils import Utils


def signal_handler(sig, frame):
    print("Gracefully stopping your experiment! Keep calm!")
    global wandb_run
    if wandb_run:
        wandb_run.finish()
    sys.exit(0)


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--fl_rounds", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--cross_device", type=bool, default=False)
parser.add_argument("--tabular", type=bool, default=False)
parser.add_argument("--num_nodes", type=int, default=None)
parser.add_argument("--split_approach", type=str, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--node_shuffle_seed", type=int, default=None)

parser.add_argument("--alpha_dirichlet", type=float, default=None)
parser.add_argument(
    "--ratio_unfair_nodes", type=float, default=None
)  # number of nodes to make unfair
parser.add_argument("--group_to_reduce", type=float, default=None, nargs="+")
parser.add_argument("--group_to_increment", type=float, default=None, nargs="+")
parser.add_argument(
    "--ratio_unfairness", type=float, default=None
)  # how much we want to unbalance the dataset on the unfair nodes
parser.add_argument("--validation_size", type=float, default=None)
parser.add_argument("--test_size", type=float, default=None)

# Parameters for privacy-preserving trainign
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--clipping", type=float, default=100000000)

# Percentage of nodes to use for training, validation and test
parser.add_argument("--fraction_fit_nodes", type=float, default=None)
parser.add_argument("--fraction_validation_nodes", type=float, default=None)
parser.add_argument("--fraction_test_nodes", type=float, default=None)

# Percentage of nodes to sample for training, validation and test
parser.add_argument("--sampled_training_nodes", type=float, default=1.0)
parser.add_argument("--sampled_validation_nodes", type=float, default=0)
parser.add_argument("--sampled_test_nodes", type=float, default=1.0)

# Parameters for the wandb logging
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--project_name", type=str, default=None)

# configuration for the clients. What percentage of CPU and GPU each client can use
parser.add_argument("--num_client_cpus", type=float, default=None)
parser.add_argument("--num_client_gpus", type=float, default=None)

parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--save_local_models", type=bool, default=False)
parser.add_argument("--save_aggregated_model", type=bool, default=False)
parser.add_argument("--aggregated_model_name", type=str, default="model")

parser.add_argument("--train_surrogate", type=bool, default=False)
parser.add_argument("--bb_name", type=str, default=None)

parser.add_argument("--train_explainer", type=bool, default=False)

parser.add_argument("--validation_samples", type=int, default=None)
parser.add_argument("--num_samples", type=int, default=None)
parser.add_argument("--validation_batch_size", type=int, default=None)

parser.add_argument("--eff_lambda", type=float, default=None)
parser.add_argument("--paired_sampling", type=bool, default=None)
parser.add_argument("--surrogate_name", type=str, default="cuda")
parser.add_argument("--splitted_data_dir", type=str, default="federated")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    # remove files in tmp/ray
    args = parser.parse_args()

    if args.node_shuffle_seed is None:
        node_shuffle_seed = int(str(time.time()).split(".")[1]) * args.seed
        args.node_shuffle_seed = node_shuffle_seed

    print(args.node_shuffle_seed)

    path_to_remove = os.listdir(f"{args.dataset_path}/{args.splitted_data_dir}/")
    for item in path_to_remove:
        if item.endswith(".pkl"):
            os.remove(
                os.path.join(f"{args.dataset_path}/{args.splitted_data_dir}/", item)
            )

    preferences = Preferences(
        dataset=args.dataset_name,
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        optimizer=args.optimizer,
        sweep=args.sweep,
        fl_rounds=args.fl_rounds,
        cross_device=args.cross_device,
        tabular=args.tabular,
        num_nodes=args.num_nodes,
        split_approach=args.split_approach,
        learning_rate=args.lr,
        alpha_dirichlet=args.alpha_dirichlet,
        ratio_unfair_nodes=args.ratio_unfair_nodes,
        group_to_reduce=tuple(args.group_to_reduce) if args.group_to_reduce else None,
        group_to_increment=tuple(args.group_to_increment)
        if args.group_to_increment
        else None,
        ratio_unfairness=tuple(args.ratio_unfairness)
        if args.ratio_unfairness
        else None,
        validation=True if args.validation_size is not None else False,
        validation_size=args.validation_size if args.validation_size else None,
        fed_dir=f"{args.dataset_path}/{args.splitted_data_dir}/",
        epsilon=args.epsilon,
        clipping=args.clipping,
        node_shuffle_seed=args.node_shuffle_seed,
        wandb=args.wandb,
        project_name=args.project_name,
        run_name=args.run_name,
        sampled_training_nodes=args.sampled_training_nodes,
        sampled_validation_nodes=args.sampled_validation_nodes,
        sampled_test_nodes=args.sampled_test_nodes,
        device=args.device,
        save_aggregated_model=args.save_aggregated_model,
        save_local_models=args.save_local_models,
        test_size=args.test_size,
        fraction_fit_nodes=args.fraction_fit_nodes,
        fraction_validation_nodes=args.fraction_validation_nodes,
        fraction_test_nodes=args.fraction_test_nodes,
        train_surrogate=args.train_surrogate,
        train_explainer=args.train_explainer,
        bb_name=args.bb_name,
        validation_samples=args.validation_samples,
        validation_batch_size=args.validation_batch_size,
        surrogate_name=args.surrogate_name,
        paired_sampling=args.paired_sampling,
        eff_lambda=args.eff_lambda,
        aggregated_model_name=args.aggregated_model_name,
        num_samples=args.num_samples,
    )

    Utils.seed_everything(args.seed)

    wandb_run = Utils.setup_wandb(preferences) if args.wandb else None

    if preferences.tabular:
        if preferences.cross_device:
            if preferences.dataset == "dutch":
                X, Z, Y = LoadDataset.load_dataset(preferences)
                if preferences.split_approach == "non_iid":
                    distribution = FederatedDataset.get_distribution(
                        preferences=preferences
                    )
                    FederatedDataset.create_federated_dataset(
                        preferences, X, Y, Z, "train", distribution
                    )
            else:
                for client_name in range(preferences.num_nodes):
                    os.system(
                        f"rm -rf {args.dataset_path}/{args.splitted_data_dir}/{client_name}/train.pt"
                    )

                    # open numpy arrays
                    X = np.load(
                        f"{args.dataset_path}/{args.splitted_data_dir}/{client_name}/{args.dataset_name}_dataframes_{client_name}.npy"
                    )
                    Y = np.load(
                        f"{args.dataset_path}/{args.splitted_data_dir}/{client_name}/{args.dataset_name}_labels_{client_name}.npy"
                    )
                    Z = np.load(
                        f"{args.dataset_path}/{args.splitted_data_dir}/{client_name}/{args.dataset_name}_groups_{client_name}.npy"
                    )
                    custom_dataset = TabularDataset(
                        x=np.hstack((X, np.ones((X.shape[0], 1)))).astype(np.float32),
                        z=[item.item() for item in Z],  # .astype(np.float32),
                        y=[item.item() for item in Y],  # .astype(np.float32),
                    )
                    torch.save(
                        custom_dataset,
                        f"{args.dataset_path}/{args.splitted_data_dir}/{client_name}/train.pt",
                    )
        else:
            X, Z, Y = LoadDataset.load_dataset(preferences)
            X, Y, Z, X_test, y_test, z_test = Utils.split_train(
                X,
                Y,
                Z,
                preferences.test_size,
            )
            distribution = None
            if preferences.split_approach == "non_iid":
                distribution = FederatedDataset.get_distribution(
                    preferences=preferences
                )
            if preferences.validation:
                print("Creating Validation Set")
                X, Y, Z, X_val, y_val, z_val = Utils.split_train(
                    X,
                    Y,
                    Z,
                    preferences.validation_size,
                )
                FederatedDataset.create_federated_dataset(
                    preferences, X_val, y_val, z_val, "validation", distribution
                )
            FederatedDataset.create_federated_dataset(
                preferences, X, Y, Z, "train", distribution
            )

            FederatedDataset.create_federated_dataset(
                preferences, X_test, y_test, z_test, "test", distribution
            )
    else:
        raise ValueError("Only tabular datasets are supported")

    if preferences.cross_device:
        preferences.num_training_nodes = int(args.num_nodes * args.fraction_fit_nodes)
        preferences.num_validation_nodes = int(
            args.num_nodes * args.fraction_validation_nodes
        )
        preferences.num_test_nodes = int(args.num_nodes * args.fraction_test_nodes)
    else:
        preferences.num_training_nodes = int(args.num_nodes)
        preferences.num_validation_nodes = int(args.num_nodes)
        preferences.num_test_nodes = int(args.num_nodes)

    num_features = Utils.get_num_features(preferences.dataset)
    if preferences.train_surrogate:
        model = Utils.get_surrogate_model(preferences.dataset, num_features)
    elif preferences.train_explainer:
        model = Utils.get_explainer_model(preferences.dataset, num_features)
    else:
        model = Utils.get_model(preferences.dataset, "cuda")

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(model_parameters)

    def fit_config(server_round: int = 0) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": args.epochs,  # number of local epochs
            "batch_size": args.batch_size,
            "dataset": preferences.dataset,
            "server_round": server_round,
        }
        return config

    def evaluate_config(server_round: int = 0) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": args.epochs,  # number of local epochs
            "batch_size": args.batch_size,
            "dataset": preferences.dataset,
        }
        return config

    def client_fn(cid: str):
        client_generator = np.random.default_rng(seed=[args.seed, cid])
        # create a single client instance
        if preferences.train_surrogate:
            return FlowerSurrogateClient(
                preferences=preferences,
                cid=cid,
                client_generator=client_generator,
            ).to_client()
        elif preferences.train_explainer:
            return FlowerExplainerClient(
                preferences=preferences,
                cid=cid,
                client_generator=client_generator,
            ).to_client()
        else:
            return FlowerClient(
                preferences=preferences,
                cid=cid,
                client_generator=client_generator,
            ).to_client()

    # these parameters are used to configure Ray and they are dependent on
    # the machine we want to use to run the experiments
    ray_num_cpus = 20
    ray_num_gpus = 1
    ram_memory = 16_000 * 1024 * 1024 * 2

    strategy = FedAvg(
        fraction_fit=args.sampled_training_nodes,
        fraction_evaluate=args.sampled_validation_nodes,
        fraction_test=args.sampled_test_nodes,
        min_fit_clients=args.sampled_training_nodes,
        min_evaluate_clients=0,
        min_available_clients=args.sampled_training_nodes,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=Aggregation.agg_metrics_train,
        evaluate_metrics_aggregation_fn=Aggregation.agg_metrics_evaluation,
        test_metrics_aggregation_fn=Aggregation.agg_metrics_test,
        preferences=preferences,
        wandb_run=wandb_run,
    )

    # (optional) specify Ray config
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": ray_num_cpus,
        "num_gpus": ray_num_gpus,
        "_memory": ram_memory,
        "_redis_max_memory": 10000000,
        "object_store_memory": 78643200,
        "logging_level": logging.ERROR,
        "log_to_driver": True,
    }
    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": args.num_client_gpus,
    }

    client_manager = SimpleClientManager(
        preferences=preferences,
    )
    server = Server(client_manager=client_manager, strategy=strategy)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_nodes,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.fl_rounds),
        ray_init_args=ray_init_args,
        server=server,
        client_manager=client_manager,
    )

    if wandb_run:
        wandb_run.finish()

# # # iid test
# # preferences = Preferences(
# #     dataset="adult",
# #     dataset_path="../data/adult/",
# #     epochs=10,
# #     device="cpu",
# #     batch_size=64,
# #     seed=42,
# #     optimizer="adam",
# #     sweep=False,
# #     fl_round=1,
# #     cross_device=False,
# #     tabular = True,
# #     num_nodes = 10,
# #     split_approach="iid",
# # )


# # if preferences.tabular:
# #     if preferences.cross_device:
# #         X, Z, y = LoadDataset.load_dataset(preferences)
# #     else:
# #         X, Z, y, X_test, Z_test, y_test = LoadDataset.load_dataset(preferences)
# #         FederatedDataset.create_federated_dataset(preferences, X, y, Z, "train")
# #         FederatedDataset.create_federated_dataset(preferences, X_test, y_test, Z_test, "test")

# # non iid test
# # preferences = Preferences(
# #     dataset="adult",
# #     dataset_path="../data/adult/",
# #     epochs=10,
# #     device="cpu",
# #     batch_size=64,
# #     seed=42,
# #     optimizer="adam",
# #     sweep=False,
# #     fl_round=1,
# #     cross_device=False,
# #     tabular = True,
# #     num_nodes = 10,
# #     split_approach="non_iid",
# #     alpha_dirichlet=1.0
# # )


# # if preferences.tabular:
# #     if preferences.cross_device:
# #         X, Z, y = LoadDataset.load_dataset(preferences)
# #     else:
# #         X, Z, y, X_test, Z_test, y_test = LoadDataset.load_dataset(preferences)
# #         FederatedDataset.create_federated_dataset(preferences, X, y, Z, "train")
# #         FederatedDataset.create_federated_dataset(preferences, X_test, y_test, Z_test, "test")

# # for node in range(preferences.num_nodes):
# #     print(f"Node {node}:")
# #     print(torch.load(f"../data/adult/federated/{node}/train.pt").targets)
# #     print(torch.load(f"../data/adult/federated/{node}/test.pt").targets)


# # representative

# preferences = Preferences(
#     dataset="adult",
#     dataset_path="../data/adult/",
#     epochs=10,
#     device="cpu",
#     batch_size=64,
#     seed=42,
#     optimizer="adam",
#     sweep=False,
#     fl_round=1,
#     cross_device=False,
#     tabular=True,
#     num_nodes=10,
#     split_approach="representative_diversity",
#     ratio_unfair_nodes=0.3,
#     group_to_reduce=(0.0, 1),
#     group_to_increment=(1.0, 1),
#     ratio_unfairness=(0.1, 0.2),
# )


# if preferences.tabular:
#     if preferences.cross_device:
#         X, Z, y = LoadDataset.load_dataset(preferences)
#     else:
#         X, Z, y, X_test, Z_test, y_test = LoadDataset.load_dataset(preferences)
#         FederatedDataset.create_federated_dataset(preferences, X, y, Z, "train")
#         FederatedDataset.create_federated_dataset(
#             preferences, X_test, y_test, Z_test, "test"
#         )

# for node in range(preferences.num_nodes):
#     print(f"Node {node}:")
#     print(torch.load(f"../data/adult/federated/{node}/train.pt").targets)
#     print(torch.load(f"../data/adult/federated/{node}/test.pt").targets)
