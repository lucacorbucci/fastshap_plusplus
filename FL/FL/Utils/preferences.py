from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Preferences:
    dataset: str
    dataset_path: str
    # The number of epochs that we want to use during the training
    epochs: int
    batch_size: int
    # seed to use during the training
    seed: int
    node_shuffle_seed: int
    optimizer: str
    learning_rate: float
    # regularization is True if we want to use the regularization to
    # reduce the unfairness of the model during the training
    epsilon: float = None
    sweep: bool = False
    fl_rounds: int = None
    noise_multiplier: float = None
    cross_device: bool = False
    tabular: bool = False
    num_nodes: int = None
    split_approach: str = None
    alpha_dirichlet: float = None
    ratio_unfair_nodes: float = None
    group_to_reduce: tuple = None
    group_to_increment: tuple = None
    ratio_unfairness: tuple = None
    validation: bool = False

    num_training_nodes: int = None
    num_validation_nodes: int = None
    num_test_nodes: int = None

    epsilon: float = None
    clipping: float = None
    noise_multiplier: float = None
    validation_size: float = None
    fed_dir: str = None
    project_name: str = None
    run_name: str = None
    wandb: bool = False

    sampled_training_nodes: float = None
    sampled_validation_nodes: float = None
    sampled_test_nodes: float = None

    device: str = None

    save_local_models: bool = False
    save_aggregated_model: bool = False
    aggregated_model_name: str = None

    test_size: float = None

    fraction_fit_nodes: float = None
    fraction_validation_nodes: float = None
    fraction_test_nodes: float = None

    train_explainer: bool = False

    train_surrogate: bool = False
    bb_name: str = None

    validation_samples: int = None
    num_samples: int = None
    validation_batch_size: int = None

    surrogate_name: str = None
    paired_sampling: bool = True
    eff_lambda: float = None
