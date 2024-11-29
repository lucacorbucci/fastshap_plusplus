from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from fastshap.utils import DatasetRepeat, ShapleySampler


def additive_efficient_normalization(pred, grand, null):
    """
    Apply additive efficient normalization.

    Args:
      pred: model predictions.
      grand: grand coalition value.
      null: null coalition value.
    """
    gap = (grand - null) - torch.sum(pred, dim=1)
    # gap = gap.detach()
    return pred + gap.unsqueeze(1) / pred.shape[1]


def multiplicative_efficient_normalization(pred, grand, null):
    """
    Apply multiplicative efficient normalization.

    Args:
      pred: model predictions.
      grand: grand coalition value.
      null: null coalition value.
    """
    ratio = (grand - null) / torch.sum(pred, dim=1)
    # ratio = ratio.detach()
    return pred * ratio.unsqueeze(1)


def evaluate_explainer(
    explainer, normalization, x, grand, null, num_players, inference=False
):
    """
    Helper function for evaluating the explainer model and performing necessary
    normalization and reshaping operations.

    Args:
      explainer: explainer model.
      normalization: normalization function.
      x: input.
      grand: grand coalition value.
      null: null coalition value.
      num_players: number of players.
      inference: whether this is inference time (or training).
    """
    # Evaluate explainer.
    pred = explainer(x)

    # Reshape SHAP values.
    if len(pred.shape) == 4:
        # Image.
        image_shape = pred.shape
        pred = pred.reshape(len(x), -1, num_players)
        pred = pred.permute(0, 2, 1)

    else:
        # Tabular.
        image_shape = None
        pred = pred.reshape(len(x), num_players, -1)

    # For pre-normalization efficiency gap.
    total = pred.sum(dim=1)

    # Apply normalization.
    if normalization:
        pred = normalization(pred, grand, null)

    # Reshape for inference.
    if inference:
        if image_shape is not None:
            pred = pred.permute(0, 2, 1)
            pred = pred.reshape(image_shape)

        return pred
    return pred, total


def calculate_grand_coalition(
    dataset,
    imputer,
    batch_size,
    link,
    device,
    num_workers,
    num_features,
    image_dataset,
):
    """
    Calculate the value of grand coalition for each input.

    Args:
      dataset: dataset object.
      imputer: imputer model.
      batch_size: minibatch size.
      num_players: number of players.
      link: link function.
      device: torch device.
      num_workers: number of worker threads.
    """
    ones = torch.ones(batch_size, num_features, dtype=torch.float32, device=device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    with torch.no_grad():
        grand = []
        for (x,) in loader:
            output = imputer(x.to(device), ones[: len(x)].to(device))
            grand.append(link(output))

        # Concatenate and return.
        grand = torch.cat(grand)
        if len(grand.shape) == 1:
            grand = grand.unsqueeze(1)

    return grand


def calculate_grand_coalition_FL(
    dataset,
    imputer,
    batch_size,
    link,
    device,
    num_workers,
    num_features,
    image_dataset,
):
    """
    Calculate the value of grand coalition for each input.

    Args:
      dataset: dataset object.
      imputer: imputer model.
      batch_size: minibatch size.
      num_players: number of players.
      link: link function.
      device: torch device.
      num_workers: number of worker threads.
    """
    ones = torch.ones(batch_size, num_features, dtype=torch.float32, device=device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    with torch.no_grad():
        grand = []
        for x, _, _ in loader:
            output = imputer(x.to(device), ones[: len(x)].to(device))
            grand.append(link(output))

        # Concatenate and return.
        grand = torch.cat(grand)
        if len(grand.shape) == 1:
            grand = grand.unsqueeze(1)

    return grand


def generate_validation_data(
    val_set,
    imputer,
    validation_samples,
    sampler,
    batch_size,
    link,
    device,
    num_workers,
    num_players,
    image_dataset,
):
    """
    Generate coalition values for validation dataset.

    Args:
      val_set: validation dataset object.
      imputer: imputer model.
      validation_samples: number of samples per validation example.
      sampler: Shapley sampler.
      batch_size: minibatch size.
      link: link function.
      device: torch device.
      num_workers: number of worker threads.
    """
    # Generate coalitions.
    val_S = sampler.sample(
        validation_samples * len(val_set), paired_sampling=True
    ).reshape(len(val_set), validation_samples, num_players)

    # Get values.
    val_values = []
    for i in range(validation_samples):
        # Set up data loader.
        dset = DatasetRepeat([val_set, TensorDataset(val_S[:, i])])
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        values = []

        for x, S in loader:
            values.append(link(imputer(x.to(device), S.to(device))).cpu().data)

        val_values.append(torch.cat(values))

    val_values = torch.stack(val_values, dim=1)
    return val_S, val_values


def generate_validation_data_FL(
    val_set,
    imputer,
    validation_samples,
    sampler,
    batch_size,
    link,
    device,
    num_workers,
    num_players,
    image_dataset,
):
    """
    Generate coalition values for validation dataset.

    Args:
      val_set: validation dataset object.
      imputer: imputer model.
      validation_samples: number of samples per validation example.
      sampler: Shapley sampler.
      batch_size: minibatch size.
      link: link function.
      device: torch device.
      num_workers: number of worker threads.
    """
    # Generate coalitions.
    val_S = sampler.sample(
        validation_samples * len(val_set), paired_sampling=True
    ).reshape(len(val_set), validation_samples, num_players)

    # Get values.
    val_values = []
    for i in range(validation_samples):
        # Set up data loader.
        dset = DatasetRepeat([val_set, TensorDataset(val_S[:, i])])
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        values = []

        for x, _, _, _, S in loader:
            values.append(link(imputer(x.to(device), S.to(device))).cpu().data)

        val_values.append(torch.cat(values))

    val_values = torch.stack(val_values, dim=1)
    return val_S, val_values


def validate(
    val_loader, imputer, explainer, null, link, normalization, num_players, device
):
    """
    Calculate mean validation loss.

    Args:
      val_loader: validation data loader.
      imputer: imputer model.
      explainer: explainer model.
      null: null coalition value.
      link: link function.
      normalization: normalization function.
    """
    with torch.no_grad():
        # Setup.
        mean_loss = 0
        N = 0
        loss_fn = nn.MSELoss()

        for x, grand, S, values in val_loader:
            # Move to device.
            x = x.to(device)
            S = S.to(device)
            grand = grand.to(device)
            values = values.to(device)

            # Evaluate explainer.
            pred, _ = evaluate_explainer(
                explainer, normalization, x, grand, null, num_players
            )

            # Calculate loss.
            print("Shapes: ", S.shape, pred.shape)
            mul = torch.matmul(S, pred)
            approx = null + mul
            loss = loss_fn(approx, values)

            # Update average.
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N

    return mean_loss


def validate_FL(
    val_loader,
    imputer,
    explainer,
    null,
    link,
    normalization,
    num_players,
    device,
    FL_evaluation=False,
    num_samples=None,
    eff_lambda=None,
):
    """
    Calculate mean validation loss.

    Args:
      val_loader: validation data loader.
      imputer: imputer model.
      explainer: explainer model.
      null: null coalition value.
      link: link function.
      normalization: normalization function.
    """
    with torch.no_grad():
        # Setup.
        mean_loss = 0
        N = 0
        loss_fn = nn.MSELoss()

        for x, _, _, _, grand, S, values in val_loader:
            # Move to device.
            x = x.to(device)
            S = S.to(device)
            grand = grand.to(device)
            values = values.to(device)

            # Evaluate explainer.
            pred, total = evaluate_explainer(
                explainer, normalization, x, grand, null, num_players
            )

            mul = torch.matmul(S, pred)
            approx = null + mul
            loss = loss_fn(approx, values)

            # Update average.
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N

    return mean_loss if isinstance(mean_loss, float) else mean_loss.item()


class FastSHAP:
    """
    Wrapper around FastSHAP explanation model.

    Args:
      explainer: explainer model (torch.nn.Module).
      imputer: imputer model (e.g., fastshap.Surrogate).
      normalization: normalization function for explainer outputs ('none',
        'additive', 'multiplicative').
      link: link function for imputer outputs (e.g., nn.Softmax).
    """

    def __init__(
        self, explainer, imputer, num_features, normalization="none", link=None
    ):
        # Set up explainer, imputer and link function.
        self.explainer = explainer
        self.imputer = imputer
        self.num_players = num_features
        self.null = None
        if link is None or link == "none":
            self.link = nn.Identity()
        elif isinstance(link, nn.Module):
            self.link = link
        else:
            raise ValueError("unsupported link function: {}".format(link))

        # Set up normalization.
        if normalization is None or normalization == "none":
            self.normalization = None
        elif normalization == "additive":
            self.normalization = additive_efficient_normalization
        elif normalization == "multiplicative":
            self.normalization = multiplicative_efficient_normalization
        else:
            raise ValueError("unsupported normalization: {}".format(normalization))

    def train(
        self,
        train_loader,
        val_loader,
        grand_train,
        grand_val,
        null,
        batch_size,
        num_samples,
        max_epochs,
        lr=2e-4,
        min_lr=1e-5,
        lr_factor=0.5,
        eff_lambda=0,
        paired_sampling=True,
        validation_samples=None,
        lookback=5,
        training_seed=None,
        validation_seed=None,
        num_workers=0,
        bar=False,
        verbose=False,
        optimizer=None,
        wandb=None,
        sampler=None,
        image_dataset=False,
    ):
        """
        Train explainer model.

        Args:
          train_data: training data with inputs only (np.ndarray, torch.Tensor,
            torch.utils.data.Dataset).
          val_data: validation data with inputs only (np.ndarray, torch.Tensor,
            torch.utils.data.Dataset).
          batch_size: minibatch size.
          num_samples: number of training samples.
          max_epochs: max number of training epochs.
          lr: initial learning rate.
          min_lr: minimum learning rate.
          lr_factor: learning rate decrease factor.
          eff_lambda: lambda hyperparameter for efficiency penalty.
          paired_sampling: whether to use paired sampling.
          validation_samples: number of samples per validation example.
          lookback: lookback window for early stopping.
          training_seed: random seed for training.
          validation_seed: random seed for generating validation data.
          num_workers: number of worker threads in data loader.
          bar: whether to show progress bar.
          verbose: verbosity.
        """
        # Set up explainer model.
        explainer = self.explainer
        num_players = self.num_players
        imputer = self.imputer
        link = self.link
        normalization = self.normalization
        explainer.train()
        device = next(explainer.parameters()).device
        self.null = null

        # Verify other arguments.
        if validation_samples is None:
            validation_samples = num_samples

        # Setup for training.
        loss_fn = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_factor,
            patience=lookback // 2,
            min_lr=min_lr,
            verbose=verbose,
        )
        self.loss_list = []
        best_loss = np.inf
        best_epoch = -1
        best_model = None
        if training_seed is not None:
            torch.manual_seed(training_seed)
        MAX_PHYSICAL_BATCH_SIZE = 512
        for epoch in range(max_epochs):
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                optimizer=optimizer,
            ) as memory_safe_data_loader:
                # Batch iterable.
                batch_iter = memory_safe_data_loader

                for x, grand in batch_iter:
                    # Sample S.
                    random_batch_size = x.shape[0]
                    S = sampler.sample(
                        random_batch_size * num_samples, paired_sampling=paired_sampling
                    )

                    # Move to device.
                    x = x.to(device)
                    S = S.to(device)
                    grand = grand.to(device)

                    # Evaluate value function.
                    x_tiled = (
                        x.unsqueeze(1)
                        .repeat(1, num_samples, *[1 for _ in range(len(x.shape) - 1)])
                        .reshape(random_batch_size * num_samples, *x.shape[1:])
                    )

                    with torch.no_grad():
                        values = link(imputer(x_tiled, S))

                    # Evaluate explainer.
                    pred, total = evaluate_explainer(
                        explainer, normalization, x, grand, null, num_players
                    )

                    # Calculate loss.
                    S = S.reshape(random_batch_size, num_samples, num_players)
                    values = values.reshape(random_batch_size, num_samples, -1)
                    matm = torch.matmul(S, pred)

                    # if image_dataset:
                    #     print("Shape: ", null.shape, matm.shape, pred.shape, S.shape)
                    #     sys.exit()
                    approx = null + matm  # torch.matmul(S, pred)

                    loss = loss_fn(approx, values)
                    if eff_lambda:
                        loss = loss + eff_lambda * loss_fn(total, grand - null)

                    # Take gradient step.
                    loss = loss * num_players
                    wandb.log({"train_loss": loss, "epoch": epoch})

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Evaluate validation loss.
                if val_loader is not None:
                    explainer.eval()
                    val_loss = (
                        num_players
                        * validate(
                            val_loader,
                            imputer,
                            explainer,
                            null,
                            link,
                            normalization,
                            num_players=self.num_players,
                            device=device,
                        ).item()
                    )
                    wandb.log({"validation_loss": val_loss, "epoch": epoch})
                    explainer.train()

        # # Copy best model.
        # for param, best_param in zip(explainer.parameters(), best_model.parameters()):
        #     param.data = best_param.data
        explainer.eval()

    def shap_values(self, x):
        """
        Generate SHAP values.

        Args:
          x: input examples.
        """
        # Data conversion.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("data must be np.ndarray or torch.Tensor")

        # Ensure null coalition is calculated.
        device = next(self.explainer.parameters()).device
        if self.null is None:
            with torch.no_grad():
                zeros = torch.zeros(
                    1, self.num_players, dtype=torch.float32, device=device
                )
                null = self.link(self.imputer(x[:1].to(device), zeros))
            if len(null.shape) == 1:
                null = null.reshape(1, 1)
            self.null = null

        # Generate explanations.
        with torch.no_grad():
            # Calculate grand coalition (for normalization).
            if self.normalization:
                grand = calculate_grand_coalition(
                    x,
                    self.imputer,
                    len(x),
                    self.link,
                    device,
                    0,
                    num_features=self.num_players,
                )
            else:
                grand = None

            # Evaluate explainer.
            x = x.to(device)
            pred = evaluate_explainer(
                self.explainer,
                self.normalization,
                x,
                grand,
                self.null,
                self.num_players,
                inference=True,
            )

        return pred.cpu().data.numpy()

    def train_FL(
        self,
        train_loader,
        grand_train,
        null,
        batch_size,
        num_samples,
        max_epochs,
        lr=2e-4,
        min_lr=1e-5,
        lr_factor=0.5,
        eff_lambda=0,
        paired_sampling=True,
        lookback=5,
        training_seed=None,
        num_workers=0,
        bar=False,
        verbose=False,
        optimizer=None,
        sampler=None,
        image_dataset=False,
        device=None,
    ):
        """
        Train explainer model.

        Args:
          train_data: training data with inputs only (np.ndarray, torch.Tensor,
            torch.utils.data.Dataset).
          val_data: validation data with inputs only (np.ndarray, torch.Tensor,
            torch.utils.data.Dataset).
          batch_size: minibatch size.
          num_samples: number of training samples.
          max_epochs: max number of training epochs.
          lr: initial learning rate.
          min_lr: minimum learning rate.
          lr_factor: learning rate decrease factor.
          eff_lambda: lambda hyperparameter for efficiency penalty.
          paired_sampling: whether to use paired sampling.
          validation_samples: number of samples per validation example.
          lookback: lookback window for early stopping.
          training_seed: random seed for training.
          validation_seed: random seed for generating validation data.
          num_workers: number of worker threads in data loader.
          bar: whether to show progress bar.
          verbose: verbosity.
        """
        # Set up explainer model.
        explainer = self.explainer
        num_players = self.num_players
        imputer = self.imputer
        link = self.link
        normalization = self.normalization
        explainer.train()
        self.null = null

        # Setup for training.
        loss_fn = nn.MSELoss()
        self.loss_list = []
        if training_seed is not None:
            torch.manual_seed(training_seed)
        MAX_PHYSICAL_BATCH_SIZE = 512
        for epoch in range(max_epochs):
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                optimizer=optimizer,
            ) as memory_safe_data_loader:
                # Batch iterable.
                batch_iter = memory_safe_data_loader

                for x, _, _, grand in batch_iter:
                    # Sample S.
                    random_batch_size = x.shape[0]
                    S = sampler.sample(
                        random_batch_size * num_samples, paired_sampling=paired_sampling
                    )

                    # Move to device.
                    x = x.to(device)
                    S = S.to(device)
                    grand = grand.to(device)

                    # Evaluate value function.
                    x_tiled = (
                        x.unsqueeze(1)
                        .repeat(1, num_samples, *[1 for _ in range(len(x.shape) - 1)])
                        .reshape(random_batch_size * num_samples, *x.shape[1:])
                    )

                    with torch.no_grad():
                        values = link(imputer(x_tiled, S))

                    # Evaluate explainer.
                    pred, total = evaluate_explainer(
                        explainer, normalization, x, grand, null, num_players
                    )

                    # Calculate loss.
                    S = S.reshape(random_batch_size, num_samples, num_players)
                    values = values.reshape(random_batch_size, num_samples, -1)
                    matm = torch.matmul(S, pred)

                    # if image_dataset:
                    #     print("Shape: ", null.shape, matm.shape, pred.shape, S.shape)
                    #     sys.exit()
                    approx = null + matm  # torch.matmul(S, pred)

                    loss = loss_fn(approx, values)
                    if eff_lambda:
                        loss = loss + eff_lambda * loss_fn(total, grand - null)

                    # Take gradient step.
                    loss = loss * num_players

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    self.loss_list.append(loss.item())

        # # Copy best model.
        # for param, best_param in zip(explainer.parameters(), best_model.parameters()):
        #     param.data = best_param.data
        explainer.eval()

        # return the average loss
        mean_loss = np.mean(self.loss_list)
        return mean_loss if isinstance(mean_loss, float) else mean_loss.item()
