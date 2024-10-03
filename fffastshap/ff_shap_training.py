from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from tqdm.auto import tqdm

from fastshap.utils import ShapleySampler


class Data(Dataset):
    def __init__(self, data):
        # Collect samples, both cat and dog and store pairs of (filepath, label) in a simple list.
        self._samples = data

    def __getitem__(self, index):
        # Access the stored path and label for the correct index
        example, label = self._samples[index]

        return example, label

    def __len__(self):
        """Total number of samples"""
        return len(self._samples)

    def get_sample_by_id(self, id_):
        id_index = [path.stem for (path, _) in self._samples].index(id_)
        return self[id_index]


def additive_efficient_normalization(pred, grand, null):
    gap = (grand - null) - torch.sum(pred, dim=1)
    # gap = gap.detach()
    return pred + gap.unsqueeze(1) / pred.shape[1]


def cosine_loss(pred, y):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_similarity = cos(y, pred)

    loss = 1 - cosine_similarity
    return loss.sum()


def augment_data(model, data):
    class_prob = [i.cpu().detach().numpy() for i in model.predict_proba(data, S)]

    device = data.device
    data = data.cpu().detach().numpy()

    aug_data = []
    for i in range(len(data)):
        aug_data.append(np.append(data[i], [class_prob[i]]))

    aug_data = torch.tensor(np.array(aug_data), dtype=torch.float32, device=device)

    return aug_data


class FF_SHAP_Training:
    def __init__(
        self, explainer, black_box, num_players, num_classes, augmentation=False
    ):
        self.explainer = explainer
        self.black_box = black_box
        self.num_players = num_players
        self.num_classes = num_classes
        self.augmentation = augmentation

    def train(
        self,
        train_data,
        train_targets,
        val_data,
        val_targets,
        batch_size,
        num_samples,
        max_epochs,
        loss_func="cosine",
        lr=2e-4,
        min_lr=1e-5,
        lr_factor=0.5,
        verbose=False,
        sampling=True,
        bar=False,
        lookback=5,
    ):
        dataset_tuples = [
            (torch.tensor(d, dtype=torch.float32), torch.tensor(l, dtype=torch.float32))
            for d, l in zip(train_data, train_targets)
        ]

        train_loader = DataLoader(
            Data(data=dataset_tuples),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        dataset_tuples = [
            (torch.tensor(d, dtype=torch.float32), torch.tensor(l, dtype=torch.float32))
            for d, l in zip(val_data, val_targets)
        ]

        val_loader = DataLoader(
            Data(data=dataset_tuples), batch_size=batch_size, pin_memory=True
        )

        explainer = self.explainer
        black_box = self.black_box
        num_players = self.num_players
        num_classes = self.num_classes
        augmentation = self.augmentation
        device = next(explainer.parameters()).device
        sampler = ShapleySampler(num_players)

        zeros = torch.zeros(1, num_players, dtype=torch.float32, device=device)
        S = sampler.sample(batch_size * num_samples, paired_sampling=True)
        S = S.to(device)
        null = black_box.predict_proba(zeros, S)

        if loss_func == "cosine":
            loss_fn = cosine_loss
        else:
            loss_fn = nn.MSELoss()

        optimizer = optim.Adam(explainer.parameters(), lr=lr)
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
        train_loss = 0

        for epoch in range(max_epochs):
            explainer.train()
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc="Training epoch")
            else:
                batch_iter = train_loader

            for x, y in batch_iter:
                x = x.to(device)
                y = y.to(device)

                if sampling:
                    S = sampler.sample(batch_size * num_samples, paired_sampling=True)
                    S = S.to(device)
                    x = (
                        x.unsqueeze(1)
                        .repeat(1, num_samples, *[1 for _ in range(len(x.shape) - 1)])
                        .reshape(batch_size * num_samples, *x.shape[1:])
                    )

                    # x = x * S

                    y = (
                        y.unsqueeze(1)
                        .repeat(1, num_samples, *[1 for _ in range(len(y.shape) - 1)])
                        .reshape(batch_size * num_samples, *y.shape[1:])
                    )

                    y = y.reshape(y.shape[0], 1) * S.repeat(1, self.num_classes)

                grand = black_box.predict_proba(x, S)
                if augmentation:
                    augmented_x = augment_data(black_box, x)
                    pred = explainer(augmented_x)
                else:
                    pred = explainer(x)

                pred = pred.reshape(len(x), num_players, -1)
                pred = additive_efficient_normalization(pred, grand, null)
                pred = pred.reshape(len(x), num_players * num_classes)

                explainer.zero_grad()
                loss = loss_fn(pred, y)
                train_loss += loss

                print(loss)
                # Take gradient step.
                loss.backward()
                optimizer.step()
                explainer.zero_grad()
                del x, y
                torch.cuda.empty_cache()

            explainer.eval()
            val_loss = 0
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                S = sampler.sample(batch_size * num_samples, paired_sampling=False)
                S = S.to(device)
                x = (
                    x.unsqueeze(1)
                    .repeat(1, num_samples, *[1 for _ in range(len(x.shape) - 1)])
                    .reshape(batch_size * num_samples, *x.shape[1:])
                )

                # x = x * S
                grand = black_box.predict_proba(x, S)
                if augmentation:
                    augmented_x = augment_data(black_box, x)
                    pred = explainer(augmented_x)
                else:
                    pred = explainer(x)

                pred = pred.reshape(len(x), num_players, -1)
                pred = additive_efficient_normalization(pred, grand, null)
                pred = pred.reshape(len(x), num_players * num_classes)

                # y = y.unsqueeze(1).repeat(
                #    1, num_samples, *[1 for _ in range(len(y.shape) - 1)]
                # ).reshape(batch_size * num_samples, *y.shape[1:])

                # y = y * S.repeat(1, self.num_classes)

                val_loss += loss_fn(pred, y)
                del x, y
                torch.cuda.empty_cache()

                # Save loss, print progress.
            if verbose:
                print("----- Epoch = {} -----".format(epoch + 1))
                print("Val loss = {:.6f}".format(val_loss))
                print("")
            scheduler.step(val_loss)

            # Check for convergence.
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model = deepcopy(explainer)
                if verbose:
                    print("New best epoch, loss = {:.6f}".format(val_loss))
                    print("")
            elif epoch - best_epoch == lookback:
                if verbose:
                    print("Stopping early at epoch = {}".format(epoch))
                break

        # Copy best model.
        for param, best_param in zip(explainer.parameters(), best_model.parameters()):
            param.data = best_param.data
        explainer.eval()

    def predict(self, data):
        explainer = self.explainer
        black_box = self.black_box
        num_players = self.num_players
        device = next(explainer.parameters()).device

        # Set up train dataset.
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=device)
        else:
            raise ValueError("train_data must be np.ndarray, torch.Tensor")

        zeros = torch.zeros(1, self.num_players, dtype=torch.float32, device=device)
        S = torch.zeros(1, self.num_players, dtype=torch.float32, device=device)
        S = S.to(device)
        null = black_box.predict_proba(zeros, S)

        grand = black_box.predict_proba(data, S)
        if self.augmentation:
            augmented_x = augment_data(black_box, data)
            pred = explainer(augmented_x).reshape(len(data), num_players, -1)
        else:
            pred = explainer(data).reshape(len(data), num_players, -1)

        pred = pred.reshape(len(data), num_players, -1)
        pred = additive_efficient_normalization(pred, grand, null)

        return pred
