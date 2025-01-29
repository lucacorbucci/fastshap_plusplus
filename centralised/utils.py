import random

import numpy as np
import pandas as pd
import shap  # https://github.com/slundberg/shap
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    TensorDataset,
)
from torchvision import datasets, transforms

from fastshap.fastshap_dp import (
    calculate_grand_coalition,
    generate_validation_data,
)
from fastshap.utils import DatasetInputOnly, DatasetRepeat, ShapleySampler


def setup_data_images(args, train_set, val_set, test_set):
    train_surr = DatasetInputOnly(train_set)
    # Set up train data loader.
    if isinstance(train_surr, torch.Tensor):
        train_set = TensorDataset(train_surr)
    elif isinstance(train_surr, Dataset):
        train_set = train_surr
    else:
        raise ValueError("train_data must be either tensor or a " "PyTorch Dataset")

    random_sampler = RandomSampler(
        train_set,
        replacement=True,
        num_samples=int(np.ceil(len(train_set) / args.batch_size)) * args.batch_size,
    )
    print(
        "Random sampler: ",
        int(np.ceil(len(train_set) / args.batch_size)) * args.batch_size,
    )
    batch_sampler = BatchSampler(
        random_sampler, batch_size=args.batch_size, drop_last=True
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        pin_memory=True,
        num_workers=0,
    )
    val_surr = None
    test_surr = None
    if val_set:
        val_surr = DatasetInputOnly(val_set)
    if test_set:
        test_surr = DatasetInputOnly(test_set)

    return train_loader, train_surr, val_surr, test_surr, random_sampler, batch_sampler


def is_image_dataset(dataset_name):
    return dataset_name in ["mnist"]


def is_dataset_supported(dataset_name):
    return dataset_name in ["adult", "dutch", "mnist", "income", "employment"]


def get_optimizer(optimizer, model, lr):
    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not recognized")


class TabularDataset(Dataset):
    def __init__(self, x, y):
        """
        Initialize the custom dataset with x (features), z (sensitive values), and y (targets).

        Args:
        x (list of tensors): List of input feature tensors.
        z (list): List of sensitive values.
        y (list): List of target values.
        """
        self.samples = x
        self.targets = y
        self.indexes = range(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single data point from the dataset.

        Args:
        idx (int): Index to retrieve the data point.

        Returns:
        sample (dict): A dictionary containing 'x', 'z', and 'y'.
        """
        x_sample = self.samples[idx]
        y_sample = self.targets[idx]

        return x_sample, y_sample


def load_dutch():
    dutch_df = pd.read_csv("../../data/dutch/dutch_census.csv")
    # dutch_df = pd.DataFrame(data[0]).astype("int32")

    dutch_df["sex_binary"] = np.where(dutch_df["sex"] == 1, 1, 0)
    dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= 300, 1, 0)

    del dutch_df["sex"]
    del dutch_df["occupation"]

    dutch_df_feature_columns = [
        "age",
        "household_position",
        "household_size",
        "prev_residence_place",
        "citizenship",
        "country_birth",
        "edu_level",
        "economic_status",
        "cur_eco_activity",
        "Marital_status",
        "sex_binary",
    ]

    return dutch_df, dutch_df_feature_columns


## Use this function to retrieve X, X, y arrays for training ML models
def dataset_to_numpy(
    _df,
    _feature_cols: list,
):
    """Args:
    _df: pandas dataframe
    _feature_cols: list of feature column names
    _metadata: dictionary with metadata
    num_sensitive_features: number of sensitive features to use
    sensitive_features_last: if True, then sensitive features are encoded as last columns
    """

    # transform features to 1-hot
    _X = _df[_feature_cols]
    _X2 = pd.get_dummies(_X, drop_first=False)
    esc = MinMaxScaler()
    _X = esc.fit_transform(_X2)

    _y = _df["occupation_binary"].values
    return _X, _y


def prepare_dutch(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tmp = load_dutch()
    feature_names = tmp[1]
    tmp = dataset_to_numpy(*tmp)
    x = tmp[0]
    y = tmp[1]

    xy = list(zip(x, y))
    random.shuffle(xy)
    x, y = zip(*xy)

    test_size = int(len(y) * 0.2)

    x_test = np.array(x[:test_size])
    y_test = np.array(y[:test_size])

    x = np.array(x[test_size:])
    y = np.array(y[test_size:])

    if args.sweep:
        np.random.seed(args.validation_seed)
        random.seed(args.validation_seed)
        torch.manual_seed(args.validation_seed)

    xy = list(zip(x, y))
    random.shuffle(xy)
    x, y = zip(*xy)

    if args.sweep:
        train_size = int(len(y) * 0.8)
        val_size = int(len(y) * 0.2)
    else:
        train_size = int(len(y))
        val_size = 0

    x_train = np.array(x[:train_size])
    y_train = np.array(y[:train_size])

    x_val = None
    y_val = None
    if val_size > 0:
        x_val = np.array(x[train_size:])
        y_val = np.array(y[train_size:])

    train_dataset = TabularDataset(
        x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
        y=y_train.astype(np.float32),
    )

    val_dataset = None
    if val_size > 0:
        val_dataset = TabularDataset(
            x=np.hstack((x_val, np.ones((x_val.shape[0], 1)))).astype(np.float32),
            y=y_val.astype(np.float32),
        )

    test_dataset = TabularDataset(
        x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
        y=y_test.astype(np.float32),
    )

    # append to x_train a column of ones to account for the bias term
    x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
    x_val = np.hstack((x_val, np.ones((x_val.shape[0], 1)))) if val_size > 0 else None
    x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        feature_names,
    )


def prepare_mnist(args):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "/raid/lcorbucci/data_private_fastshap/",
        train=True,
        download=True,
        transform=transform,
    )
    val_dataset = None
    test_dataset = None
    if args.sweep:
        # split the training dataset into training and validation
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [50000, 10000]
        )
    else:
        test_dataset = datasets.MNIST(
            "/raid/lcorbucci/data_private_fastshap/", train=False, transform=transform
        )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


def create_dataset(data, labels):
    # convert Y to tensor
    labels = torch.tensor([0 if item is False else 1 for item in labels])

    # convert data to tensor
    data = torch.tensor(data, dtype=torch.float32)
    return TensorDataset(data, labels)


def data_scaling(X_train, X_val, X_test):
    # Data scaling
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    if X_val is not None:
        X_val = ss.transform(X_val)
    if X_test is not None:
        X_test = ss.transform(X_test)

    return X_train, X_val, X_test


def prepare_data(args):
    # Load and split data
    if args.dataset_name == "adult":
        X_train, X_test, Y_train, Y_test = train_test_split(
            *shap.datasets.adult(), test_size=0.2, random_state=args.seed
        )
        feature_names = X_train.columns.tolist()

        X_val, Y_val = None, None
        if args.sweep:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=0.2, random_state=args.validation_seed
            )
            X_test = None
            Y_test = None
        X_train, X_val, X_test = data_scaling(X_train, X_val, X_test)

        # create a dataloader for training using X_train and Y_train
        train_set = create_dataset(X_train, Y_train)
        if X_val is not None:
            val_set = create_dataset(X_val, Y_val)
        else:
            val_set = None

        if X_test is not None:
            test_set = create_dataset(X_test, Y_test)
        else:
            test_set = None

        return (
            train_set,
            val_set,
            test_set,
            X_train,
            X_val,
            X_test,
            Y_train,
            Y_val,
            Y_test,
            X_train.shape[1],
            feature_names,
        )

    elif args.dataset_name == "dutch":
        (
            train_set,
            val_set,
            test_set,
            X_train,
            X_val,
            X_test,
            Y_train,
            Y_val,
            Y_test,
            features_names,
        ) = prepare_dutch(args)

        return (
            train_set,
            val_set,
            test_set,
            X_train,
            X_val,
            X_test,
            Y_train,
            Y_val,
            Y_test,
            X_train.shape[1],
            features_names,
        )

    elif args.dataset_name == "mnist":
        (
            train_set,
            val_set,
            test_set,
        ) = prepare_mnist(args)

        return (
            train_set,
            val_set,
            test_set,
        )
    elif args.dataset_name == "income":
        feature_names = [f"feature_{i}" for i in range(1, 11)]

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            feature_names,
        )
    elif args.dataset_name == "employment":
        feature_names = [f"feature_{i}" for i in range(1, 17)]

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            feature_names,
        )


def prepare_dataset_for_explainer(
    args,
    train_data,
    val_data,
    batch_size,
    imputer,
    num_samples,
    link,
    device,
    validation_samples,
    num_players,
    validation_seed=None,
    image_dataset=False,
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

    # Set up validation dataset.
    if isinstance(val_data, np.ndarray):
        x_val = torch.tensor(val_data, dtype=torch.float32)
        val_set = TensorDataset(x_val)
    elif isinstance(val_data, torch.Tensor):
        val_set = TensorDataset(val_data)
    elif isinstance(val_data, Dataset):
        val_set = val_data
    else:
        raise ValueError("train_data must be np.ndarray, torch.Tensor or " "Dataset")

    num_workers = 0

    # Grand coalition value.
    grand_train = calculate_grand_coalition(
        train_set,
        imputer,
        batch_size * num_samples,
        link,
        device,
        num_workers,
        num_players,
        image_dataset,
    ).cpu()
    grand_val = calculate_grand_coalition(
        val_set,
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

        null = link(imputer(train_set[0][0].unsqueeze(0).to(device), zeros))

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
    if validation_seed is not None:
        torch.manual_seed(validation_seed)
    val_S, val_values = generate_validation_data(
        val_set,
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
    val_set = DatasetRepeat([val_set, TensorDataset(grand_val, val_S, val_values)])
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * num_samples,
        pin_memory=True,
        num_workers=num_workers,
    )

    print("Dataset ready for the explainer")
    return train_loader, val_loader, grand_train, grand_val, null, sampler
