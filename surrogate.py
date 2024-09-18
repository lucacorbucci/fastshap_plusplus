import os.path
import warnings

# import shapreg  # https://github.com/iancovert/shapley-regression
import torch
import torch.nn as nn

# import Functional torch
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm.auto import tqdm

from fastshap import KLDivLoss
from fastshap.utils import MaskLayer1d
from utils import prepare_data

warnings.simplefilter("ignore")
import argparse

import wandb
from fastshap.surrogate_dp import SurrogateDP
from fastshap.utils import setup_data


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--surrogate_name", type=str, default="")
parser.add_argument("--dataset_name", type=str, default="adult")

# arguments for the surrogate
parser.add_argument("--validation_samples", type=int, default=None)
parser.add_argument("--validation_batch_size", type=int, default=None)


def setup_wandb(args):
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project=args.project_name,
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "epsilon": args.epsilon,
            "gradnorm": args.clipping,
            "optimizer": args.optimizer,
        },
    )
    return wandb_run


def load_model(model_name):
    if os.path.isfile(model_name):
        model = torch.load(model_name)
    else:
        raise FileNotFoundError("Model not found")
    return model


# def prepare_data():
#     rnd = int(str(time.time()).split(".")[1]) * 42

#     # Load and split data
#     X_train, X_test, Y_train, Y_test = train_test_split(
#         *shap.datasets.adult(), test_size=0.2, random_state=42
#     )
#     X_train, X_val, Y_train, Y_val = train_test_split(
#         X_train, Y_train, test_size=0.2, random_state=rnd
#     )

#     # Data scaling
#     num_features = X_train.shape[1]
#     feature_names = X_train.columns.tolist()
#     ss = StandardScaler()
#     ss.fit(X_train)
#     X_train = ss.transform(X_train.values)
#     X_val = ss.transform(X_val.values)
#     X_test = ss.transform(X_test.values)

#     return X_train, X_val, X_test, Y_train, Y_val, Y_test, num_features, feature_names


def get_surrogate_model():
    # Create surrogate model
    surr = nn.Sequential(
        MaskLayer1d(value=0, append=True),
        nn.Linear(2 * num_features, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )
    return surr


def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not found")


args = parser.parse_args()


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

print("FEATURES: ", num_features)

model = load_model(args.model_name)
train_loader, random_sampler, batch_sampler = setup_data(
    train_data=X_train, batch_size=args.batch_size
)
surr = get_surrogate_model()
surr = ModuleValidator.fix(surr)
ModuleValidator.validate(surr, strict=False)
privacy_engine = PrivacyEngine()
optimizer = get_optimizer(args.optimizer, surr, args.lr)

if args.epsilon:
    surr, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=surr,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=args.clipping,
        target_epsilon=args.epsilon,
        target_delta=1e-5,
        epochs=args.epochs,
    )
else:
    surr, optimizer, train_loader = privacy_engine.make_private(
        module=surr,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=10000000000,
        noise_multiplier=0,
    )
print("Created private model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
surr = surr.to(device)
model = model.to(device)
# Set up surrogate object: we pass the model that we have defined as
# surrogate and the number of input features of the training dataset
# we used to train the black box.
surrogate = SurrogateDP(surr, num_features)
print("Created surrogate")

# Set up original model
original_model = nn.Sequential(model, nn.Softmax(dim=1))


wandb_run = setup_wandb(args)

surrogate.train_original_model(
    X_train,  # We pass the training dataset of the black box to the surrogate object
    X_val,  # We pass the validation dataset of the black box to the surrogate object
    original_model,  # black box we want to explain
    batch_size=args.batch_size,
    max_epochs=args.epochs,
    loss_fn=KLDivLoss(),
    validation_samples=args.validation_samples,  # this number is multiplied with the length of the validation dataset
    validation_batch_size=args.validation_batch_size,  # size of the mini batch
    verbose=True,
    lr=args.lr,
    optimizer=optimizer,
    train_loader=train_loader,
    random_sampler=random_sampler,
    batch_sampler=batch_sampler,
    bar=True,
    wandb=wandb,
)

if args.save_model:
    surr.cpu()
    torch.save(surr, f"{args.surrogate_name}.pt")


# Save surrogate
# surr.cpu()
# torch.save(surr, 'census_surrogate.pt')
# surr.to(device)


# # Check for model
# if os.path.isfile("census_surrogate.pt"):
#     # Set up original model
#     def original_model(x):
#         pred = model.predict(x.cpu().numpy())
#         pred = np.stack([1 - pred, pred]).T
#         return torch.tensor(pred, dtype=torch.float32, device=x.device)

#     print("Loading saved surrogate model")
#     surr = torch.load("census_surrogate.pt").to(device)
#     surrogate = SurrogateDP(surr, num_features)
#     surrogate.train_original_model(
#         X_train,  # We pass the training dataset of the black box to the surrogate object
#         X_val,  # We pass the validation dataset of the black box to the surrogate object
#         original_model,  # black box we want to explain
#         batch_size=64,
#         max_epochs=100,
#         loss_fn=KLDivLoss(),
#         validation_samples=10,  # number of samples per validation example
#         validation_batch_size=10000,  # size of the mini batch
#         verbose=True,
#     )

# else:

# Train
# What happens inside the train_original_model
# - A UniformSampler is created to sample the data from the validation set
# - Given the validation set, this is multiplied by validation_samples and
# the UniformSampler will then create a matrix of size (len(validation_set) * validation_samples, num_features)
# Inside this matrix there will be 1 or 0, the value is based on a random threshold. This is to mask some of the features
# - For each sample in this "augmented" matrix we need to compute the corresponding
# prediction of the black box model. This is done by calling the original_model using the
# validation set and then augmenting the prediction to match the size of the augmented groups_matrix
# - Then a validation_set is created with the repeated samples of the original validation set
# the corresponding repeated predictions and the masked features matrix (S_val)
# It is important to notice that the validation set will have the following shape:
# [repeated_val_data, repeated_predictions, masked_features]
# - Then we set the optimizer. Note that this is another hyperparameter but it is set as Adam
# directly from the code.
# - The training loop starts:
#    - We iterate over the batches of the training data
#    - We compute the prediction of the original model for each of the batches
#    - We compute the prediction of the surrogate model on the batch masked using a sampling mask (S)
#    - We compute the loss using the prediction of the surrogate model and the prediction of the original model
#    - We compute the gradients and update the surrogate model
#    - After each batch, we evaluate the surrogate model on the validation set
