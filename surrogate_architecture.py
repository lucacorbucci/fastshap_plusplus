import torch
import torch.nn as nn
import torch.nn.functional as F

from fastshap.utils import (
    DatasetInputOnly,
    KLDivLoss,
    MaskLayer1d,
    MaskLayer2d,
)


class ImageSurrogate(nn.Module):
    def __init__(self):
        super(ImageSurrogate, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1)  # Change the input channels from 1 to 2
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x


def get_tabular_surrogate(num_features):
    return nn.Sequential(
        MaskLayer1d(value=0, append=True),
        nn.Linear(2 * num_features, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )


def get_image_surrogate(args):
    if args.dataset_name == "mnist":
        return nn.Sequential(MaskLayer2d(value=0, append=True), ImageSurrogate())
    else:
        raise FileNotFoundError("Dataset not suppoted")
