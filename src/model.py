"""
This module contains the model architecture and functions for
constructing relevant components, i.e. the loss function and optimizer.
"""
import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
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
        output = F.log_softmax(x, dim=1)
        return output


def get_optimizer(optimizer_string: str, model_parameters, learning_rate: float):
    """
    Get optimizer from name string, params, and learning rate. Supported
    optimizer names: adam, sgd.
    """
    optimizers = {
        "adam": torch.optim.Adam(params=model_parameters, lr=learning_rate),
        "sgd": torch.optim.SGD(params=model_parameters, lr=learning_rate),
    }
    return optimizers[optimizer_string.lower()]


def get_loss(loss_string: str):
    """
    Get loss from name string. Supported loss names: cross_entropy,
    binary_cross_entropy, mean_squared_error.
    """
    losses = {
        "cross_entropy": torch.nn.CrossEntropyLoss(),
        "binary_cross_entropy": torch.nn.BCELoss(),
        "mean_squared_error": torch.nn.MSELoss(),
    }
    return losses[loss_string.lower()]


def get_model():
    """
    Helper function to instantiate the model.
    """
    return CNN()


def main():
    pass


if __name__ == "__main__":
    main()
