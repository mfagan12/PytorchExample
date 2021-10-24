"""
This module contains utilities for loading and processing datasets for
the neural network.
"""
import torch
import torchvision
from torchvision.transforms import ToTensor


def load_mnist():
    train_data = torchvision.datasets.MNIST(
        root="../data/", train=True, download=True, transform=ToTensor()
    )
    val_data = torchvision.datasets.MNIST(
        root="../data/", train=False, download=True, transform=ToTensor()
    )
    return train_data, val_data


def make_dataloaders(train_data, val_data, batch_size=16):
    train_dl = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return train_dl, val_dl


def main():
    train_data, val_data = load_mnist()
    print(train_data)
    print(val_data)


if __name__ == "__main__":
    main()
