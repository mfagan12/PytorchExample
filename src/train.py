"""
This module will contain training and evaluation code for the neural
network.
"""
import torch
from torch import nn
import yaml
from rich.progress import track

import data
import model


def train_step(model, opt, criterion, inputs, labels, batch: int, size: int):
    """Perform a training update for a single batch within an epoch of model
    training."""
    model.train()

    pred = model(inputs)
    loss = criterion(pred, labels)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss


def fit(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    num_epochs: int,
    device,
) -> None:
    """Train the model on the data for the specified number of epochs,
    using the given optimizer and loss function.

    Args:
        model (Module): the model to train
        train_data (DataLoader): data to use for training
        val_data (DataLoader): data to monitor metrics
        optimizer (Optimizer): the optimizer to use
        loss (Loss): the objective function for training
        epochs (int): number of epochs to train for
    """
    model.to(device)
    size = len(train_dataloader)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for batch, (data, labels) in enumerate(
            track(train_dataloader, description=f"Training epoch {epoch}:")
        ):
            data.to(device)
            labels.to(device)
            loss = train_step(model, optimizer, loss_fn, data, labels, batch, size)


def evaluate(test_data):
    pass


def main() -> None:
    with open("../config/train_params.yaml", "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    train_data, val_data = data.make_dataloaders(*data.load_mnist(), batch_size=16)

    cnn = model.get_model()
    optimizer = model.get_optimizer(
        optimizer_string=params["optimizer"],
        model_parameters=nn.ParameterList(cnn.parameters()),
        learning_rate=params["learning_rate"],
    )
    loss_fn = model.get_loss(params["loss"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Training...")
    fit(
        model=cnn,
        train_dataloader=train_data,
        val_dataloader=val_data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=params["epochs"],
        device=device,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
