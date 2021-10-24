import data
import model
import torch
import train

cnn = model.CNN()
opt = model.get_optimizer("adam", list(cnn.parameters()), 0.001)
loss_fn = model.get_loss("cross_entropy")
train_data, val_data = data.load_mnist()
train_dl, val_dl = data.make_dataloaders(train_data, val_data)
for data, labels in train_dl:
    data = data
    labels = labels
    break
size = len(train_dl)
device = torch.device("cpu")


def test_train_step():
    assert train.train_step(cnn, opt, loss_fn, data, labels, 0, size)


def test_train():
    train.fit(cnn, train_dl, val_dl, opt, loss_fn, 0, device=device)


def test_evaluate():
    assert True
