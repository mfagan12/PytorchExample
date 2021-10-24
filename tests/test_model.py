'''
Tests for the model functions.
'''
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import model

cnn = model.CNN()


def test_model():
    '''
    Test that the neural network can be properly loaded.
    '''
    assert model.get_model()


def test_get_optimizer():
    opt = model.get_optimizer("adam", list(cnn.parameters()), 0.01)
    assert type(opt) == type(Adam(params=list(cnn.parameters())))


def test_get_loss():
    loss_fn = model.get_loss("cross_entropy")
    assert type(loss_fn) == type(CrossEntropyLoss())
