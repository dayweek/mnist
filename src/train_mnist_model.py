import torch
from IPython.display import display
from torch.utils.data import DataLoader

from train import Optimizer, train
from models import Linear
from data import load_train_dataset, load_test_dataset
from torch import nn

def mnist_loss(preds, truths):
    s = preds.sigmoid().flatten()

    return torch.where(truths == 1, 1 - s, s).mean()

def model_accuracy(model, test_dataset):
    test_x, test_y = test_dataset.x, test_dataset.y
    preds = model(test_x).sigmoid().flatten()
    return ((preds > 0.5) == test_y).float().mean()

def train_mnist_model(save=False):
    epochs = 10
    lr = 0.1
    batch_size = 40

    dl = DataLoader(load_train_dataset(), batch_size=batch_size, shuffle=True)
    test_dataset = load_test_dataset()
    
    model = Linear()
    opt = Optimizer(model.parameters(), lr)

    train(model, dl, test_dataset, opt, epochs, mnist_loss, model_accuracy)

    if save:
        torch.save(model.params, 'mnist_model.pt')
