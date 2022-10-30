import torch
from torch.utils.data import DataLoader

from train import Optimizer, train
from models import Linear
from data import load_train_dataset, load_test_dataset
from torch import nn

import torch.nn.functional as F

def mnist_loss(logits, truths):
    return F.cross_entropy(logits, truths)

def model_accuracy(model, x, y):
    model.eval()
    logits = model(x)
    model.train()
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean()

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
