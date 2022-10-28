import torch
from IPython.display import display
from torch.utils.data import DataLoader

from train import Optimizer, train
from models import LinearModel
from data import load_train_dataset, load_test_dataset

def train_mnist_model(model, save=False):
    def mnist_loss(preds, truths):
        s = preds.sigmoid()
        return torch.where(truths == 1, 1 - s, s).mean()

    def model_accuracy(model, test_dataset):
        test_x, test_y = test_dataset
        preds = model(test_x).sigmoid()
        return ((preds > 0.5) == test_y).float().mean()

    test_dataset = load_test_dataset()
    dl = DataLoader(load_train_dataset(), batch_size=40, shuffle=True)
    
    weights = torch.randn(28*28)
    bias = torch.randn(1)
    model = LinearModel(weights, bias)
    opt = Optimizer(model.params, 0.1)

    print(f'accuracy before {model_accuracy(model, test_dataset):.4f}')
    train(model, dl, opt, 10, mnist_loss)
    print(f'accuracy after {model_accuracy(model, test_dataset):.4f}')

    if save:
        torch.save(model.params, 'mnist_model.pt')