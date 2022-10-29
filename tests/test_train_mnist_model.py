
from torch.utils.data import DataLoader

from train import Optimizer, train
from models import LinearModel
from data import load_train_dataset, load_test_dataset

from train_mnist_model import mnist_loss, model_accuracy

import torch

torch.manual_seed(42)

def test_linear_model_accuracy_improves():
    train_dataset = load_train_dataset()
    test_dataset = load_test_dataset()
    dl = DataLoader(train_dataset, batch_size=40, shuffle=True)
    
    model = LinearModel(28*28, 1)

    opt = Optimizer(model.parameters(), 0.1)

    accuracy_before = model_accuracy(model, test_dataset) 

    train(model, dl, test_dataset, opt, 10, mnist_loss, model_accuracy)

    accuracy_after = model_accuracy(model, test_dataset)

    assert accuracy_after > (1.2 * accuracy_before), "accuracy did not improve by 20 percent"
    assert accuracy_after > 0.9, "accuracy not over 95%"