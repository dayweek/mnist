
from src.models import Linear, MLP, SimpleLenet
from src.train import mnist_loss, model_accuracy, Optimizer, train
from data_modules.mnist import MNISTDataModule

import torch

torch.manual_seed(42)

def test_linear_model_accuracy_improves():
    module = MNISTDataModule('data')
    module.prepare_data()
    module.setup()

    dl = module.train_dataloader()
    val_dl = module.val_dataloader()
    
    model = Linear()

    opt = Optimizer(model.parameters(), 0.1)

    accuracy_before = model_accuracy(model, val_dl) 

    train(model, dl, val_dl, opt, 2, mnist_loss, model_accuracy)

    accuracy_after = model_accuracy(model, val_dl)

    assert accuracy_after > (1.2 * accuracy_before), "accuracy did not improve by 20 percent"
    assert accuracy_after > 0.7, "accuracy is low"

def test_mlp_model_accuracy_improves():
    module = MNISTDataModule('data')
    module.prepare_data()
    module.setup()

    dl = module.train_dataloader()
    val_dl = module.val_dataloader()
    
    model = MLP()

    opt = Optimizer(model.parameters(), 0.1)

    accuracy_before = model_accuracy(model, val_dl) 

    train(model, dl, val_dl, opt, 2, mnist_loss, model_accuracy)

    accuracy_after = model_accuracy(model, val_dl) 

    assert accuracy_after > (1.2 * accuracy_before), "accuracy did not improve by 20 percent"
    assert accuracy_after > 0.7, "accuracy is low"

def test_simplelenet_model_accuracy_improves():
    module = MNISTDataModule('data')
    module.prepare_data()
    module.setup()

    dl = module.train_dataloader()
    val_dl = module.val_dataloader()
    
    model = SimpleLenet()

    opt = torch.optim.SGD(model.parameters(), 0.1)

    accuracy_before = model_accuracy(model, val_dl) 

    train(model, dl, val_dl, opt, 2, mnist_loss, model_accuracy)

    accuracy_after = model_accuracy(model, val_dl)

    assert accuracy_after > (1.2 * accuracy_before), "accuracy did not improve by 20 percent"
    assert accuracy_after > 0.7, "accuracy is low"