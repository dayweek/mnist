import argparse
import importlib
import torch

from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

from torch.utils.data import DataLoader

from src.train import Optimizer, train
from src.data import load_train_dataset, load_test_dataset
import torch.nn.functional as F

def mnist_loss(logits, truths):
    return F.cross_entropy(logits, truths)

def model_accuracy(model, x, y):
    model.eval()
    logits = model(x)
    model.train()
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean()

def import_class(class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'."""
    #module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module("src.models")
    class_ = getattr(module, class_name)
    return class_

def setup_parser():
    # General Args
    parser = argparse.ArgumentParser()

    # Train Args
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--optimizer', default='Optimizer', type=str)
    parser.add_argument('--model_class', default='Linear')

    # Model specicif Args
    temp_args, _ = parser.parse_known_args()
    model_class = import_class(temp_args.model_class)
    model_group = parser.add_argument_group("Model Args")
    if "add_to_argparse" in dir(model_class):
        model_class.add_to_argparse(model_group)
    return parser

def main():
    print("Starting")
    parser = setup_parser()
    args = parser.parse_args()

    model_class = import_class(args.model_class)
    model = model_class(args)
    lr = args.lr
    epochs = args.epochs
    if args.optimizer == 'Optimizer':
        opt_class = Optimizer
    else:
        opt_class = getattr(torch.optim, args.optimizer)
    
    train_dataset = load_train_dataset()
    test_dataset = load_test_dataset()
    dl = DataLoader(train_dataset, batch_size=40, shuffle=True)
    
    opt = opt_class(model.parameters(), lr)

    train(model, dl, test_dataset, opt, epochs, mnist_loss, model_accuracy)

if __name__ == "__main__":
    main()