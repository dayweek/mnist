import argparse
import importlib
import torch

from data_modules.mnist import MNISTDataModule

from src.train import Optimizer, train, mnist_loss, model_accuracy
import torch.nn.functional as F

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
    parser.add_argument('--epochs', default=2, type=int)
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

    module = MNISTDataModule('data')
    module.prepare_data()
    module.setup()

    dl = module.train_dataloader()
    val_dl = module.val_dataloader()
    
    opt = opt_class(model.parameters(), lr)

    train(model, dl, val_dl, opt, epochs, mnist_loss, model_accuracy)

    torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    main()