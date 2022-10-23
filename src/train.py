from PIL import Image
from torch import tensor
import torch
import src.utils
from pathlib import Path
from IPython.display import display
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from torch import nn

from torchvision import transforms
convert_tensor = transforms.ToTensor()
image_from_tensor = transforms.ToPILImage()

def train_and_save_model(download=False):
    def convert(a):
        return 0 if a == 7 else 1

    if download:
        src.utils.download_mnist("data")

    test_labels = src.utils.read_labels('data/t10k-labels-idx1-ubyte', 'test')
    train_labels = src.utils.read_labels('data/train-labels-idx1-ubyte', 'train')
    keys = list(train_labels)
    test_keys = list(test_labels)

    train_x = torch.stack([convert_tensor(Image.open(Path("data") / key)) for key in keys if train_labels[key] in [3, 7]]).squeeze()
    train_x = train_x.view(-1, 28*28)
    train_y = tensor([train_labels[key] for key in keys if train_labels[key] in [3,7]]).squeeze()
    train_y = tensor([ convert(i) for i in train_y])


    test_x = torch.stack([convert_tensor(Image.open(Path("data") / key)) for key in test_keys if test_labels[key] in [3, 7]]).squeeze()
    test_x = test_x.view(-1, 28*28)
    test_y = tensor([test_labels[key] for key in test_keys if test_labels[key] in [3,7]]).squeeze()
    test_y = tensor([ convert(i) for i in test_y])

    dset = list(zip(train_x,train_y))

    weights = torch.randn(28*28).requires_grad_()
    bias = torch.randn(1).requires_grad_()
    dataloader = DataLoader(dset, batch_size=40, shuffle=True)
    lr = 0.04
    epochs = 15

    def mnist_loss(preds, truths):
        s = preds.sigmoid()
        return torch.where(truths == 1, 1 - s, s).mean()

    def accuracy(weights, bias):
        preds = (test_x @ weights + bias).sigmoid()
        return ((preds > 0.5) == test_y).float().mean()

    def train_epoch(dl, weights, bias, pbar):
        for _, batch in enumerate(dl):
            x, y = batch
            preds = x @ weights + bias
            loss = mnist_loss(preds, y)
            loss.backward()
            weights.data -= lr * weights.grad.data
            bias.data -= lr * bias.grad.data
            weights.grad.zero_()
            bias.grad.zero_()
        print(loss)
    
    print(f'accuracy before {accuracy(weights, bias):.4f}')

    with tqdm(total=epochs, unit='e') as pbar:
        for i in range(epochs):
            train_epoch(dataloader, weights, bias, pbar)
            pbar.update(1)

    print(f'accuracy after {accuracy(weights, bias):.4f}')

    model = {"weights": weights, "bias":bias}
    torch.save(model, 'linear_model.pt')
