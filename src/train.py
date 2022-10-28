from PIL import Image
from torch import tensor
import torch
import src.utils
from pathlib import Path
from IPython.display import display
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from torchvision import transforms

convert_tensor = transforms.ToTensor()
image_from_tensor = transforms.ToPILImage()

def train_and_save_model(download=False, save=False):
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

    def mnist_loss(preds, truths):
        s = preds.sigmoid()
        return torch.where(truths == 1, 1 - s, s).mean()

    def model_accuracy(model, dl):
        preds = model(test_x).sigmoid()
        return ((preds > 0.5) == test_y).float().mean()
  
    class Optimizer():
        def __init__(self, params, lr):
            for param in params:
                param.requires_grad_()
            self.params = params
            self.lr = lr
            
        def zero_grad(self):
            for param in self.params:
                param.grad = None
        
        def step(self):
            for param in self.params:
                param.data -= self.lr * param.grad.data

    def train_epoch(model, dl, opt, loss_f):
        for _, batch in enumerate(dl):
            x, y = batch
            preds = model(x)
            loss = loss_f(preds, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        return loss
                
                
    def train(model, dl, opt, epochs, loss_f):
        print(f'accuracy before {model_accuracy(model, dl):.4f}')
        with tqdm(total=epochs, unit='e') as pbar:
            for epoch in range(epochs):
                loss = train_epoch(model, dl, opt, loss_f)
                print(f'loss {loss}')
                pbar.update(1)
        print(f'accuracy after {model_accuracy(model, dl):.4f}')


    dl = DataLoader(dset, batch_size=40, shuffle=True)

    def linear_model(x):
        return x @ weights + bias
    
    weights = torch.randn(28*28)
    bias = torch.randn(1)
    
    opt = Optimizer([weights, bias], 0.1)
    train(linear_model, dl, opt, 10, mnist_loss)

    if save:
        model = {"weights": weights, "bias":bias}
        torch.save(model, 'linear_model.pt')

train_and_save_model(download=False, save=False)