from torch import no_grad
from tqdm.autonotebook import tqdm

import torch

import torch.nn.functional as F

class Optimizer():
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
        
    def zero_grad(self):
        for param in self.params:
            param.grad = None
    
    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad.data

def training_step(model, batch, opt, loss_f):
    x, y = batch
    preds = model(x)
    loss = loss_f(preds, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss

def mnist_loss(logits, truths):
    return F.cross_entropy(logits, truths)

def model_accuracy(model, dl, num=None):
    accs = []
    model.eval()
    i = 0
    for _, batch in enumerate(dl):
        x, y = batch
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        accs.append((preds == y).float().mean())
        i+=1
        if num is not None and i > num:
            break
    model.train()
    return torch.stack(accs).mean()

def train_epoch(model, dl, val_dataset, opt, loss_f, metrics_f, pbar):
    for idx, batch in enumerate(dl):
        loss = training_step(model, batch, opt, loss_f)

        if idx % 40 == 0:
            acc = metrics_f(model, dl, num=5)
            pbar.set_postfix_str(f'loss: {loss:.4f}, acc: {acc:.4f}')

      
def train(model, dl, val_dl, opt, epochs, loss_f, metric_f):
    model.eval()
    print(f'Accuracy before {metric_f(model, val_dl):.4f}')
    model.train()
    with tqdm(total=epochs, unit='e') as pbar:
        for _ in range(epochs):
            train_epoch(model, dl, val_dl, opt, loss_f, metric_f, pbar)
            pbar.update(1)
            print(f'Epoch accuracy {metric_f(model, val_dl):.4f}')
    model.eval()
    print(f'Accuracy after {metric_f(model, val_dl):.4f}')
    model.train()