from torch import no_grad
from tqdm.autonotebook import tqdm

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

def validation_step(model, batch, metrics_f):
    x, y = batch
    with no_grad():
        accuracy = metrics_f(model, x, y)
    return accuracy

def train_epoch(model, dl, opt, loss_f, metrics_f, pbar):
    loss_average = AverageLastN(30)
    for _, batch in enumerate(dl):
        loss = training_step(model, batch, opt, loss_f)
        accuracy = validation_step(model, batch, metrics_f)
        pbar.set_postfix_str(f'loss: {loss_average.add(loss):.4f}')
      
def train(model, dl, val_dataset, opt, epochs, loss_f, metric_f):
    model.eval()
    print(f'Accuracy before {metric_f(model, val_dataset.x, val_dataset.y):.4f}')
    model.train()
    with tqdm(total=epochs, unit='e', mininterval=0.5) as pbar:
        for _ in range(epochs):
            train_epoch(model, dl, opt, loss_f, metric_f, pbar)
            pbar.update(1)
    model.eval()
    print(f'Accuracy after {metric_f(model, val_dataset.x, val_dataset.y):.4f}')


class AverageLastN():
    def __init__(self, n):
        self.n = n
        self.a = []

    def add(self, x):
        self.a.append(x)
        if len(self.a) > self.n:
            self.a.pop(0)

        return sum(self.a) / len(self.a)