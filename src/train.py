from tqdm.autonotebook import tqdm

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
    with tqdm(total=epochs, unit='e') as pbar:
        for epoch in range(epochs):
            loss = train_epoch(model, dl, opt, loss_f)
            print(f'loss {loss}')
            pbar.update(1)