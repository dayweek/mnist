import torch

class LinearModel():
    def __init__(self, input, output):
        self.params = [torch.randn(input, requires_grad=True), torch.randn(output, requires_grad=True)]

    def __call__(self, x):
        return x @ self.params[0] + self.params[1]

    def parameters(self):
        return self.params