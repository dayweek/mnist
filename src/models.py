import torch

INPUT_DIM = 784
OUTPUT_DIM = 10

from torch import nn
import torch.nn.functional as F

class Linear():
    def __init__(self, args=None):
        args = vars(args) if args is not None else {}
        self.params = [torch.randn(args.get("input_dim", INPUT_DIM), args.get("output_dim", OUTPUT_DIM), requires_grad=True),torch.randn(args.get("output_dim", OUTPUT_DIM), requires_grad=True)]

    def __call__(self, x):
        return x.flatten(1) @ self.params[0] + self.params[1]

    def parameters(self):
        return self.params

    def train(self):
        pass

    def eval(self):
        pass

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--input_dim", type=int, default=INPUT_DIM)
        parser.add_argument("--output_dim", type=int, default=OUTPUT_DIM)
        return parser

MLP_DIM1=784
MLP_DIM2=20
MLP_DIM3=10

class MLP():
    def __init__(self, args=None):
        args = vars(args) if args is not None else {}
        self.lin1 = torch.randn((args.get("mlp_dim1", MLP_DIM1), (args.get("mlp_dim2", MLP_DIM2))), requires_grad=True)
        self.bias1 = torch.randn(args.get("mlp_dim2", MLP_DIM2), requires_grad=True)
        self.lin2 = torch.randn((args.get("mlp_dim2", MLP_DIM2), (args.get("mlp_dim3", MLP_DIM3))), requires_grad=True)
        self.bias2 = torch.randn(args.get("mlp_dim3", MLP_DIM3), requires_grad=True)
        self.params = [self.lin1, self.lin2, self.bias1, self.bias2]

    def __call__(self, x):
        out = x.flatten(1) @ self.lin1 + self.bias1
        out = out.max(torch.tensor(0.0))
        out = out @ self.lin2 + self.bias2
        return out

    def train(self):
        pass

    def eval(self):
        pass

    def parameters(self):
        return self.params

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--mlp_dim1", type=int, default=MLP_DIM1)
        parser.add_argument("--mlp_dim2", type=int, default=MLP_DIM2)
        parser.add_argument("--mlp_dim3", type=int, default=MLP_DIM3)
        return parser

class SimpleLenet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2) # -> 6 channels, 28x28
        self.pool = nn.MaxPool2d(2) # -> 6 channels, 14x14
        self.conv2 = nn.Conv2d(6, 120, 14) #-> 120 channels, 1x1
        self.fc1 = nn.Linear(120, 10)
        self.fc2 = nn.Linear(10, 10)

    def __call__(self, x):
        xx = F.relu(self.conv1(x))
        xx = F.relu(self.pool(xx))
        xx = F.relu(self.conv2(xx))
        xx = xx.flatten(1)
        xx = F.relu(self.fc1(xx))
        return self.fc2(xx)