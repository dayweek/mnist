import torch

INPUT_DIM = 784
OUTPUT_DIM = 1

class Linear():
    def __init__(self, args=None):
        args = vars(args) if args is not None else {}
        self.params = [torch.randn(args.get("input_dim", INPUT_DIM), requires_grad=True),torch.randn(args.get("output_dim", OUTPUT_DIM), requires_grad=True)]

    def __call__(self, x):
        return x @ self.params[0] + self.params[1]

    def parameters(self):
        return self.params

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--input_dim", type=int, default=INPUT_DIM)
        parser.add_argument("--output_dim", type=int, default=OUTPUT_DIM)
        return parser

MLP_DIM1=784
MLP_DIM2=8
MLP_DIM3=1

class MLP():
    def __init__(self, args=None):
        args = vars(args) if args is not None else {}
        self.lin1 = torch.randn((args.get("mlp_dim1", MLP_DIM1), (args.get("mlp_dim2", MLP_DIM2))), requires_grad=True)
        self.bias1 = torch.randn(args.get("mlp_dim2", MLP_DIM2), requires_grad=True)
        self.lin2 = torch.randn((args.get("mlp_dim2", MLP_DIM2), (args.get("mlp_dim3", MLP_DIM3))), requires_grad=True)
        self.bias2 = torch.randn(args.get("mlp_dim3", MLP_DIM3), requires_grad=True)
        self.params = [self.lin1, self.lin2, self.bias1, self.bias2]

    def __call__(self, x):
        out = x @ self.lin1 + self.bias1
        out = out.max(torch.tensor(0.0))
        out = out @ self.lin2 + self.bias2
        return out

    def parameters(self):
        return self.params

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--mlp_dim1", type=int, default=MLP_DIM1)
        parser.add_argument("--mlp_dim2", type=int, default=MLP_DIM2)
        parser.add_argument("--mlp_dim3", type=int, default=MLP_DIM3)
        return parser