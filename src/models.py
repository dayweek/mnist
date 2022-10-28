class LinearModel():
    def __init__(self, weights, bias):
        self.params = [weights, bias]

    def __call__(self, x):
        return x @ self.params[0] + self.params[1]