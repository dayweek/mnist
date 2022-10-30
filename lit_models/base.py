import pytorch_lightning as pl
from src.train_mnist_model import mnist_loss
import torch.optim as optim

class BaseLitModel(pl.LightningModule):
    def __init__(self, model, args=None):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outs = self(x)
        loss = mnist_loss(outs, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer