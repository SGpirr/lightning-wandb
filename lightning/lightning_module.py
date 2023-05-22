# Weights & Biases
import wandb
from lightning.pytorch.loggers import WandbLogger

# Pytorch modules
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Pytorch-Lightning
import lightning.pytorch as pl

# Dataset
from torchvision.datasets import MNIST
from torchvision import transforms



## model module
class LitMNIST(pl.LightningModule):

    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        '''method used to define our model parameters'''
        super().__inita__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, n_layer_1)
        self.layer_2 = torch.nn.Linear(n_layer_1, n_layer_2)
        self.layer_3 = torch.nn.Linear(n_layer_2, n_classes)

        # optimizer parameters
        self.lr = lr

        # metrics
        self.accuracy = pl.metrics.Accuracy()

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters()

    def forward(self, x):
        '''method used for inference input -> output'''

        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Log training loss
        self.log('train_loss', loss)

        # Log metrics
        #self.log('train_acc', self.accuracy(logits, y))

        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Log validation loss (will be automatically averaged over an epoch)
        self.log('valid_loss', loss)

        # Log metrics
        #self.log('valid_acc', self.accuracy(logits, y))

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Log test loss
        self.log('test_loss', loss)

        # Log metrics
        #self.log('test_acc', self.accuracy(logits, y))
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr)