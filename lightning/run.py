import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning_module import LitMNIST, MNISTDataModule, pl
from argparse import ArgumentParser

from decouple import config
LOG_ROOT = config('LOG_ROOT')

# wandb.init(settings=wandb.Settings(code_dir="."), project='MNIST')

def main(hparams):    
    wandb_logger = WandbLogger(
        log_model=True, 
        settings=wandb.Settings(code_dir="."), 
        project='MNIST_test',
        log_root=LOG_ROOT,
    )
    
    # setup data
    mnist = MNISTDataModule()

    # setup model - choose different hyperparameters per experiment
    model = LitMNIST(n_layer_1=128, n_layer_2=256, lr=1e-3)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_acc", 
        mode="max", 
        save_top_k=2)

    trainer = pl.Trainer(
        logger=wandb_logger,    # W&B integration
        max_epochs=4,            # number of epochs
        strategy='auto',
        # num_workers=16,
        accelerator=hparams.accelerator, devices=hparams.devices,
        callbacks=[checkpoint_callback],        
    )
    
    Dataloader(dataset, num_workers=8, pin_memory=True)

    trainer.fit(model, mnist)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=-1)
    args = parser.parse_args()

    main(args)