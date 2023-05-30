import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
from lightning_module import LitMNIST, MNISTDataModule, pl
from lightning.pytorch.cli import LightningCLI
from decouple import config
LOG_ROOT = config('LOG_ROOT')

# wandb.init(settings=wandb.Settings(code_dir="."), project='MNIST')

# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         # parser.link_arguments("model.tokenizer_path", "data.tokenizer_path")
#         parser.add_lightning_class_args(WandbLogger, "my_wandb_logger")
#         parser.set_defaults({"my_early_stopping.monitor": "val_loss", "my_early_stopping.patience": 5})

def main():    
    cli = LightningCLI(LitMNIST, MNISTDataModule,
                      save_config_kwargs={"overwrite": True}
                      )

if __name__ == "__main__":
    main()