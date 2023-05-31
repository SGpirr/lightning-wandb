from lightning_module import LitMNIST, MNISTDataModule
from lightning.pytorch.cli import LightningCLI
            
            
def main():    
    cli = LightningCLI(LitMNIST, MNISTDataModule,
                      save_config_kwargs={"overwrite": True}
                      )

if __name__ == "__main__":
    main()