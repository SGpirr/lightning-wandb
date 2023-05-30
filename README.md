## Install

### Conda env
```bash
export CONDA_ENV=lightning-wandb
conda create -n $CONDA_ENV python=3.9 -y
conda activate $CONDA_ENV
conda install nb_conda_kernels -y
pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name $CONDA_ENV --display-name $CONDA_ENV
```
### python packages
```bash
pip install -r requirements.txt
```

