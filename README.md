## 설치

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

### wandb 설정
```bash
wandb login
```

## 훈련
- "lightning-wandb" 환경 필요
```bash
cd lightning
python lightning_cli.py --config config.yaml fit
```

## 배포
- "lightning-wandb" 환경 필요
```bash
cd deploy
python fast.py
```
### api 설명
- docs page: http://0.0.0.0:8889/docs  
- api endpoint: http://0.0.0.0:8889/predict