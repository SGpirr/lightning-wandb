## 도커 컨테이너 실행
주피터랩이 설치된 nvidia cuda 컨테이너 입니다.
- 브라우저 주소창에 localhost:40070(또는 원하는 포트 설정)로 접속
- 초기 jupyter lab 패스워드: univa
- "-v ${PWD}:/home": docker run을 실행하는 작업공간을 컨테이너의 /home 디렉터리와 마운트하는 옵션입니다.  
    컨테이너의 /home 에서 저장된 결과물은 컨테이너 외부에서도 접근 가능하고, 종료 후에도 사라지지 않습니다.
### windows wsl2
```bash
docker run -d --name test --gpus -1 -v ${PWD}:/home -p 40070-40075:40070-40075 hooy9765/template:ubuntu20_cu118 bash start_jupyter.sh --port 40070
```
### linux
```bash
docker run -d --name test --gpus -1 -v ${PWD}:/home -p 40070-40075:40070-40075 hooy9765/dip_template:latest bash start_jupyter.sh --port 40070
```

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
fastapi 서버를 실행하면, 작성한 api 함수를 자동으로 분석하여 docs 페이지를 생성해줍니다.  
해당 docs페이지 기본 엔드포인트 경로는 /docs입니다
- docs page: http://0.0.0.0:40071/docs  
- api endpoint: http://0.0.0.0:40071/predict