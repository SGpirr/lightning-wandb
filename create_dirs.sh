export data_dir=data
export log_dir=logs
export model_dir=models

# 디렉터리 생성
mkdir -p $data_dir $log_dir $model_dir

# .env 파일에 생성한 디렉터리 경로 저장
(
echo DATA_ROOT=$PWD/$data_dir && 
echo LOG_ROOT=$PWD/$log_dir &&
echo MODEL_ROOT=$PWD/$model_dir
) >> .env