@REM docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v %cd%/model-repository:/models nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models
@REM RUn no gpu
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v %cd%/model-repository:/models nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models
