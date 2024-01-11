FROM docker.io/asakhare/llava_base as base


# Set environment variable for the model path
ENV MODEL_PATH ./data/finetune/liuhaotian/llava-v1.5-13b

CMD ["/bin/bash", "-c", "source /miniconda/bin/activate llava && \
    /miniconda/envs/llava/bin/python -m llava.serve.controller --host 0.0.0.0 --port 10000 & \
    /miniconda/envs/llava/bin/python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload & \
    /miniconda/envs/llava/bin/python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path $MODEL_PATH && \
    wait"]
