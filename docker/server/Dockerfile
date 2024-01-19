FROM docker.io/asakhare/llava_base as base

# Set environment variable for the model path
ENV MODEL_PATH=./data/liuhaotian/llava-v1.5-13b \
    HOST=0.0.0.0 \
    PORT=7860

# Expose port
EXPOSE $PORT

CMD ["/bin/bash", "-c", "source /venv/llava/bin/activate llava && \
    /venv/llava/bin/python -m llava.serve.controller --host $HOST --port 10000 & \
    /venv/llava/bin/python -m llava.serve.gradio_web_server --controller http://localhost:10000 --port $PORT --model-list-mode reload & \
    /venv/llava/bin/python -m llava.serve.model_worker --host $HOST --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path $MODEL_PATH && \
    wait"]
