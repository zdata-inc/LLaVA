# Use an official CUDA-enabled Python runtime as a parent image
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on

# Set the working directory
WORKDIR /

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y  \
            software-properties-common \
            apt-utils \
            wget \
            git

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda -u && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Activate the Conda environment
RUN /miniconda/bin/conda create -n llava python=3.10 -y && \
    echo "conda activate llava" >> /.bashrc

# Stage 2: Install LLaVA and python modules
FROM base as setup

# Copy LLaVA repo
RUN mkdir /LLaVA
COPY . /LLaVA

#ARG LLAVA_COMMIT=9a26bd1435b4ac42c282757f2c16d34226575e96
#RUN git clone https://github.com/haotian-liu/LLaVA.git && \
#    cd LLaVA && \
#    git checkout ${LLAVA_COMMIT}

WORKDIR /LLaVA

# Create data and output directories
RUN mkdir data && mkdir output

SHELL ["/bin/bash", "--login", "-c"]

RUN source /miniconda/bin/activate llava && \
    conda install pip && \
    pip install --upgrade pip && \
    pip install -e . && \
    pip install -e ".[train]" && \
    pip install flash-attn --no-build-isolation

ENV LORA_ENABLE=true \
    LORA_R=value \
    LORA_ALPHA=value \
    DEEPSPEED=./scripts/zero2.json \
    MODEL_NAME_OR_PATH=./data/lmsys/vicuna-13b-v1.5 \
    VERSION=plain \
    DATA_PATH=./data/blip_laion_cc_sbu_558k.json \
    IMAGE_FOLDER=./data/images \
    VISION_TOWER=./data/openai/clip-vit-large-patch14-336 \
    MM_PROJECTOR_TYPE=mlp2x_gelu \
    TUNE_MM_MLP_ADAPTER=True \
    MM_VISION_SELECT_LAYER=-2 \
    MM_USE_IM_START_END=False \
    MM_USE_IM_PATCH_TOKEN=False \
    BF16=True \
    OUTPUT_DIR=./outputs \
    NUM_TRAIN_EPOCHS=1 \
    PER_DEVICE_TRAIN_BATCH_SIZE=32 \
    PER_DEVICE_EVAL_BATCH_SIZE=4 \
    GRADIENT_ACCUMULATION_STEPS=1 \
    EVALUATION_STRATEGY=no \
    SAVE_STRATEGY=steps \
    SAVE_STEPS=24000 \
    SAVE_TOTAL_LIMIT=1 \
    LEARNING_RATE=1e-3 \
    WEIGHT_DECAY=0. \
    WARMUP_RATIO=0.03 \
    LR_SCHEDULER_TYPE=cosine \
    LOGGING_STEPS=1 \
    TF32=True \
    MODEL_MAX_LENGTH=2048 \
    GRADIENT_CHECKPOINTING=True \
    DATALOADER_NUM_WORKERS=4 \
    LAZY_PREPROCESS=True \
    REPORT_TO=wandb

# Default entry point to the training script with conditional logic
ENTRYPOINT ["/bin/bash", "-c", "/miniconda/envs/llava/bin/deepspeed llava/train/train_mem.py \
                                    --lora_enable $LORA_ENABLE \
                                    --lora_r $LORA_R \
                                    --lora_alpha $LORA_ALPHA \
                                    --deepspeed $DEEPSPEED \
                                    --model_name_or_path $MODEL_NAME_OR_PATH \
                                    --version $VERSION \
                                    --data_path $DATA_PATH \
                                    --image_folder $IMAGE_FOLDER \
                                    --vision_tower $VISION_TOWER \
                                    --mm_projector_type $MM_PROJECTOR_TYPE \
                                    --tune_mm_mlp_adapter $TUNE_MM_MLP_ADAPTER \
                                    --mm_vision_select_layer $MM_VISION_SELECT_LAYER \
                                    --mm_use_im_start_end $MM_USE_IM_START_END \
                                    --mm_use_im_patch_token $MM_USE_IM_PATCH_TOKEN \
                                    --bf16 $BF16 \
                                    --output_dir $OUTPUT_DIR \
                                    --num_train_epochs $NUM_TRAIN_EPOCHS \
                                    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
                                    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
                                    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                                    --evaluation_strategy $EVALUATION_STRATEGY \
                                    --save_strategy $SAVE_STRATEGY \
                                    --save_steps $SAVE_STEPS \
                                    --save_total_limit $SAVE_TOTAL_LIMIT \
                                    --learning_rate $LEARNING_RATE \
                                    --weight_decay $WEIGHT_DECAY \
                                    --warmup_ratio $WARMUP_RATIO \
                                    --lr_scheduler_type $LR_SCHEDULER_TYPE \
                                    --logging_steps $LOGGING_STEPS \
                                    --tf32 $TF32 \
                                    --model_max_length $MODEL_MAX_LENGTH \
                                    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
                                    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
                                    --lazy_preprocess $LAZY_PREPROCESS \
                                    --report_to $REPORT_TO $@; \
                                    "]

# Default CMD without any arguments
CMD []
