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
