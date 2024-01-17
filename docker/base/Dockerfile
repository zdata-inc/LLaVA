# Use an official CUDA-enabled Python runtime as a parent image
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y  \
            software-properties-common \
            apt-utils \
            wget \
            git \
            pciutils  # Required for nvidia-smi

# Install NVIDIA drivers and utilities
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    apt-get update && \
    apt-get -y install nvidia-driver-525=525.147.05-0ubuntu1

# Install Miniconda
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda -u && \
#    rm Miniconda3-latest-Linux-x86_64.sh

# Add Conda to the system PATH
#ENV PATH="/miniconda/bin:${PATH}"

# Activate the Conda environment
#RUN echo "conda activate llava" >> /etc/profile.d/conda.sh
#RUN echo "source /etc/profile.d/conda.sh" >> /.bashrc

SHELL ["/bin/bash", "--login", "-c"]

#RUN conda create -n llava python=3.10 -y && \
#    source activate llava && \
#    conda install pip && \
#    pip install --upgrade pip && \
#    pip install -e . && \
#    pip install -e ".[train]" && \
#    pip install flash-attn --no-build-isolation

# Install Python 3.10
RUN apt-get install -y python3.10 python3.10-venv

# Create a virtual environment (venv) named llava
RUN python3.10 -m venv /venv/llava && \
    source /venv/llava/bin/activate && \
    pip install --upgrade pip && \
    pip install -e . && \
    pip install -e ".[train]" && \
    pip install flash-attn

# Set the working directory
WORKDIR /

# Copy LLaVA repo
RUN mkdir /LLaVA
COPY . /LLaVA

WORKDIR /LLaVA

# Create data and output directories
RUN mkdir data && mkdir output
