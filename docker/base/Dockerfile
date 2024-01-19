# Use an official CUDA-enabled Python runtime as a parent image
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on

# Update and install essential packages
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y  \
            software-properties-common \
            apt-utils \
            wget \
            git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-venv

# Set working directory
WORKDIR /LLaVA

# Copy LLaVA repo
COPY . .

# Create a virtual environment (venv) named llava
RUN python3.10 -m venv /venv/llava && \
    /venv/llava/bin/pip install --upgrade pip && \
    /venv/llava/bin/pip install -e . && \
    /venv/llava/bin/pip install -e ".[train]" && \
    /venv/llava/bin/pip install flash-attn

# Create data and output directories
RUN mkdir data output
