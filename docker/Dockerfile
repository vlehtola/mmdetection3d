ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG MMENG="0.10.3"
ARG MMCV="2.1.0"
ARG MMDET="3.3.0"
ARG MMDET3D="1.4.0"

ENV PYTHONUNBUFFERED TRUE

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && rm -f /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list

# Install the required packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    # MMDet3D Requirements
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH" \
    FORCE_CUDA="1"

# TORCHSEVER
RUN pip install torchserve torch-model-archiver

#conda create --name openmmlab python=3.8 -y
#conda activate openmmlab

## install torch-relavant libraries
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


# Install MMEngine, MMCV and MMDetection
RUN pip install openmim
RUN mim install mmengine==${MMENG}
RUN mim install mmcv==${MMCV}
RUN mim install mmdet==${MMDET}
RUN mim install mmdet3d==${MMDET3D}

    # TORCHSEVER
RUN pip install torchserve torch-model-archiver

    # Needed for itckul las files
RUN pip install laspy

# Install MMDetection3D
RUN conda clean --all \
    && git clone https://github.com/vlehtola/mmdetection3d.git -b main /mmdetection3d \
    && cd /mmdetection3d \
    && pip install -e .

WORKDIR /mmdetection3d

# Add a build argument for invalidating cache, use with
# docker build --build-arg CACHEBUST=$(date +%s)
ARG CACHEBUST=1

# For development, Check for updates that are not cached
RUN git pull
