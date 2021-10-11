ARG CUDA=11.0
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
      build-essential \
      cmake \
      git \
      wget \
      vim \
      less \
    && rm -rf /var/lib/apt/lists/*

# Time zone
ENV LANG C.UTF-8
ENV TZ Asia/Shanghai

# Install Miniconda package manger.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda package manger.(LC made)
# COPY ./Miniconda3-4.4.10-Linux-x86_64.sh /tmp/Miniconda3-4.4.10-Linux-x86_64.sh
# RUN bash /tmp/Miniconda3-4.4.10-Linux-x86_64.sh  -b -p /opt/conda \
#     && rm /tmp/Miniconda3-latest-Linux-x86_64.sh


# Install conda packages.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda update -qy conda \
    && conda install -y -c conda-forge \
      pip \
      python==3.6.9 \
      numpy pandas \
      tqdm scipy \
      scikit-learn \
      matplotlib \
      tensorflow-gpu==2.0 tensorflow-estimator==2.0 keras==2.3.1

RUN pip install --upgrade pip && pip install biopython==1.68

WORKDIR /home/LOGO-master


