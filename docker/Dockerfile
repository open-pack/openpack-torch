# == Stage.1: Make requirements.txt by poetry ==
FROM python:3.10-slim as builder
WORKDIR /root/src/
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export --without-hashes --dev --output requirements.txt

# == Stage.2: Build Custom Images (GPU enabled) ==
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
LABEL pytorch="2.1.1" \
    cuda="12.1" \
    cudnn="8"
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential packages via apt
USER root
RUN apt-get update && apt-get install -y \
    git \
    emacs \
    wget \
    curl \
    build-essential \
    iputils-ping \
    net-tools \
    sudo
# Change timezone
RUN apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
ENV TZ Asia/Tokyo
RUN pip install --upgrade pip
# Setup Alias
RUN echo 'alias python=python3' >> ~/.bashrc

# -- Setup Working Directory --
ENV HOME /root
ENV SHELL /bin/bash
WORKDIR /content
ENV PATH ${PATH}:/root/.local/bin

# -- Python Setting --
COPY --from=builder /root/src/requirements.txt .
RUN pip install --no-deps --verbose -r /content/requirements.txt
RUN pip install jupyterlab

