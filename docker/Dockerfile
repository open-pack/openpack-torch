# == Stage.1: Make requirements.txt by poetry ==
FROM python:3.7-slim as builder
WORKDIR /root/src/
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt > requirements.txt

# == Stage.2: Build Custom Images (GPU enabled) ==
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
LABEL pytorch="1.11.0" \
    cuda="11.3" \
    cudnn="8"
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential packages via apt
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
USER root
ENV HOME /root
ENV SHELL /bin/bash
WORKDIR /content
ENV PATH ${PATH}:/root/.local/bin

# -- Python Setting --
RUN echo 'alias python=python3' >> ~/.bashrc
COPY --from=builder /root/src/requirements.txt .
RUN pip install -r /content/requirements.txt --no-deps
RUN pip install jupyterlab matplotlib seaborn

