#ARG FROM_IMAGE=tensorflow/tensorflow:latest-gpu
ARG FROM_IMAGE=tensorflow/tensorflow:latest
FROM $FROM_IMAGE

RUN apt-get update \
    && apt-get install -y \
      unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/work
COPY requirements.txt requirements.txt

# Install from requirements.txt
RUN pip install -r requirements.txt

WORKDIR /home/work/src
