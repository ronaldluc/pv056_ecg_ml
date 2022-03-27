#FROM tensorflow/tensorflow:latest-gpu
FROM tensorflow/tensorflow:latest

RUN apt-get update \
    && apt-get install -y \
      unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/work
COPY requirements.txt requirements.txt

# Install from requirements.txt
RUN pip install -r requirements.txt

WORKDIR /home/work/src
