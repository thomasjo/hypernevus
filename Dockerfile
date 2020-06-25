FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get install --yes rsync

WORKDIR /tmp
ADD requirements.txt .
RUN pip install --upgrade -r requirements.txt

WORKDIR /root
