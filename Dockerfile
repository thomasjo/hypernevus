FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update && apt-get install --yes rsync

ADD requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
