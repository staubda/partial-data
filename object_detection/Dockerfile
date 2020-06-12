FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

RUN apt-get update && apt-get install -y git protobuf-compiler

# Download object detection research repo and set it up
RUN git clone https://github.com/tensorflow/models.git
ARG research_dir=/tf/models/research
WORKDIR $research_dir
RUN protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH=$PYTHONPATH:$research_dir:$research_dir/slim

# Uninstall numpy so version that definitely works can be installed
RUN pip uninstall -y numpy

# Install research repo python deps
RUN pip install \
    numpy==1.17.1 \
    Cython \
    contextlib2 \ 
    pillow \
    lxml \
    jupyter \
    matplotlib \
    tf_slim

# Install research repo python deps that depend on the previous deps being installed
RUN pip install pycocotools
