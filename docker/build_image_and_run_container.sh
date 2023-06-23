#!/bin/bash

set -eux

IMAGE_NAME=$1
CONTAINER_NAME=$2

cd $(dirname $0)

docker build -t $IMAGE_NAME .

docker run -it \
--privileged \
--gpus=all \
--net=host \
--env DISPLAY=$DISPLAY \
--env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
--env TERM=xterm-256color \
--volume /tmp/.X11-unix/:/tmp/.X11-unix \
--volume $HOME/data:/root/data \
--name $CONTAINER_NAME \
$IMAGE_NAME bash
