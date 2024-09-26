#!/bin/bash

# Default arguments
COMMAND=bash
NUM_WORKERS=$(nvidia-smi -L | wc -l)
PORT=$(( ${RANDOM} % 9000 + 1000 ))
HOST_NAME=$(cat /etc/hostname)
GPUS_ARGS="--gpus all"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --keep-alive) KEEP_ALIVE="true" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ ${KEEP_ALIVE} == 'true' ]]
then
    DOCKER_ARGS="-tid --restart always"
    COMMAND=""
else
    DOCKER_ARGS="-ti --rm"
fi



IMAGE_NAME=ir_det
IMAGE_NAME_=${IMAGE_NAME//\//_}
IMAGE_NAME_=${IMAGE_NAME_//:/_}

docker run --ipc=host --shm-size=8gb --pid=host \
    ${DOCKER_ARGS} \
    -e HOST_NAME=${HOST_NAME} \
    -e NUM_WORKERS=${NUM_WORKERS} \
    -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${HOME}:${HOME} \
    ${GPUS_ARGS} \
    -p 4${PORT}:8888 \
    --name ${USER}.${IMAGE_NAME_}.${PORT} \
    ${IMAGE_NAME} ${COMMAND}
