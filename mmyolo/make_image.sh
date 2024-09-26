#!/bin/bash


while [[ "$#" -gt 0 ]]; do
    case $1 in
        # --nocache) NOCACHE="true" ;;
        # -m|--model) MODEL="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Get directory of this script
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DOCKER_DIR=${SOURCE_DIR}/docker

#Get version from autolabel3d/__init__.py


# if [[ ${MODEL} == 'resnet' ]];then
#     IMAGE_NAME=mmdet:res
#     DOCKER_BUILDKIT=1 docker build --build-arg CACHEBUST=$(date +%s) -f ./docker/Dockerfile.resnet --no-cache -t ${IMAGE_NAME} ${DOCKER_DIR}
# elif [[ ${MODEL} == "streampetr" ]];then
IMAGE_NAME=mmy
DOCKER_BUILDKIT=1 docker build --build-arg CACHEBUST=$(date +%s) -f ./docker/Dockerfile --no-cache -t ${IMAGE_NAME} ${DOCKER_DIR}
# fi
