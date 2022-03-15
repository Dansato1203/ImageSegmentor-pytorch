#! /bin/bash
DOCKER_IMAGE="pytorch_train:sf_train"

gpu="--gpus=all"
if ! command -v nvidia-smi &> /dev/null
then
    echo "NVIDIA Driver not be found"
    gpu=""
fi

xhost +
docker run $gpu -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --runtime=nvidia \
    --name pytorch_train \
    $DOCKER_IMAGE \
    bash
