#! /bin/bash
DOCKER_IMAGE="pytorch_train:latest"

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
	-v $PWD/train_result:/train/train_result:rw \
	$DOCKER_IMAGE \
