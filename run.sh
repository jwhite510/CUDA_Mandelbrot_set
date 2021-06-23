#!/bin/bash
sudo docker build -t nvidiatest .
# mount this directory to the image
# allow docker image to open a window
xhost +
sudo docker run -v "$(pwd)":/app -it\
	-e DISPLAY=unix$DISPLAY \
	-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
	--gpus all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	nvidiatest bash
