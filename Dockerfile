FROM nvidia/cuda:9.2-devel-ubuntu18.04

# RUN which nvcc
RUN apt-get update
RUN apt-get install cmake -y

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install python3-tk -y
RUN apt-get install x11-xserver-utils -y
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install ipdb
RUN pip3 install pudb

# install SFML
RUN apt-get install libsfml-dev -y
