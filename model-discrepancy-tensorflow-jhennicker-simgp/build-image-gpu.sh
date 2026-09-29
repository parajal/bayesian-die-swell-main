#!/bin/bash
CONTAINER_ENGINE="podman"
cd docker
$CONTAINER_ENGINE pull tensorflow/tensorflow:latest-gpu
$CONTAINER_ENGINE build --build-arg BASE_IMAGE_TAG=latest-gpu --no-cache -t jhale/tfp:latest-gpu .
cd ../
