#!/bin/bash
cd docker
$CONTAINER_ENGINE pull tensorflow/tensorflow:latest
$CONTAINER_ENGINE build --build-arg BASE_IMAGE_TAG=latest --no-cache -t jhale/tfp:latest .
cd ../
