#!/bin/bash
podman run --annotation nvidia=true -e NVIDIA_VISIBLE_DEVICES=all -ti -v $(pwd):/shared -w /shared jhale/tfp:latest-gpu
