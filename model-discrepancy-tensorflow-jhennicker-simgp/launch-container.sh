#!/bin/bash
$CONTAINER_ENGINE run -ti -v $(pwd):/shared -w /shared jhale/tfp:latest-gpu
