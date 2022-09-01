#!/bin/bash

export DATA_DIR=/home/ubuntu/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
export MODEL_DIR=/home/ubuntu/models
cd ~/sources/inference/vision/classification_and_detection
./run_local.sh tf resnet50
