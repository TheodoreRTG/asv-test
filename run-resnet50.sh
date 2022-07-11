#!/bin/bash

export DATA_DIR=/root/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
export MODEL_DIR=/root/models
cd ~
./src/inference/vision/classification_and_detection/run_local.sh tf resnet50
