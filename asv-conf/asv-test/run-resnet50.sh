#!/bin/bash

export DATA_DIR=/mnt/datasets/data/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
export MODEL_DIR=/mnt/datasets/data/models
cd ~/src/inference/vision/classification_and_detection/
./run_local.sh tf resnet50
