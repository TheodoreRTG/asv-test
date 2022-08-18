#!/bin/bash

export DATA_DIR=/home/ubuntu/CK-TOOLS/dataset-coco-2017-val
export MODEL_DIR=/home/ubuntu/models
cd ~/sources/inference/vision/classification_and_detection
./run_local.sh tf ssd-mobilenet
