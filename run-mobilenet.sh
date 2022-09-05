#!/bin/bash

export DATA_DIR=/mnt/datasets/data/CK-TOOLS/dataset-coco-2017-val
export MODEL_DIR=/mnt/datasets/data/models
cd /root/asv-test/inference/vision/classification_and_detection
./run_local.sh tf ssd-mobilenet
