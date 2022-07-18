#!bin/bash

#apt-get -y update
#apt-get -y install --reinstall build-essential
#apt-get -y install git python3-venv python3-dev libhdf5-dev gcc-aarch64-linux-gnu
#cd ~
#rm -rf tf_venv
rm -rf ~/src
umount -f -l /mnt/datasets
#mkdir tf_venv
#cd tf_venv
#python3 -m venv .
#source bin/activate
#python -m pip install --upgrade pip wheel
#python -m pip install h5py
#python -m pip install cython
#python -m pip install google protobuf
#python -m pip install --no-binary pycocotools pycocotools
#python -m pip install absl-py pillow
#python -m pip install --extra-index-url https://snapshots.linaro.org/ldcg/python-cache/ numpy==1.19.5
#python -m pip install --extra-index-url https://snapshots.linaro.org/ldcg/python-cache/ matplotlib
#python -m pip install ck
ck pull repo:ck-env
#python -m pip install scikit-build
#python -m pip install --extra-index-url https://snapshots.linaro.org/ldcg/python-cache/ tensorflow-io-gcs-filesystem==0.21.0 h5py==3.1.0
#python -m pip install tensorflow-aarch64==2.7.0
mkdir ~/src
cd ~/src
git clone https://github.com/mlcommons/inference.git
cd inference
git checkout r1.1
git cherry-pick -n 215c057fc6690a47f3f66c72c076a8f73d66cb12
git submodule update --init --recursive
cd loadgen
python setup.py develop
cd ../vision/classification_and_detection/
python setup.py develop

mkdir /mnt/datasets
mount -t nfs 10.40.96.10:/mnt/nvme /mnt/datasets
#cd ~
#export DATA_DIR=/root/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
#export MODEL_DIR=/root/models
#cd ~/src/inference/vision/classification_and_detection
#./run_local.sh tf resnet50
