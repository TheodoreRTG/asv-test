class TensorflowImageClassification:
    timeout = 1800
    def setup(model, lib, inter_list, intra_list, batch_size):
        import os

        import time
        import csv
        ##########
        import tensorflow as tf
        import tensorflow_hub as hub

        import requests
        from PIL import Image
        from io import BytesIO

        import matplotlib.pyplot as plt
        import numpy as np
        import sys
                
        if model == "tp":
          os.environ['TF_ENABLE_ONEDNN_OPTS'] = "1"
        else:
          os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
        os.environ['model'] = model
        os.environ['lib'] = lib
        os.environ['inter'] = inter_list
        os.environ['intra'] = intra_list
        os.environ['batch'] = batch_size
        
    def track_image_classification(model, lib, inter_list, intra_list, batch_size):
        import subprocess
        import re
        import os
        #r = open('tf-image.txt', 'w')
        output = subprocess.run(['python3 2image_classification_with_tf_hub-itr-setthreads.py $model $lib $inter $intra $batch'], stdout=r, capture_output=True)
        #r.close()
        return output
    track_image_classification.params = (["efficientnetv2-s",  "efficientnetv2-m"], ['tp', 'eigen'], "16", "16", ['1', '16', '32'])
    track_QPS_mobilenet.unit = "Inference Time"
