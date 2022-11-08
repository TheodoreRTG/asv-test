class TensorflowImageClassification:
    timeout = 1800
    def setup_cache(self):
        accuracy = {"inception_v3":0,}
        return accuracy

    def track_inception_v3(self, accuracy, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/root/asv-test')
        import tensorflow as tf
        from functs import run_image_bench, run_image_bench_accuracy
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
        
        inference = run_image_bench(self, model, lib, inter_list, intra_list, batch_size)
        if batch_size == 1:
            acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
            accuracyfin = dict()
            accuracyfin["inception_v3"] = acc
            accuracy = accuracyfin
            print(accuracyfin)
            print(accuracy)
            #setattr(self, 'accuracy["inception_v3"]', acc)
            #print(self.accuracy["inception_v3"])
        return inference, acc

    track_inception_v3.params = (["inception_v3"], ["tp"], [16], [16], [1])
    track_inception_v3.unit = "Inference Time, Accuracy"

    def track_xaccuracy_inception_v3(self, accuracy):
        #print(self.accuracy)
        acc_result = accuracy["inception_v3"]
        return acc_result
    
    track_xaccuracy_inception_v3.unit = "Accuracy"
