class TensorflowObjectDetection:
    timeout = 1800

    def track_object_detection_CenterNet_HourGlass104_Keypoints_512x512(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/root/asv-test')

        from functs import load_image_into_numpy_array
        import os
        import pathlib

        import matplotlib
        import matplotlib.pyplot as plt

        import io
        import scipy.misc
        import numpy as np
        from six import BytesIO
        from PIL import Image, ImageDraw, ImageFont
        from six.moves.urllib.request import urlopen

        import tensorflow as tf
        import tensorflow_hub as hub
        ##########################
        import time
        import csv
        import sys
        
        inference = run_object_detect_bench(self, model,lib, inter_list, intra_list, batch_size)
        return inference

    track_object_detection_CenterNet_HourGlass104_Keypoints_512x512.params = (['CenterNet HourGlass104 Keypoints 512x512'], ['tp', 'eigen'], [16], [16], [1,16,32])

    track_object_detection_CenterNet_HourGlass104_Keypoints_512x512.unit = "Inference Time"
    
    
    def track_object_detection_CenterNet_HourGlass104_1024x1024(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/root/asv-test')

        from functs import load_image_into_numpy_array
        import os
        import pathlib

        import matplotlib
        import matplotlib.pyplot as plt

        import io
        import scipy.misc
        import numpy as np
        from six import BytesIO
        from PIL import Image, ImageDraw, ImageFont
        from six.moves.urllib.request import urlopen

        import tensorflow as tf
        import tensorflow_hub as hub
        ##########################
        import time
        import csv
        import sys
        
        inference = run_object_detect_bench(self, model,lib, inter_list, intra_list, batch_size)
        return inference

    track_object_detection_CenterNet_HourGlass104_1024x1024.params = (['CenterNet HourGlass104 1024x1024'], ['tp', 'eigen'], [16], [16], [1,16,32])

    track_object_detection_CenterNet_HourGlass104_1024x1024.unit = "Inference Time"
    
