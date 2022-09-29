class TensorflowImageClassification:
    timeout = 1800
    
    def track_inception_v3(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/root/asv-test')
        import tensorflow as tf
        from functs import run_image_bench
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
        return inference

    track_inception_v3.params = (["inception_v3"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_inception_v3.unit = "Inference Time"


    def track_nasnet_mobile(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/root/asv-test')
        import tensorflow as tf
        from functs import run_image_bench
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
        return inference

    track_nasnet_mobile.params = (["nasnet_mobile"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_nasnet_mobile.unit = "Inference Time"
