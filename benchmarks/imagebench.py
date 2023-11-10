class TensorflowImageClassification:
    timeout = 1800
    
    def track_inception_v3(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
        import tensorflow as tf
#        from functs import run_image_bench, run_image_bench_accuracy
        from . import functs
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
    track_inception_v3.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_nasnet_mobile(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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

        return inference

    track_nasnet_mobile.params = (["nasnet_mobile"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_nasnet_mobile.unit = "Inference Time"
    track_nasnet_mobile.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    def track_efficientnetv2_s(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_s.params = (["efficientnetv2-s"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_s.unit = "Inference Time"
    track_efficientnetv2_s.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    def track_efficientnetv2_m(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_m.params = (["efficientnetv2-m"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_m.unit = "Inference Time"
    track_efficientnetv2_m.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    def track_efficientnetv2_l(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_l.params = (["efficientnetv2_l"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_l.unit = "Inference Time"
    track_efficientnetv2_l.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
     
    def track_efficientnetv2_s_21k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_s_21k.params = (["efficientnetv2-s-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_s_21k.unit = "Inference Time"
    track_efficientnetv2_s_21k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    def track_efficientnetv2_m_21k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_m_21k.params = (["efficientnetv2-m-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_m_21k.unit = "Inference Time"
    track_efficientnetv2_m_21k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_efficientnetv2_l_21k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_l_21k.params = (["efficientnetv2-l-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_l_21k.unit = "Inference Time"
    track_efficientnetv2_l_21k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_efficientnetv2_xl_21k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_xl_21k.params = (["efficientnetv2-xl-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_xl_21k.unit = "Inference Time"
    track_efficientnetv2_xl_21k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
       
    def track_efficientnetv2_b0_21k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b0_21k.params = (["efficientnetv2-b0-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b0_21k.unit = "Inference Time"
    track_efficientnetv2_b0_21k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    def track_efficientnetv2_b1_21k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b1_21k.params = (["efficientnetv2-b1-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b1_21k.unit = "Inference Time"
    track_efficientnetv2_b1_21k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
     
    def track_efficientnetv2_b2_21k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b2_21k.params = (["efficientnetv2-b2-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b2_21k.unit = "Inference Time"
    track_efficientnetv2_b2_21k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
  
    def track_efficientnetv2_b3_21k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b3_21k.params = (["efficientnetv2-b3-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b3_21k.unit = "Inference Time"
    track_efficientnetv2_b3_21k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
   
    def track_efficientnetv2_s_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_s_21k_ft1k.params = (["efficientnetv2-s-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_s_21k_ft1k.unit = "Inference Time"
    track_efficientnetv2_s_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_efficientnetv2_m_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_m_21k_ft1k.params = (["efficientnetv2-m-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_m_21k_ft1k.unit = "Inference Time"
    track_efficientnetv2_m_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
     
    def track_efficientnetv2_l_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_l_21k_ft1k.params = (["efficientnetv2-l-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_l_21k_ft1k.unit = "Inference Time"
    track_efficientnetv2_l_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
  
    def track_efficientnetv2_xl_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_xl_21k_ft1k.params = (["efficientnetv2-xl-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_xl_21k_ft1k.unit = "Inference Time"
    track_efficientnetv2_xl_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
   
    def track_efficientnetv2_b0_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b0_21k_ft1k.params = (["efficientnetv2-b0-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b0_21k_ft1k.unit = "Inference Time"
    track_efficientnetv2_b0_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
  
    def track_efficientnetv2_b1_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b1_21k_ft1k.params = (["efficientnetv2-b1-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b1_21k_ft1k.unit = "Inference Time"
    track_efficientnetv2_b1_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_efficientnetv2_b2_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b2_21k_ft1k.params = (["efficientnetv2-b2-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b2_21k_ft1k.unit = "Inference Time"
    track_efficientnetv2_b2_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_efficientnetv2_b3_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b3_21k_ft1k.params = (["efficientnetv2-b3-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b3_21k_ft1k.unit = "Inference Time"
    track_efficientnetv2_b3_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
          
    def track_efficientnetv2_b0(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b0.params = (["efficientnetv2-b0"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b0.unit = "Inference Time"
    track_efficientnetv2_b0.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
   
    def track_efficientnetv2_b1(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b1.params = (["efficientnetv2-b1"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b1.unit = "Inference Time"
    track_efficientnetv2_b1.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
 
    def track_efficientnetv2_b2(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnetv2_b2.params = (["efficientnetv2-b2"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b2.unit = "Inference Time"
    track_efficientnetv2_b2.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
   
    def track_efficientnetv2_b3(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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

        return inference

    track_efficientnetv2_b3.params = (["efficientnetv2-b3"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b3.unit = "Inference Time"
    track_efficientnetv2_b3.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
  
    def track_efficientnet_b0(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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

        return inference

    track_efficientnet_b0.params = (["efficientnet_b0"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b0.unit = "Inference Time"
    track_efficientnet_b0.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_efficientnet_b1(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnet_b1.params = (["efficientnet_b1"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b1.unit = "Inference Time"
    track_efficientnet_b1.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_efficientnet_b2(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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

        return inference

    track_efficientnet_b2.params = (["efficientnet_b2"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b2.unit = "Inference Time"
    track_efficientnet_b2.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_efficientnet_b3(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnet_b3.params = (["efficientnet_b3"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b3.unit = "Inference Time"
    track_efficientnet_b3.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_efficientnet_b4(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnet_b4.params = (["efficientnet_b4"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b4.unit = "Inference Time"
    track_efficientnet_b4.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    def track_efficientnet_b5(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnet_b5.params = (["efficientnet_b5"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b5.unit = "Inference Time"
    track_efficientnet_b5.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
     
    def track_efficientnet_b6(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_efficientnet_b6.params = (["efficientnet_b6"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b6.unit = "Inference Time"
    track_efficientnet_b6.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_efficientnet_b7(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference
      
    track_efficientnet_b7.params = (["efficientnet_b7"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b7.unit = "Inference Time"
    track_efficientnet_b7.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
        
    def track_bit_s_r50x1(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_bit_s_r50x1.params = (["bit_s-r50x1"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_bit_s_r50x1.unit = "Inference Time"
    track_bit_s_r50x1.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_inception_resnet_v2(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_inception_resnet_v2.params = (["inception_resnet_v2"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_inception_resnet_v2.unit = "Inference Time"
    track_inception_resnet_v2.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    def track_resnet_v1_50(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_resnet_v1_50.params = (["resnet_v1_50"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v1_50.unit = "Inference Time"
    track_resnet_v1_50.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    def track_resnet_v1_101(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_resnet_v1_101.params = (["resnet_v1_101"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v1_101.unit = "Inference Time"
    track_resnet_v1_101.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    def track_resnet_v1_152(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_resnet_v1_152.params = (["resnet_v1_152"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v1_152.unit = "Inference Time"
    track_resnet_v1_152.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
        
    def track_resnet_v2_50(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_resnet_v2_50.params = (["resnet_v2_50"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v2_50.unit = "Inference Time"
    track_resnet_v2_50.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
          
    def track_resnet_v2_101(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_resnet_v2_101.params = (["resnet_v2_101"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v2_101.unit = "Inference Time"
    track_resnet_v2_101.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
      
    def track_resnet_v2_152(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_resnet_v2_152.params = (["resnet_v2_152"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v2_152.unit = "Inference Time"
    track_resnet_v2_152.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_nasnet_large(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_nasnet_large.params = (["nasnet_large"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_nasnet_large.unit = "Inference Time"
    track_nasnet_large.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    def track_pnasnet_large(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_pnasnet_large.params = (["pnasnet_large"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_pnasnet_large.unit = "Inference Time"
    track_pnasnet_large.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
     
    def track_mobilenet_v2_100_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_mobilenet_v2_100_224.params = (["mobilenet_v2_100_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v2_100_224.unit = "Inference Time"
    track_mobilenet_v2_100_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
 
    def track_mobilenet_v2_130_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_mobilenet_v2_130_224.params = (["mobilenet_v2_130_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v2_130_224.unit = "Inference Time"
    track_mobilenet_v2_130_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
     
    def track_mobilenet_v2_140_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_mobilenet_v2_140_224.params = (["mobilenet_v2_140_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v2_140_224.unit = "Inference Time"
    track_mobilenet_v2_140_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_mobilenet_v3_small_100_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_mobilenet_v3_small_100_224.params = (["mobilenet_v3_small_100_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v3_small_100_224.unit = "Inference Time"
    track_mobilenet_v3_small_100_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_mobilenet_v3_small_075_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_mobilenet_v3_small_075_224.params = (["mobilenet_v3_small_075_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v3_small_075_224.unit = "Inference Time"
    track_mobilenet_v3_small_075_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_mobilenet_v3_large_100_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_mobilenet_v3_large_100_224.params = (["mobilenet_v3_large_100_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v3_large_100_224.unit = "Inference Time"
    track_mobilenet_v3_large_100_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
           
    def track_mobilenet_v3_large_075_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        return inference

    track_mobilenet_v3_large_075_224.params = (["mobilenet_v3_large_075_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v3_large_075_224.unit = "Inference Time"
    track_mobilenet_v3_large_075_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])

    
class TensorflowAccuracyImageClassification:
    
    def track_accuracy_inception_v3(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_inception_v3.params = (["inception_v3"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_inception_v3.unit = "Accuracy"
    track_accuracy_inception_v3.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
   
    
    def track_accuracy_nasnet_mobile(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_nasnet_mobile.params = (["nasnet_mobile"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_nasnet_mobile.unit = "Accuracy"
    track_accuracy_nasnet_mobile.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_accuracy_efficientnetv2_l(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_l.params = (["efficientnetv2_l"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_l.unit = "Accuracy"
    track_accuracy_efficientnetv2_l.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
     
    def track_accuracy_efficientnetv2_l_21k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_l_21k.params = (["efficientnetv2-l-21k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_l_21k.unit = "Accuracy"
    track_accuracy_efficientnetv2_l_21k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_accuracy_efficientnetv2_l_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_l_21k_ft1k.params = (["efficientnetv2-l-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_l_21k_ft1k.unit = "Accuracy"
    track_accuracy_efficientnetv2_l_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
    def track_accuracy_efficientnetv2_b0_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b0_21k_ft1k.params = (["efficientnetv2-b0-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b0_21k_ft1k.unit = "Accuracy"
    track_accuracy_efficientnetv2_b0_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnetv2_b1_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b1_21k_ft1k.params = (["efficientnetv2-b1-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b1_21k_ft1k.unit = "Accuracy"
    track_accuracy_efficientnetv2_b1_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnetv2_b2_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b2_21k_ft1k.params = (["efficientnetv2-b2-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b2_21k_ft1k.unit = "Accuracy"
    track_accuracy_efficientnetv2_b2_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
           
    def track_accuracy_efficientnetv2_b3_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b3_21k_ft1k.params = (["efficientnetv2-b3-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b3_21k_ft1k.unit = "Accuracy"
    track_accuracy_efficientnetv2_b3_21k_ft1k.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
           
    def track_accuracy_efficientnetv2_b0(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b0.params = (["efficientnetv2-b0"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b0.unit = "Accuracy"
    track_accuracy_efficientnetv2_b0.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
    
           
    def track_accuracy_efficientnetv2_b1(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b1.params = (["efficientnetv2-b1"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b1.unit = "Accuracy"
    track_accuracy_efficientnetv2_b1.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnetv2_b2(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b2.params = (["efficientnetv2-b2"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b2.unit = "Accuracy"
    track_accuracy_efficientnetv2_b2.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnetv2_b3(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b3.params = (["efficientnetv2-b3"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b3.unit = "Accuracy"
    track_accuracy_efficientnetv2_b3.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnet_b0(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b0.params = (["efficientnet_b0"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b0.unit = "Accuracy"
    track_accuracy_efficientnet_b0.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnet_b1(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b1.params = (["efficientnet_b1"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b1.unit = "Accuracy"
    track_accuracy_efficientnet_b1.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnet_b2(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b2.params = (["efficientnet_b2"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b2.unit = "Accuracy"
    track_accuracy_efficientnet_b2.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnet_b3(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b3.params = (["efficientnet_b3"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b3.unit = "Accuracy"
    track_accuracy_efficientnet_b3.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnet_b4(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b4.params = (["efficientnet_b4"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b4.unit = "Accuracy"
    track_accuracy_efficientnet_b4.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnet_b5(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc
    track_accuracy_efficientnet_b5.params = (["efficientnet_b5"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b5.unit = "Accuracy"
    track_accuracy_efficientnet_b5.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnet_b6(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b6.params = (["efficientnet_b6"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b6.unit = "Accuracy"
    track_accuracy_efficientnet_b6.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_efficientnet_b7(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b7.params = (["efficientnet_b7"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b7.unit = "Accuracy"
    track_accuracy_efficientnet_b7.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_bit_s_r50x1(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_bit_s_r50x1.params = (["bit_s-r50x1"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_bit_s_r50x1.unit = "Accuracy"
    track_accuracy_bit_s_r50x1.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_inception_resnet_v2(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_inception_resnet_v2.params = (["inception_resnet_v2"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_inception_resnet_v2.unit = "Accuracy"
    track_accuracy_inception_resnet_v2.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_resnet_v1_50(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v1_50.params = (["resnet_v1_50"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v1_50.unit = "Accuracy"
    track_accuracy_resnet_v1_50.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_resnet_v1_101(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v1_101.params = (["resnet_v1_101"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v1_101.unit = "Accuracy"
    track_accuracy_resnet_v1_101.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_resnet_v1_152(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v1_152.params = (["resnet_v1_152"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v1_152.unit = "Accuracy"
    track_accuracy_resnet_v1_152.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_resnet_v2_50(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v2_50.params = (["resnet_v2_50"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v2_50.unit = "Accuracy"
    track_accuracy_resnet_v2_50.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_resnet_v2_101(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v2_101.params = (["resnet_v2_101"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v2_101.unit = "Accuracy"
    track_accuracy_resnet_v2_101.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_resnet_v2_152(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v2_152.params = (["resnet_v2_152"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v2_152.unit = "Accuracy"
    track_accuracy_resnet_v2_152.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_nasnet_large(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_nasnet_large.params = (["nasnet_large"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_nasnet_large.unit = "Accuracy"
    track_accuracy_nasnet_large.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_pnasnet_large(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_pnasnet_large.params = (["pnasnet_large"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_pnasnet_large.unit = "Accuracy"
    track_accuracy_pnasnet_large.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_mobilenet_v2_100_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v2_100_224.params = (["mobilenet_v2_100_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v2_100_224.unit = "Accuracy"
    track_accuracy_mobilenet_v2_100_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_mobilenet_v2_130_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v2_130_224.params = (["mobilenet_v2_130_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v2_130_224.unit = "Accuracy"
    track_accuracy_mobilenet_v2_130_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_mobilenet_v2_140_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v2_140_224.params = (["mobilenet_v2_140_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v2_140_224.unit = "Accuracy"
    track_accuracy_mobilenet_v2_140_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_mobilenet_v3_large_100_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v3_large_100_224.params = (["mobilenet_v3_large_100_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v3_large_100_224.unit = "Accuracy"
    track_accuracy_mobilenet_v3_large_100_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
           
    def track_accuracy_mobilenet_v3_large_075_224(self, model, lib, inter_list, intra_list, batch_size):
        import sys
        sys.path.append('/home/buildslave/workspace/asv-test')
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v3_large_075_224.params = (["mobilenet_v3_large_075_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v3_large_075_224.unit = "Accuracy"
    track_accuracy_mobilenet_v3_large_075_224.param_names = (["Model"], ["Library"], ["inter threads"], ["intra threads"], ["Batch Size"])
        
