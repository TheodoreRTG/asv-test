class TensorflowImageClassification:
    timeout = 1800
    
    def track_accuracy_inception_v3(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_inception_v3.params = (["inception_v3"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_inception_v3.unit = "Accuracy"
   
    
    def track_accuracy_nasnet_mobile(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_nasnet_mobile.params = (["nasnet_mobile"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_nasnet_mobile.unit = "Accuracy"
        
    def track_accuracy_efficientnetv2_s(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_s.params = (["efficientnetv2-s"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_s.unit = "Accuracy"
    
    
    def track_accuracy_efficientnetv2_m(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_m.params = (["efficientnetv2-m"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_m.unit = "Accuracy"  
    
    def track_accuracy_efficientnetv2_l(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_l.params = (["efficientnetv2_l"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_l.unit = "Accuracy"
     
    
    def track_accuracy_efficientnetv2_s_21k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_s_21k.params = (["efficientnetv2-s-21k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_s_21k.unit = "Accuracy"
    
       
    def track_accuracy_efficientnetv2_m_21k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_m_21k.params = (["efficientnetv2-m-21k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_m_21k.unit = "Accuracy"
    
           
    def track_accuracy_efficientnetv2_l_21k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_l_21k.params = (["efficientnetv2-l-21k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_l_21k.unit = "Accuracy"
    
           
    def track_accuracy_efficientnetv2_xl_21k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_xl_21k.params = (["efficientnetv2-xl-21k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_xl_21k.unit = "Accuracy"
    
           
    def track_accuracy_efficientnetv2_b0_21k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b0_21k.params = (["efficientnetv2-b0-21k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b0_21k.unit = "Accuracy"
    
           
    def track_accuracy_efficientnetv2_b1_21k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b1_21k.params = (["efficientnetv2-b1-21k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b1_21k.unit = "Accuracy"
           
    def track_accuracy_efficientnetv2_b2_21k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b2_21k.params = (["efficientnetv2-b2-21k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b2_21k.unit = "Accuracy"
    
           
    def track_accuracy_efficientnetv2_b3_21k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b3_21k.params = (["efficientnetv2-b3-21k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b3_21k.unit = "Accuracy"
           
    def track_accuracy_efficientnetv2_s_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_s_21k_ft1k.params = (["efficientnetv2-s-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_s_21k_ft1k.unit = "Accuracy"
           
    def track_accuracy_efficientnetv2_m_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_m_21k_ft1k.params = (["efficientnetv2-m-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_m_21k_ft1k.unit = "Accuracy"
    
    def track_accuracy_efficientnetv2_l_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_l_21k_ft1k.params = (["efficientnetv2-l-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_l_21k_ft1k.unit = "Accuracy"
           
    def track_accuracy_efficientnetv2_xl_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_xl_21k_ft1k.params = (["efficientnetv2-xl-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_xl_21k_ft1k.unit = "Accuracy"
    
           
    def track_accuracy_efficientnetv2_b0_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b0_21k_ft1k.params = (["efficientnetv2-b0-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b0_21k_ft1k.unit = "Accuracy"
           
    def track_accuracy_efficientnetv2_b1_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b1_21k_ft1k.params = (["efficientnetv2-b1-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b1_21k_ft1k.unit = "Accuracy"
           
    def track_accuracy_efficientnetv2_b2_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b2_21k_ft1k.params = (["efficientnetv2-b2-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b2_21k_ft1k.unit = "Accuracy"
    
           
    def track_accuracy_efficientnetv2_b3_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b3_21k_ft1k.params = (["efficientnetv2-b3-21k-ft1k"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b3_21k_ft1k.unit = "Accuracy"
    
           
    def track_accuracy_efficientnetv2_b0(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b0.params = (["efficientnetv2-b0"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b0.unit = "Accuracy"
    
           
    def track_accuracy_efficientnetv2_b1(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b1.params = (["efficientnetv2-b1"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b1.unit = "Accuracy"
           
    def track_accuracy_efficientnetv2_b2(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b2.params = (["efficientnetv2-b2"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b2.unit = "Accuracy"
           
    def track_accuracy_efficientnetv2_b3(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnetv2_b3.params = (["efficientnetv2-b3"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnetv2_b3.unit = "Accuracy"
           
    def track_accuracy_efficientnet_b0(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b0.params = (["efficientnet_b0"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b0.unit = "Accuracy"
           
    def track_accuracy_efficientnet_b1(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b1.params = (["efficientnet_b1"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b1.unit = "Accuracy"
           
    def track_accuracy_efficientnet_b2(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b2.params = (["efficientnet_b2"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b2.unit = "Accuracy"
           
    def track_accuracy_efficientnet_b3(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b3.params = (["efficientnet_b3"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b3.unit = "Accuracy"
           
    def track_accuracy_efficientnet_b4(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b4.params = (["efficientnet_b4"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b4.unit = "Accuracy"
           
    def track_accuracy_efficientnet_b5(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc
    track_accuracy_efficientnet_b5.params = (["efficientnet_b5"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b5.unit = "Accuracy"
           
    def track_accuracy_efficientnet_b6(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b6.params = (["efficientnet_b6"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b6.unit = "Accuracy"
           
    def track_accuracy_efficientnet_b7(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_efficientnet_b7.params = (["efficientnet_b7"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_efficientnet_b7.unit = "Accuracy"
           
    def track_accuracy_bit_s_r50x1(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_bit_s_r50x1.params = (["bit_s-r50x1"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_bit_s_r50x1.unit = "Accuracy"
           
    def track_accuracy_inception_resnet_v2(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_inception_resnet_v2.params = (["inception_resnet_v2"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_inception_resnet_v2.unit = "Accuracy"
           
    def track_accuracy_resnet_v1_50(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v1_50.params = (["resnet_v1_50"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v1_50.unit = "Accuracy"
           
    def track_accuracy_resnet_v1_101(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v1_101.params = (["resnet_v1_101"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v1_101.unit = "Accuracy"
           
    def track_accuracy_resnet_v1_152(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v1_152.params = (["resnet_v1_152"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v1_152.unit = "Accuracy"
           
    def track_accuracy_resnet_v2_50(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v2_50.params = (["resnet_v2_50"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v2_50.unit = "Accuracy"
           
    def track_accuracy_resnet_v2_101(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v2_101.params = (["resnet_v2_101"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v2_101.unit = "Accuracy"
           
    def track_accuracy_resnet_v2_152(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_resnet_v2_152.params = (["resnet_v2_152"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_resnet_v2_152.unit = "Accuracy"
           
    def track_accuracy_nasnet_large(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_nasnet_large.params = (["nasnet_large"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_nasnet_large.unit = "Accuracy"
           
    def track_accuracy_pnasnet_large(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_pnasnet_large.params = (["pnasnet_large"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_pnasnet_large.unit = "Accuracy"
           
    def track_accuracy_mobilenet_v2_100_224(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v2_100_224.params = (["mobilenet_v2_100_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v2_100_224.unit = "Accuracy"
           
    def track_accuracy_mobilenet_v2_130_224(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v2_130_224.params = (["mobilenet_v2_130_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v2_130_224.unit = "Accuracy"
           
    def track_accuracy_mobilenet_v2_140_224(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v2_140_224.params = (["mobilenet_v2_140_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v2_140_224.unit = "Accuracy"
           
    def track_accuracy_mobilenet_v3_small_100_224(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v3_small_100_224.params = (["mobilenet_v3_small_100_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v3_small_100_224.unit = "Accuracy"
           
    def track_accuracy_mobilenet_v3_small_075_224(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v3_small_075_224.params = (["mobilenet_v3_small_075_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v3_small_075_224.unit = "Accuracy"
           
    def track_accuracy_mobilenet_v3_large_100_224(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v3_large_100_224.params = (["mobilenet_v3_large_100_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v3_large_100_224.unit = "Accuracy"
           
    def track_accuracy_mobilenet_v3_large_075_224(self, model, lib, inter_list, intra_list, batch_size):
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
        
        acc = run_image_bench_accuracy(self, model, lib, inter_list, intra_list, batch_size)
        return acc

    track_accuracy_mobilenet_v3_large_075_224.params = (["mobilenet_v3_large_075_224"], ["tp", "eigen"], [16], [16], [1])
    track_accuracy_mobilenet_v3_large_075_224.unit = "Accuracy"
        
    def track_inception_v3(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_inception_v3.params = (["inception_v3"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_inception_v3.unit = "Inference Time"
    
    def track_accuracy_inception_v3(self, accuracy):
        acc_result = accuracy["inception_v3"]
        return acc_result
    
    track_accuracy_inception_v3.unit = "Accuracy"
    
    def track_nasnet_mobile(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_nasnet_mobile.params = (["nasnet_mobile"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_nasnet_mobile.unit = "Inference Time"

    def track_accuracy_nasnet_mobile(self, accuracy):
        acc_result = accuracy["nasnet_mobile"]
        return acc_result
    
    track_accuracy_nasnet_mobile.unit = "Accuracy"
    
    def track_efficientnetv2_s(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_s.params = (["efficientnetv2-s"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_s.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_s(self, accuracy):
        acc_result = accuracy["efficientnetv2-s"]
        return acc_result
    
    track_accuracy_efficientnetv2_s.unit = "Accuracy"
    
    def track_efficientnetv2_m(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_m.params = (["efficientnetv2-m"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_m.unit = "Inference Time"  
    
    def track_accuracy_efficientnetv2_m(self, accuracy):
        acc_result = accuracy["efficientnetv2-m"]
        return acc_result
    
    track_accuracy_efficientnetv2_m.unit = "Accuracy"
    
    def track_efficientnetv2_l(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_l.params = (["efficientnetv2_l"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_l.unit = "Inference Time"
     
     def track_accuracy_efficientnetv2_l(self, accuracy):
        acc_result = accuracy["efficientnetv2-l"]
        return acc_result
    
    track_accuracy_efficientnetv2_l.unit = "Accuracy"
    
    def track_efficientnetv2_s_21k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_s_21k.params = (["efficientnetv2-s-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_s_21k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_s_21k(self, accuracy):
        acc_result = accuracy["efficientnetv2-s-21k"]
        return acc_result
    
    track_accuracy_efficientnetv2_s_21k.unit = "Accuracy"
       
    def track_efficientnetv2_m_21k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_m_21k.params = (["efficientnetv2-m-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_m_21k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_m_21k(self, accuracy):
        acc_result = accuracy["efficientnetv2-m-21k"]
        return acc_result
    
    track_accuracy_efficientnetv2_m_21k.unit = "Accuracy"
           
    def track_efficientnetv2_l_21k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_l_21k.params = (["efficientnetv2-l-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_l_21k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_l_21k(self, accuracy):
        acc_result = accuracy["efficientnetv2-l-21k"]
        return acc_result
    
    track_accuracy_efficientnetv2_l_21k.unit = "Accuracy"
           
    def track_efficientnetv2_xl_21k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_xl_21k.params = (["efficientnetv2-xl-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_xl_21k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_xl_21k(self, accuracy):
        acc_result = accuracy["efficientnetv2-xl-21k"]
        return acc_result
    
    track_accuracy_efficientnetv2_xl_21k.unit = "Accuracy"
           
    def track_efficientnetv2_b0_21k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b0_21k.params = (["efficientnetv2-b0-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b0_21k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b0_21k(self, accuracy):
        acc_result = accuracy["efficientnetv2-b0-21k"]
        return acc_result
    
    track_accuracy_efficientnetv2_b0_21k.unit = "Accuracy"
           
    def track_efficientnetv2_b1_21k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b1_21k.params = (["efficientnetv2-b1-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b1_21k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b1_21k(self, accuracy):
        acc_result = accuracy["efficientnetv2-b1-21k"]
        return acc_result
    
    track_accuracy_efficientnetv2_b1_21k.unit = "Accuracy"
           
    def track_efficientnetv2_b2_21k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b2_21k.params = (["efficientnetv2-b2-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b2_21k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b2_21k(self, accuracy):
        acc_result = accuracy["efficientnetv2-b2-21k"]
        return acc_result
    
    track_accuracy_efficientnetv2_b2_21k.unit = "Accuracy"
           
    def track_efficientnetv2_b3_21k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b3_21k.params = (["efficientnetv2-b3-21k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b3_21k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b3_21k(self, accuracy):
        acc_result = accuracy["efficientnetv2-b3-21k"]
        return acc_result
    
    track_accuracy_efficientnetv2_b3_21k.unit = "Accuracy"
           
    def track_efficientnetv2_s_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_s_21k_ft1k.params = (["efficientnetv2-s-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_s_21k_ft1k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_s_21k_ft1k(self, accuracy):
        acc_result = accuracy["efficientnetv2-s-21k-ft1k"]
        return acc_result
    
    track_accuracy_efficientnetv2_s_21k_ft1k.unit = "Accuracy"
           
    def track_efficientnetv2_m_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_m_21k_ft1k.params = (["efficientnetv2-m-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_m_21k_ft1k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_m_21k_ft1k(self, accuracy):
        acc_result = accuracy["efficientnetv2-m-21k-ft1k"]
        return acc_result
    
    track_accuracy_efficientnetv2_m_21k_ft1k.unit = "Accuracy"
           
    def track_efficientnetv2_l_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_l_21k_ft1k.params = (["efficientnetv2-l-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_l_21k_ft1k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_l_21k_ft1k(self, accuracy):
        acc_result = accuracy["efficientnetv2-l-21k-ft1k"]
        return acc_result
    
    track_accuracy_efficientnetv2_l_21k_ft1k.unit = "Accuracy"
           
    def track_efficientnetv2_xl_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_xl_21k_ft1k.params = (["efficientnetv2-xl-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_xl_21k_ft1k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_xl_21k_ft1k(self, accuracy):
        acc_result = accuracy["efficientnetv2-xl-21k-ft1k"]
        return acc_result
    
    track_accuracy_efficientnetv2_xl_21k_ft1k.unit = "Accuracy"
           
    def track_efficientnetv2_b0_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b0_21k_ft1k.params = (["efficientnetv2-b0-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b0_21k_ft1k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b0_21k_ft1k(self, accuracy):
        acc_result = accuracy["efficientnetv2-b0-21k-ft1k"]
        return acc_result
    
    track_accuracy_efficientnetv2_b0_21k_ft1k.unit = "Accuracy"
           
    def track_efficientnetv2_b1_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b1_21k_ft1k.params = (["efficientnetv2-b1-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b1_21k_ft1k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b1_21k_ft1k(self, accuracy):
        acc_result = accuracy["efficientnetv2-b1-21k-ft1k"]
        return acc_result
    
    track_accuracy_efficientnetv2_b1_21k_ft1k.unit = "Accuracy"
           
    def track_efficientnetv2_b2_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b2_21k_ft1k.params = (["efficientnetv2-b2-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b2_21k_ft1k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b2_21k_ft1k(self, accuracy):
        acc_result = accuracy["efficientnetv2-b2-21k-ft1k"]
        return acc_result
    
    track_accuracy_efficientnetv2_b2_21k_ft1k.unit = "Accuracy"
           
    def track_efficientnetv2_b3_21k_ft1k(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b3_21k_ft1k.params = (["efficientnetv2-b3-21k-ft1k"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b3_21k_ft1k.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b3_21k_ft1k(self, accuracy):
        acc_result = accuracy["efficientnetv2-b3-21k-ft1k"]
        return acc_result
    
    track_accuracy_efficientnetv2_b3_21k_ft1k.unit = "Accuracy"
           
    def track_efficientnetv2_b0(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b0.params = (["efficientnetv2-b0"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b0.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b0(self, accuracy):
        acc_result = accuracy["efficientnetv2-b0"]
        return acc_result
    
    track_accuracy_efficientnetv2_b0.unit = "Accuracy"
           
    def track_efficientnetv2_b1(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b1.params = (["efficientnetv2-b1"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b1.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b1(self, accuracy):
        acc_result = accuracy["efficientnetv2-b1"]
        return acc_result
    
    track_accuracy_efficientnetv2_b1.unit = "Accuracy"
           
    def track_efficientnetv2_b2(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b2.params = (["efficientnetv2-b2"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b2.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b(self, accuracy):
        acc_result = accuracy["efficientnetv2-b2"]
        return acc_result
    
    track_accuracy_efficientnetv2_b.unit = "Accuracy"
           
    def track_efficientnetv2_b3(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnetv2_b3.params = (["efficientnetv2-b3"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnetv2_b3.unit = "Inference Time"
    
    def track_accuracy_efficientnetv2_b3(self, accuracy):
        acc_result = accuracy["efficientnetv2-b3"]
        return acc_result
    
    track_accuracy_efficientnetv2_b3.unit = "Accuracy"
           
    def track_efficientnet_b0(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnet_b0.params = (["efficientnet_b0"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b0.unit = "Inference Time"
    
    def track_accuracy_efficientnet_b0(self, accuracy):
        acc_result = accuracy["efficientnet_b0"]
        return acc_result
    
    track_accuracy_efficientnet_b0.unit = "Accuracy"
           
    def track_efficientnet_b1(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnet_b1.params = (["efficientnet_b1"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b1.unit = "Inference Time"
    
    def track_accuracy_efficientnet_b1(self, accuracy):
        acc_result = accuracy[["efficientnet_b1"]
        return acc_result
    
    track_accuracy_efficientnet_b1.unit = "Accuracy"
           
    def track_efficientnet_b2(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnet_b2.params = (["efficientnet_b2"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b2.unit = "Inference Time"
    
    def track_accuracy_efficientnet_b2(self, accuracy):
        acc_result = accuracy["efficientnet_b2"]
        return acc_result
    
    track_accuracy_efficientnet_b2.unit = "Accuracy"
           
    def track_efficientnet_b3(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnet_b3.params = (["efficientnet_b3"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b3.unit = "Inference Time"
    
    def track_accuracy_efficientnet_b3(self, accuracy):
        acc_result = accuracy["efficientnet_b3"]
        return acc_result
    
    track_accuracy_efficientnet_b3.unit = "Accuracy"
           
    def track_efficientnet_b4(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnet_b4.params = (["efficientnet_b4"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b4.unit = "Inference Time"
    
    def track_accuracy_efficientnet_b4(self, accuracy):
        acc_result = accuracy["efficientnet_b4"]
        return acc_result
    
    track_accuracy_efficientnet_b4.unit = "Accuracy"
           
    def track_efficientnet_b5(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnet_b5.params = (["efficientnet_b5"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b5.unit = "Inference Time"
    
    def track_accuracy_efficientnet_b5(self, accuracy):
        acc_result = accuracy["efficientnet_b5"]
        return acc_result
    
    track_accuracy_efficientnet_b5.unit = "Accuracy"
           
    def track_efficientnet_b6(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnet_b6.params = (["efficientnet_b6"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b6.unit = "Inference Time"
    
    def track_accuracy_efficientnet_b6(self, accuracy):
        acc_result = accuracy["efficientnet_b6"]
        return acc_result
    
    track_accuracy_efficientnet_b6.unit = "Accuracy"
           
    def track_efficientnet_b7(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_efficientnet_b7.params = (["efficientnet_b7"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_efficientnet_b7.unit = "Inference Time"
    
    def track_accuracy_efficientnet_b7(self, accuracy):
        acc_result = accuracy["efficientnet_b7"]
        return acc_result
    
    track_accuracy_efficientnet_b7.unit = "Accuracy"
           
    def track_bit_s_r50x1(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_bit_s_r50x1.params = (["bit_s-r50x1"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_bit_s_r50x1.unit = "Inference Time"
    
    def track_accuracy_bit_s_r50x1(self, accuracy):
        acc_result = accuracy["bit_s-r50x1"]
        return acc_result
    
    track_accuracy_bit_s-r50x1.unit = "Accuracy"
           
    def track_inception_resnet_v2(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_inception_resnet_v2.params = (["inception_resnet_v2"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_inception_resnet_v2.unit = "Inference Time"
    
    def track_accuracy_inception_resnet_v2(self, accuracy):
        acc_result = accuracy["inception_resnet_v2"]
        return acc_result
    
    track_accuracy_inception_resnet_v2.unit = "Accuracy"
           
    def track_resnet_v1_50(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_resnet_v1_50.params = (["resnet_v1_50"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v1_50.unit = "Inference Time"
    
    def track_accuracy_resnet_v1_50(self, accuracy):
        acc_result = accuracy["resnet_v1_50"]
        return acc_result
    
    track_accuracy_resnet_v1_50.unit = "Accuracy"
           
    def track_resnet_v1_101(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_resnet_v1_101.params = (["resnet_v1_101"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v1_101.unit = "Inference Time"
    
    def track_accuracy_resnet_v1_101(self, accuracy):
        acc_result = accuracy["resnet_v1_101"]
        return acc_result
    
    track_accuracy_resnet_v1_101.unit = "Accuracy"
           
    def track_resnet_v1_152(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_resnet_v1_152.params = (["resnet_v1_152"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v1_152.unit = "Inference Time"
    
    def track_accuracy_resnet_v1_152(self, accuracy):
        acc_result = accuracy["resnet_v1_152"]
        return acc_result
    
    track_accuracy_resnet_v1_152.unit = "Accuracy"
           
    def track_resnet_v2_50(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_resnet_v2_50.params = (["resnet_v2_50"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v2_50.unit = "Inference Time"
    
    def track_accuracy_resnet_v2_50(self, accuracy):
        acc_result = accuracy["resnet_v2_50"]
        return acc_result
    
    track_accuracy_resnet_v2_50.unit = "Accuracy"
           
    def track_resnet_v2_101(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_resnet_v2_101.params = (["resnet_v2_101"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v2_101.unit = "Inference Time"
    
    def track_accuracy_resnet_v2_101(self, accuracy):
        acc_result = accuracy["resnet_v2_101"]
        return acc_result
    
    track_accuracy_resnet_v2_101.unit = "Accuracy"
           
    def track_resnet_v2_152(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_resnet_v2_152.params = (["resnet_v2_152"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_resnet_v2_152.unit = "Inference Time"
    
    def track_accuracy_resnet_v2_152(self, accuracy):
        acc_result = accuracy["resnet_v2_152"]
        return acc_result
    
    track_accuracy_resnet_v2_152.unit = "Accuracy"
           
    def track_nasnet_large(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_nasnet_large.params = (["nasnet_large"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_nasnet_large.unit = "Inference Time"
    
    def track_accuracy_nasnet_large(self, accuracy):
        acc_result = accuracy["nasnet_large"]
        return acc_result
    
    track_accuracy_nasnet_large.unit = "Accuracy"
           
    def track_pnasnet_large(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_pnasnet_large.params = (["pnasnet_large"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_pnasnet_large.unit = "Inference Time"
    
    def track_accuracy_pnasnet_large(self, accuracy):
        acc_result = accuracy["pnasnet_large"]
        return acc_result
    
    track_accuracy_pnasnet_large.unit = "Accuracy"
           
    def track_mobilenet_v2_100_224(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_mobilenet_v2_100_224.params = (["mobilenet_v2_100_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v2_100_224.unit = "Inference Time"
    
    def track_accuracy_mobilenet_v2_100_224(self, accuracy):
        acc_result = accuracy["mobilenet_v2_100_224"]
        return acc_result
    
    track_accuracy_mobilenet_v2_100_224.unit = "Accuracy"
           
    def track_mobilenet_v2_130_224(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_mobilenet_v2_130_224.params = (["mobilenet_v2_130_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v2_130_224.unit = "Inference Time"
    
    def track_accuracymobilenet_v2_130_224(self, accuracy):
        acc_result = accuracy["mobilenet_v2_130_224"]
        return acc_result
    
    track_accuracy_mobilenet_v2_130_224.unit = "Accuracy"
           
    def track_mobilenet_v2_140_224(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_mobilenet_v2_140_224.params = (["mobilenet_v2_140_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v2_140_224.unit = "Inference Time"
    
    def track_accuracy_mobilenet_v2_140_224(self, accuracy):
        acc_result = accuracy["mobilenet_v2_140_224"]
        return acc_result
    
    track_accuracy_mobilenet_v2_140_224.unit = "Accuracy"
           
    def track_mobilenet_v3_small_100_224(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_mobilenet_v3_small_100_224.params = (["mobilenet_v3_small_100_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v3_small_100_224.unit = "Inference Time"
    
    def track_accuracy_mobilenet_v3_small_100_224(self, accuracy):
        acc_result = accuracy["mobilenet_v3_small_100_224"]
        return acc_result
    
    track_accuracy_mobilenet_v3_small_100_224.unit = "Accuracy"
           
    def track_mobilenet_v3_small_075_224(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_mobilenet_v3_small_075_224.params = (["mobilenet_v3_small_075_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v3_small_075_224.unit = "Inference Time"
    
    def track_accuracy_mobilenet_v3_small_075_224(self, accuracy):
        acc_result = accuracy["mobilenet_v3_small_075_224"]
        return acc_result
    
    track_accuracy_mobilenet_v3_small_075_224.unit = "Accuracy"
           
    def track_mobilenet_v3_large_100_224(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_mobilenet_v3_large_100_224.params = (["mobilenet_v3_large_100_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v3_large_100_224.unit = "Inference Time"
    
    def track_accuracy_mobilenet_v3_large_100_224(self, accuracy):
        acc_result = accuracy["mobilenet_v3_large_100_224"]
        return acc_result
    
    track_accuracy_mobilenet_v3_large_100_224.unit = "Accuracy"
           
    def track_mobilenet_v3_large_075_224(self, model, lib, inter_list, intra_list, batch_size):
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
            accuracy[model] = acc
        return inference

    track_mobilenet_v3_large_075_224.params = (["mobilenet_v3_large_075_224"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_mobilenet_v3_large_075_224.unit = "Inference Time"
    
    def track_accuracy_mobilenet_v3_large_075_224(self, accuracy):
        acc_result = accuracy["mobilenet_v3_large_075_224"]
        return acc_result
    track_accuracy_mobilenet_v3_large_075_224.unit = "Accuracy"
