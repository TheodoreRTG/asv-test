class TensorflowImageClassification:
    timeout = 1800
#    def setup(self, model, lib, inter_list, intra_list, batch_size):
 #       os.environ['model'] = model
 #       os.environ['lib'] = lib
 #       os.environ['inter'] = inter_list
 #       os.environ['intra'] = intra_list
 #       os.environ['batch'] = batch_size
#        
    def track_image_classification(self, model, lib, inter_list, intra_list, batch_size):
        #from .test-image_classification_with_tf_hub-itr-setthreads import load_image
        import sys
        sys.path.append('/root/asv-test/asv-conf/asv-test')
        import tensorflow as tf
        from functs import preprocess_image, load_image_from_url, load_image, show_image
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
        

        model_name = model
        inter_op_threads = inter_list
        intra_op_threads = intra_list
        mb = batch_size
        benchname = lib
        print("*"*150)
        print("Model name =", model_name, " batch=", mb, " for ", benchname)
        # Set TF threads
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        print("Inter threads = ", tf.config.threading.get_inter_op_parallelism_threads(), "AND Intra threads = ", tf.config.threading.get_intra_op_parallelism_threads())
        ############
        original_image_cache = {}

        # # # #

        image_size = 224
        dynamic_size = False

        model_handle_map = {
          "efficientnetv2-s": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2",
          "efficientnetv2-m": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/classification/2",
          "efficientnetv2-l": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2",
          "efficientnetv2-s-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/classification/2",
          "efficientnetv2-m-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/classification/2",
          "efficientnetv2-l-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/classification/2",
          "efficientnetv2-xl-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/classification/2",
          "efficientnetv2-b0-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2",
          "efficientnetv2-b1-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/classification/2",
          "efficientnetv2-b2-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/classification/2",
          "efficientnetv2-b3-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/classification/2",
          "efficientnetv2-s-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/classification/2",
          "efficientnetv2-m-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/classification/2",
          "efficientnetv2-l-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/classification/2",
          "efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/classification/2",
          "efficientnetv2-b0-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/classification/2",
          "efficientnetv2-b1-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/classification/2",
          "efficientnetv2-b2-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/classification/2",
          "efficientnetv2-b3-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/classification/2",
          "efficientnetv2-b0": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2",
          "efficientnetv2-b1": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/classification/2",
          "efficientnetv2-b2": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/classification/2",
          "efficientnetv2-b3": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/classification/2",
          "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1",
          "efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/classification/1",
          "efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/classification/1",
          "efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/classification/1",
          "efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/classification/1",
          "efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/classification/1",
          "efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/classification/1",
          "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/classification/1",
          "bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/ilsvrc2012_classification/1",
          "inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/classification/4",
          "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4",
          "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/classification/4",
          "resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/classification/4",
          "resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/classification/4",
          "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4",
          "resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/classification/4",
          "resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/4",
          "nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/classification/4",
          "nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/classification/4",
          "pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/classification/4",
          "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
          "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4",
          "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/4",
          "mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5",
          "mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5",
          "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5",
          "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5",
        }



        model_image_size_map = {
          "efficientnetv2-s": 384,
          "efficientnetv2-m": 480,
          "efficientnetv2-l": 480,
          "efficientnetv2-b0": 224,
          "efficientnetv2-b1": 240,
          "efficientnetv2-b2": 260,
          "efficientnetv2-b3": 300,
          "efficientnetv2-s-21k": 384,
          "efficientnetv2-m-21k": 480,
          "efficientnetv2-l-21k": 480,
          "efficientnetv2-xl-21k": 512,
          "efficientnetv2-b0-21k": 224,
          "efficientnetv2-b1-21k": 240,
          "efficientnetv2-b2-21k": 260,
          "efficientnetv2-b3-21k": 300,
          "efficientnetv2-s-21k-ft1k": 384,
          "efficientnetv2-m-21k-ft1k": 480,
          "efficientnetv2-l-21k-ft1k": 480,
          "efficientnetv2-xl-21k-ft1k": 512,
          "efficientnetv2-b0-21k-ft1k": 224,
          "efficientnetv2-b1-21k-ft1k": 240,
          "efficientnetv2-b2-21k-ft1k": 260,
          "efficientnetv2-b3-21k-ft1k": 300, 
          "efficientnet_b0": 224,
          "efficientnet_b1": 240,
          "efficientnet_b2": 260,
          "efficientnet_b3": 300,
          "efficientnet_b4": 380,
          "efficientnet_b5": 456,
          "efficientnet_b6": 528,
          "efficientnet_b7": 600,
          "inception_v3": 299,
          "inception_resnet_v2": 299,
          "mobilenet_v2_100_224": 224,
          "mobilenet_v2_130_224": 224,
          "mobilenet_v2_140_224": 224,
          "nasnet_large": 331,
          "nasnet_mobile": 224,
          "pnasnet_large": 331,
          "resnet_v1_50": 224,
          "resnet_v1_101": 224,
          "resnet_v1_152": 224,
          "resnet_v2_50": 224,
          "resnet_v2_101": 224,
          "resnet_v2_152": 224,
          "mobilenet_v3_small_100_224": 224,
          "mobilenet_v3_small_075_224": 224,
          "mobilenet_v3_large_100_224": 224,
          "mobilenet_v3_large_075_224": 224,
        }


        model_handle = model_handle_map[model_name]

        print(f"Selected model: {model_name} : {model_handle}")


        max_dynamic_size = 512
        if model_name in model_image_size_map:
          image_size = model_image_size_map[model_name]
          dynamic_size = False
          print(f"Images will be converted to {image_size}x{image_size}")
        else:
          dynamic_size = True
          print(f"Images will be capped to a max size of {max_dynamic_size}x{max_dynamic_size}")

        labels_file = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"

        #download labels and creates a maps
        downloaded_file = tf.keras.utils.get_file("labels.txt", origin=labels_file)

        classes = []

        with open(downloaded_file) as f:
          labels = f.readlines()
          classes = [l.strip() for l in labels]

        """You can select one of the images below, or use your own image. Just remember that the input size for the models vary and some of them use a dynamic input size (enabling inference on the unscaled image). Given that, the method `load_image` will already rescale the image to the expected format."""

        #@title Select an Input Image

        image_name = "turtle" # @param ['tiger', 'bus', 'car', 'cat', 'dog', 'apple', 'banana', 'turtle', 'flamingo', 'piano', 'honeycomb', 'teapot']

        images_for_test_map = {
            "tiger": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Bengal_tiger_%28Panthera_tigris_tigris%29_female_3_crop.jpg",
            #by Charles James Sharp, CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons
            "bus": "https://upload.wikimedia.org/wikipedia/commons/6/63/LT_471_%28LTZ_1471%29_Arriva_London_New_Routemaster_%2819522859218%29.jpg",
            #by Martin49 from London, England, CC BY 2.0 <https://creativecommons.org/licenses/by/2.0>, via Wikimedia Commons
            "car": "https://upload.wikimedia.org/wikipedia/commons/4/49/2013-2016_Toyota_Corolla_%28ZRE172R%29_SX_sedan_%282018-09-17%29_01.jpg",
            #by EurovisionNim, CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons
            "cat": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
            #by Alvesgaspar, CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0>, via Wikimedia Commons
            "dog": "https://upload.wikimedia.org/wikipedia/commons/archive/a/a9/20090914031557%21Saluki_dog_breed.jpg",
            #by Craig Pemberton, CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0>, via Wikimedia Commons
            "apple": "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg",
            #by Abhijit Tembhekar from Mumbai, India, CC BY 2.0 <https://creativecommons.org/licenses/by/2.0>, via Wikimedia Commons
            "banana": "https://upload.wikimedia.org/wikipedia/commons/1/1c/Bananas_white_background.jpg",
            #by fir0002  flagstaffotos [at] gmail.com		Canon 20D + Tamron 28-75mm f/2.8, GFDL 1.2 <http://www.gnu.org/licenses/old-licenses/fdl-1.2.html>, via Wikimedia Commons
            "turtle": "https://upload.wikimedia.org/wikipedia/commons/8/80/Turtle_golfina_escobilla_oaxaca_mexico_claudio_giovenzana_2010.jpg",
            #by Claudio Giovenzana, CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0>, via Wikimedia Commons
            "flamingo": "https://upload.wikimedia.org/wikipedia/commons/b/b8/James_Flamingos_MC.jpg",
            #by Christian Mehlführer, User:Chmehl, CC BY 3.0 <https://creativecommons.org/licenses/by/3.0>, via Wikimedia Commons
            "piano": "https://upload.wikimedia.org/wikipedia/commons/d/da/Steinway_%26_Sons_upright_piano%2C_model_K-132%2C_manufactured_at_Steinway%27s_factory_in_Hamburg%2C_Germany.png",
            #by "Photo: © Copyright Steinway & Sons", CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0>, via Wikimedia Commons
            "honeycomb": "https://upload.wikimedia.org/wikipedia/commons/f/f7/Honey_comb.jpg",
            #by Merdal, CC BY-SA 3.0 <http://creativecommons.org/licenses/by-sa/3.0/>, via Wikimedia Commons
            "teapot": "https://upload.wikimedia.org/wikipedia/commons/4/44/Black_tea_pot_cropped.jpg",
            #by Mendhak, CC BY-SA 2.0 <https://creativecommons.org/licenses/by-sa/2.0>, via Wikimedia Commons
        }

        img_url = images_for_test_map[image_name]
        image, original_image = load_image(img_url, image_size, dynamic_size, max_dynamic_size)
        show_image(image, 'Scaled image')

        """Now that the model was chosen, loading it with TensorFlow Hub is simple.

        This also calls the model with a random input as a "warmup" run. Subsequent calls are often much faster, and you can compare this with the latency below.

        *Note:* models that use a dynamic size might need a fresh "warmup" run for each image size.
        """

        # Commented out IPython magic to ensure Python compatibility.
        classifier = hub.load(model_handle)

        input_shape = image.shape
        print("SHAPE BEFORE", input_shape)
        input_shape_final = (mb, input_shape[1], input_shape[2], input_shape[3])
        print("SHAPE AFTER", input_shape_final)
        warmup_input = tf.random.uniform(input_shape_final, 0, 1.0)
        warmup_logits = classifier(warmup_input).numpy()

        """Everything is ready for inference. Here you can see the top 5 results from the model for the selected image."""

        # Commented out IPython magic to ensure Python compatibility.
        # Run model on image
        inference_times = []
        tries=5
        for _ in range(tries):
            start_time = time.time_ns()
            probabilities = tf.nn.softmax(classifier(warmup_input)).numpy()
            end_time = time.time_ns()
            inference_time = np.round((end_time - start_time) / 1e6, 2)
            inference_times.append(inference_time)
            print('DONE,DONE', flush=True)
        print(inference_times)
        perf=np.min(inference_times)
        print("Inference time:", perf)
        return perf
#        with open('output-'+benchname+str(mb)+'.csv', 'a+', newline='') as f:
#            writer = csv.writer(f)
#            writer.writerow([model_name, benchname, str(mb), str(inter_op_threads), str(intra_op_threads), str(perf) ])
#        top_5 = tf.argsort(probabilities, axis=-1, direction="DESCENDING")[0][:5].numpy()
#        np_classes = np.array(classes)

        # Some models include an additional 'background' class in the predictions, so
        # we must account for this when reading the class labels.
        #includes_background_class = probabilities.shape[1] == 1001

        #for i, item in enumerate(top_5):
        #  class_index = item if includes_background_class else item + 1
        #  line = f'({i+1}) {class_index:4} - {classes[class_index]}: {probabilities[0][top_5][i]}'
        #  print(line)

        #show_image(image, '')

        """## Learn More

        If you want to learn more and try how to do Transfer Learning with these models you can try this tutorial: [Transfer Learning for Image classification](https://www.tensorflow.org/hub/tutorials/tf2_image_retraining) 

        If you want to check on more image models you can check them out on [tfhub.dev](https://tfhub.dev/s?module-type=image-augmentation,image-classification,image-classification-logits,image-classifier,image-feature-vector,image-generator,image-object-detection,image-others,image-pose-detection,image-segmentation,image-style-transfer,image-super-resolution,image-rnn-agent)
        """
  #      os.system("image_classification_with_tf_hub-itr-setthreads.py $model $lib $inter $intra $batch")
        
    track_image_classification.params = (["inception_v3",  "nasnet_mobile"], ["tp", "eigen"], [16], [16], [1, 16, 32])
    track_image_classification.unit = "Inference Time"
