
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def CheckDevices():
    # tf.config.list_physical_devices('GPU')
    print("----------")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first 2 GPU's, the one's that are linked
      try:
        tf.config.experimental.set_visible_devices([gpus[0],gpus[1]], 'GPU') # Let's only use GPU 0 and 1, i'm not sure ou PSU can handle all 3...
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

    

CheckDevices()

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # tf.debugging.set_log_device_placement(True)
    # https://www.tensorflow.org/lite/models/object_detection/overview 

    URL = 'http://download.tensorflow.org/models/object_detection/pet_faces_tfrecord.tar.gz'

    WORKINGDIR = os.getcwd()
    STORAGEDIR =  WORKINGDIR + "\\DataSet\\Lesson2\\"
    DOWNLOAD_TARGET = os.path.join( STORAGEDIR,'pet_faces.tar.gz')
    EXTRACTION_DIRECTORY =  os.path.join( STORAGEDIR,os.path.dirname('Data\\'))

    path_to_zip = tf.keras.utils.get_file(DOWNLOAD_TARGET, origin=URL, extract=True, cache_subdir=".\\", cache_dir=EXTRACTION_DIRECTORY)

    PATH = os.path.join(os.path.dirname(path_to_zip), 'pet_faces')
    
    train_dir = os.path.join(PATH, 'train')