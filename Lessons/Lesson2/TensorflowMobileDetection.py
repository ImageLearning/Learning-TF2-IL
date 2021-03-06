
import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# import tensorflow_datasets as tfds
# tfds.disable_progress_bar()

from PIL import Image  #Pillow needed


def preprocess(img, label):
    resized_image = tf.image.resize(image, [224,224])
    final_image = keras.applications.xception.preprocess_image(resized_image)
    return final_image, label


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
    
    # You can also do info.splits.total_num_examples to get the total
    # number of examples in the dataset.
    num_train_examples = info.splits['train'].num_examples
    num_test_examples = info.splits['test'].num_examples

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    

    # code from page 480 of of the model
    dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
    dataset_size = info.splits["train"] = num_examples # 3670
    classnames =  info.features["label"].names #["dandelion", "daisy", ...]
    n_classes = info.features["label"].num_classes # 5

    test_split, valid)split, train)_split = tfds.Split.TRAIN.subsplit([10,15,75])

    test_set = tfds.load("tf_flowers", split=test_split, as_supervised=True)
    valid_set = tfds.load("tf_flowers", split=valid_split, as_supervised=True)
    train_set = tfds.load("tf_flowers", split=train_split, as_supervised=True)

    batch_size = 32

    train_set = train_set.shuffle(1000)
    train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
    valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
    test_set = test_Set.map(preprocess).batch(batch_size).prefetch(1)

