from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

#https://www.tensorflow.org/lite/convert

#https://www.tensorflow.org/lite/r2/convert/

import os
import sys
import io
import numpy as np
import tensorflow as tf

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md


import tensorflow as tf

saved_model_dir = os.path.join(os.getcwd(),'./DataSet/Tensorflow-FrozenModelArchive/11-Class-Octocat-Recognizer')

print(str(saved_model_dir))

print("---")

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("./DataSet/Tensorflow-FrozenModelArchive/11-Class-Octocat-Recognizer.tflite", "wb").write(tflite_model)