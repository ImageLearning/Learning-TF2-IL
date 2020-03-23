# https://github.com/tf-coreml/tf-coreml


#### - https://www.raywenderlich.com/577-core-ml-and-vision-machine-learning-in-ios-11-tutorial
#### - https://developer.apple.com/documentation/coreml/converting_trained_models_to_core_ml
#### - https://developer.apple.com/documentation/vision/classifying_images_with_vision_and_core_ml
#### - https://github.com/tf-coreml/tf-coreml
#### - https://developer.apple.com/documentation/vision/classifying_images_with_vision_and_core_ml

# https://hackernoon.com/integrating-tensorflow-model-in-an-ios-app-cecf30b9068d 
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import io
import numpy as np
import tensorflow as tf
import tfcoreml as tf_converter

print(str(sys.path))

print("---")

print(str(os.getcwd()))

tf_converter.convert(tf_model_path='./DataSet/Tensorflow-FrozenModelArchive/11-Class-Octocat-Recognizer/frozen_inference_graph.pb',
                     mlmodel_path='./DataSet/CoreML-FrozenModelArchive/11-class-octocat.mlmodel',
                     output_feature_names=['softmax:0'])