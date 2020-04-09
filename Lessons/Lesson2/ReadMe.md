
2. Basic ML & Actions & Your first Image Classifier
> In this lesson we will write our first Tensorflow / ML code. We will setup actions to confirm that the code we check in works and adhere to good principles (python linter). We'll end with a classic Image recognition sample

Object Detection
- https://www.tensorflow.org/lite/models/object_detection/overview
- https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193
- https://cloud.google.com/blog/products/gcp/training-an-object-detector-using-cloud-machine-learning-engine
Object Classification 
- https://www.tensorflow.org/lite/models/image_classification/overview 


First we detect if there IS a sticker
- What the bounds of it are
- maybe it's orientation
then we classify which octocat it is later

--------

https://github.com/marketplace?type=actions&query=python

https://github.com/marketplace/actions/python-lint

--------

Running this took under 30 minutes on my pc. if it took longer than that on yours you might consider adding a progress bar

https://www.tensorflow.org/api_docs/python/tf/keras/utils/Progbar
tf.keras.utils.Progbar(
    target, width=30, verbose=1, interval=0.05, stateful_metrics=None,
    unit_name='step'
)
