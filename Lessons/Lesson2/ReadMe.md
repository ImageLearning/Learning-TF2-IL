
2. Basic ML & Actions & Your first Image recognizer
> In this lesson we will write our first Tensorflow / ML code. We will setup actions to confirm that the code we check in works and adhere to good principles (python linter). We'll end with a classic Image recognition sample


--------

Running this took under 30 minutes on my pc. if it took longer than that on yours you might consider adding a progress bar

https://www.tensorflow.org/api_docs/python/tf/keras/utils/Progbar

tf.keras.utils.Progbar(
    target, width=30, verbose=1, interval=0.05, stateful_metrics=None,
    unit_name='step'
)