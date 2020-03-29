import tensorflow as tf

# From : https://www.tensorflow.org/guide/gpu
# To find out which devices your operations and tensors are assigned to, put tf.debugging.set_log_device_placement(True) as the first statement of your program. Enabling device placement logging causes any Tensor allocations or operations to be printed.
tf.debugging.set_log_device_placement(True)

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

    
def SetupStrategy():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        tf.debugging.set_log_device_placement(True)
    return strategy

def SetupTensorflow():
    CheckDevices()
    return SetupStrategy()