import tensorflow as tf

# From : https://www.tensorflow.org/guide/gpu
# To find out which devices your operations and tensors are assigned to, put tf.debugging.set_log_device_placement(True) as the first statement of your program. Enabling device placement logging causes any Tensor allocations or operations to be printed.
tf.debugging.set_log_device_placement(True)


# N.B. The machine I am working on has 3 GPU's 
# - They are numbered 0, 1, and 2
#  Found device 0 with properties:
#      pciBusID: 0000:4f:00.0 name: Quadro RTX 8000 computeCapability: 7.5
#      coreClock: 1.77GHz coreCount: 72 deviceMemorySize: 47.62GiB deviceMemoryBandwidth: 625.94GiB/s
#  Found device 1 with properties:
#      Quadro RTX 8000 computeCapability: 7.5
#      coreClock: 1.77GHz coreCount: 72 deviceMemorySize: 47.62GiB deviceMemoryBandwidth: 625.94GiB/s
#  Found device 2 with properties:
#     pciBusID: 0000:e6:00.0 name: Quadro RTX 4000 computeCapability: 7.5
#     coreClock: 1.545GHz coreCount: 36 deviceMemorySize: 8.00GiB deviceMemoryBandwidth: 387.49GiB/s
#
# Devices 0 and 1 are connected with an NVLink : https://www.nvidia.com/en-us/data-center/nvlink/
#  and setup with nvidia-smi per the instructions here https://imagelearning.community/docs/Tensorflow/install-windows/  
#
#   Adding visible gpu devices: 0, 1
#   Device interconnect StreamExecutor with strength 1 edge matrix:
#         0 1 
#    0:   N Y 
#    1:   Y N 

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
    tf.debugging.set_log_device_placement(True)

