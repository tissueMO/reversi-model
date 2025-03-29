#!/bin/bash

nvidia-smi

python -c '
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", len(tf.config.list_physical_devices("GPU")) > 0)
print("GPU Devices:", tf.config.list_physical_devices("GPU"))
'
