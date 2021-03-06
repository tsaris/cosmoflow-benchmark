"""
This module contains some utility callbacks for Keras training.
"""

# System
from time import time

# Externals
import tensorflow as tf

class TimingCallback(tf.keras.callbacks.Callback):
    """A Keras Callback which records the time of each epoch"""
    def __init__(self):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.starttime
        self.times.append(epoch_time)
        logs['time'] = epoch_time

#class LearningRateScheduleCallback(tf.keras.callbacks.Callback):
#    def __init__(self, multiplier,
#                 start_epoch=0, end_epoch=None,
#                 momentum_correction=True):
#        super().__init__()
#        self.start_epoch = start_epoch
#        self.end_epoch = end_epoch
#        self.momentum_correction = momentum_correction
#        self.initial_lr = None
#        self.restore_momentum = None
