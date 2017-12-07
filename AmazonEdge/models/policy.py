from keras.models import Sequential, Model
from keras.layers import convolutional, merge, Input, BatchNormalization
from keras.layers.core import Activation, Flatten
import AmazonEdge.models.nn_util import neur
import numpy as np

@neuralnet