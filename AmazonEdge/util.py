import os
import itertools
import numpy as np
from AmazonEdge import amazon


def flatten_idx(position, size):
    (x,y) = position
    return x * size + y