from keras import backend as K
from keras.models import model_from_json
from keras.engine.topology import Layer
from AmazonEdge.preprocessing.preprocessing import Preprocess
import json

class NeuralNetBase(object):
    """Base class for neural network classes handing feature processing, construction
    of a 'forward' function, etc.
    """

    # keep track of subclasses to make generic saving/loading cleaner.
    # subclasses can be 'registered' with the @neuralnet decorator
    subclasses = {}

    def __int__(self, feature_list, **kwargs):
        """create a neural net object that preprocesses according to feature_list and uses
        a neural network specified by keyword arguments (using subclass' create_network())

        optional argument: init_network (boolean). If set to False, skips initializing
        self.model and self.forward and the calling function should set them.
        """
        self.preprocessor = Preprocess(feature_list)
        kwargs["input_dim"] = self.preprocessor.output_dim

        if kwargs.get('init_network', True):
            # self.__class__ refers to the subclass so that subclasses only
            # need to override create_network()
            self.model = self.__class__.create_network(**kwargs)
            # self.forward is a lambda function wrapping a Keras function
            self.forward = self._model_forward()