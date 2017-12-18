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

    def _model_forward(self):
        """Construct a function using the current keras backend that, when given a batch
        of inputs, simply processes them forward and returns the output

        This is an opposed to model.compile(), which takes a loss function
        and training method.
        """
        # The uses_learning_phase property is True if the model contains layers that behave
        # differently during training and testing, e.g. Dropout or BatchNormalization.
        # In these cases, K.learning_phase() is a reference to a backend variable that should
        # be set to 0 when using the network in prediction mode and is automatically set to 1
        # during training.
        if self.model.uses_learning_phase:
            forward_function = K.function([self.model.input, K.learning_phase()],
                                          [self.model.output])
            # the forward_funtion returns a list of tensors
            # the first [0] gets the front tensor.
            return lambda inpt: forward_function([input, 0])[0]
        else:
            # identical but without a second input argument for the learning phase
            forward_function = K.function([self.model.input], [self.model.output])
            return lambda inpt: forward_function([inpt])[0]

    @staticmethod
    def load_model(json_file):
        """create a new neural net object from the architecture specified in json_file
        """
        with open(json_file, 'r') as f:
            object_specs = json.load(f)

        # Create object; may be a subclass of networks saved in specs['class']
        class_name = object_specs.get('class', 'CNNPolicy')
        try:
            network_class = NeuralNetBase.subclasses[class_name]
        except KeyError:
            raise ValueError("Unknown neural network type in json file: {}\n"
                             "(was it registered with the @neuralnet decorator?)"
                             .format(class_name))

        # create new object
        new_net = network_class(object_specs['feature_list'], init_network=False)

        new_net.model = model_from_json(object_specs['keras_model'], custom_objects ={'Bias': Bias})
        if 'weights_file' in object_specs:
            new_net.model.load_weights(object_specs['weights_file'])
        new_net.forward = new_net._model_forward()
        return new_net

    def save_model(self, json_file, weights_file=None):
        """write the network """