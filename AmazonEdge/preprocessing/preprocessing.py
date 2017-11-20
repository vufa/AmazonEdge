import numpy as np
import AmazonEdge.amazon as am

def get_board(state):
    """A feature encoding WHITE BLACK BARRIER and EMPTY on separate planes, but plane 0
    always refers to the current player and plane 1 to the opponent
    """
    planes = np.zeros((4, state.site, state.site))
    planes[0, :, :] = state.board == state.current_player  # own stone
    planes[1, :, :] = state.board == -state.current_player  # opponent stone
    planes[2, :, :] = state.board == am.BARRIER  # barrier placed
    planes[3, :, :] = state.board == am.EMPTY  # empty space

# named features and their sizes are defined here
FEATURES = {
    "board": {
        "size": 4,
        "function": get_board
    },
    "ones": {
        "size": 1,
        "function": lambda state: np.ones((1, state.size, state.size))
    },
    "turns_since": {
        "size": 8,
        "function":
    }
}

DEFAULT_FEATURES = [
    "board", "ones", "turns_since", "liberties", "capture_size",
    "self_atari_size", "liberties_after", "ladder_capture", "ladder_escape",
    "sensibleness", "zeros"]

class Preprocess(object):
    """a class to convert from AmazonEdge objects to tensors of one-hot
    features for NN inputs
    """

    def __init__(self, feature_list = DEFAULT_FEATURES):
        """create a preprocessor object that will concatenate together the
        given list of features
        """

        self.output_dim = 0
        self.feature_list = feature_list
        self.processors = [None] * len(feature_list)
