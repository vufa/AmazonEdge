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

def get_legal(state):
    """Zero at all illegal moves, one at all legal moves.
    """
    feature = np.zeros((1, state.size, state.size))
    for (x, y) in state.get_legal_moves():
        feature[0, x, y] = 1
    return feature

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
    "zeros": {
        "size": 1,
        "function": lambda state: np.zeros((1, state.size, state.size))
    },
    "legal": {
        "size": 1,
        "function": get_legal
    }
}

DEFAULT_FEATURES = [
    "board", "ones", "zeros"]


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
        for i in range(len(feature_list)):
            feat = feature_list[i].lower()
            if feat in FEATURES:
                self.processors[i] = FEATURES[feat]["function"]
                self.output_dim += FEATURES[feat]["size"]
            else:
                raise ValueError("uknown feature: %s" % feat)

    def state_to_tensor(self, state):
        """Convert a GameState to a Theano-compatible tensor
        """
        feat_tensors = [proc(state) for proc in self.processors]

        # concatenate along feature dimensiion then add in a singleton 'batch' dimension
        f, s = self.output_dim, state.size
        return np.concatenate(feat_tensors).reshape(1, f, s, s)