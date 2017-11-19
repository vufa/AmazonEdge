import numpy as np

WHITE = -1
BLACK = +1
EMPTY = 0
PASS_MOVE = None

class GameState(object):
    """
    State of Amazon chess and some basic functions to interact with it
    """

    # Looking up positions adjacent to a given takes a surprising
    # amount of time, hence this shared lookup table {boardsize: {position: [neighbors]}}
    __NEIGHBORS_CACHE = {}

    def __init__(self, size=10):
        self.board = np.zeros((size, size))
        self.board.fill(EMPTY)
        self.size = size
        self.current_player = BLACK
        self.ko = None
        self.handicaps = []
        self.history = []
        