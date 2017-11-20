import numpy as np

WHITE = -1
BLACK = +1
BARRIER = 2
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
        self.is_end_of_game = False
        # `self.liberty_sets` is a 2D array with the same indexes as `board`
        # each entry points to a set of tuples - the liberties of a stone's
        # connected block. By caching liberties in this way, we can directly
        # optimize update functions (e.g. do_move) and in doing so indirectly
        # speed up any function that queries liberties
        self._create_neighbors_cache()
        self.liberty_sets = [[set() for _ in range(size)] for _ in range(size)]
        for x in range(size):
            for y in range(size):
                self.liberty_sets[x][y] = set(self._neighbors((x, y)))
        # separately cache the 2D numpy array of the _size_ of liberty sets
        # at each board position
        self.liberty_counts = np.zeros((size, size), dtype=np.int)
        self.liberty_counts.fill(-1)
