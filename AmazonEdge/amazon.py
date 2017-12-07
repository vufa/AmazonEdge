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

    # a 10x10 board
    def __init__(self, size=10):
        self.board = np.zeros((size, size))
        self.board.fill(EMPTY)
        self.size = size
        self.current_player = BLACK
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

    def _on_board(self, position):
        """simply return True if position is within the bounds of [0, self.size]
        """
        (x, y) = position
        return x >= 0 and y >= 0 and x < self.size and y < self.size

    def is_legal(self, action):
        """determine if the given action (x,y) is a legal move
        """
        (x, y) = action
        if not self._on_board(action):
            return False
        if self.board[x][y] != EMPTY:
            return False
        return True

    def get_legal_moves(self):
        if self.__legal_move_cache is not None:
            return self.__legal_move_cache
        self.__legal_move_cache = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal((x, y)):
                    self.__legal_move_cache.append((x, y))
        return self.get_legal_moves()
