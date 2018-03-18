#
# Copyright (c) 2018 CountStarlight
# Licensed under The MIT License (MIT)
# See: LICENSE
#
# Policy players
import numpy as np
from AmazonEdge import amazon
from AmazonEdge import mcts
from operator import itemgetter


class GreedyPolicyPlayer(object):
    """A player that uses a greedy policy (i.e. chooses the highest probability
       move each turn)
    """

    def __init__(self, policy_function, pass_when_offered=False, move_limit=None):
        self.policy = policy_function
        self.pass_when_offered = pass_when_offered
        self.move_limit = move_limit

    def get_move(self, state):
        # list with sensible moves
        sensible_moves = [move for move in state.get_legal_moves]

        # check if there are sensible moves left to do
        if len(sensible_moves) > 0:
            move_probs = self.policy.eval_state(state, sensible_moves)
            max_prob = max(move_probs, key=itemgetter(1))
            return max_prob[0]

        # No 'sensible' moves available, so game over
        return amazon.GAME_OVER