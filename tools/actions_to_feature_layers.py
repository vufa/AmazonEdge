#
# Copyright (c) 2018 CountStarlight
# Licensed under The MIT License (MIT)
# See: LICENSE
#
import h5py
import numpy as np

ActionsPath = 'data/actions/actions.txt'
OutputPath = 'data/hdf5/HDF5_FILE.hdf5'


class ToFeatures(object):
    """Read actions file and generate feature layers
    """

    def __init__(self):
        self.statesData = []
        self.actionsData = []
        self.meta = []
        self.stateLen = 0
        self.actions = np.loadtxt(ActionsPath, dtype=np.uint8)
        self.actionLen = len(self.actions)
        self.states = np.zeros((self.actionLen, 6, 10, 10), dtype=np.uint8)
        self.output = h5py.File(OutputPath, 'w')
        # Feature: board
        self.states_f = np.zeros((10, 10), dtype=np.uint8)  # states of black
        self.states_s = np.zeros((10, 10), dtype=np.uint8)  # states of white
        self.states_b = np.zeros((10, 10), dtype=np.uint8)  # states of barrier
        self.states_e = np.ones((10, 10), dtype=np.uint8)  # states of empty spaces
        # Feature: ones
        self.states_o = np.ones((10, 10), dtype=np.uint8)
        # Feature: zeros
        self.states_z = np.zeros((10, 10), dtype=np.uint8)
        # Init
        # BLACK
        self.states_f[0][3] = 1
        self.states_f[0][6] = 1
        self.states_f[3][0] = 1
        self.states_f[3][9] = 1
        # WHITE
        self.states_s[6][0] = 1
        self.states_s[6][9] = 1
        self.states_s[9][3] = 1
        self.states_s[9][6] = 1
        # EMPTY PLACE
        self.states_e[0][3] = 0
        self.states_e[0][6] = 0
        self.states_e[3][0] = 0
        self.states_e[3][9] = 0
        self.states_e[6][0] = 0
        self.states_e[6][9] = 0
        self.states_e[9][3] = 0
        self.states_e[9][6] = 0

    def generator(self):
        for i in range(len(self.actions)):
            for j in range(10):
                for k in range(10):
                    self.states[i][0][j][k] = self.states_f[j][k]
                    self.states[i][1][j][k] = self.states_s[j][k]
                    self.states[i][2][j][k] = self.states_b[j][k]
                    self.states[i][3][j][k] = self.states_e[j][k]
                    self.states[i][4][j][k] = self.states_o[j][k]
                    self.states[i][5][j][k] = self.states_z[j][k]
            (x1, y1, x2, y2, x3, y3) = self.actions[i]
            self.make_move((x1, y1), (x2, y2), (x3, y3))
        self.output['states'] = self.states
        self.output['actions'] = self.actions
        self.output.close()

    def make_move(self, f, t, b):
        (x1, y1) = f
        (x2, y2) = t
        (x3, y3) = b
        if 1 == self.states_f[10 - y1][x1 - 1]:  # Is BLACK
            self.states_f[10 - y1][x1 - 1] = 0
            self.states_f[10 - y2][x2 - 1] = 1
        else:  # Is white
            self.states_s[10 - y1][x1 - 1] = 0
            self.states_s[10 - y2][x2 - 1] = 1
        # For Barrier layer
        self.states_b[10 - y3][x3 - 1] = 1
        # For Empty layer
        self.states_e[10 - y1][x1 - 1] = 1
        self.states_e[10 - y2][x2 - 1] = 0
        self.states_e[10 - y3][x3 - 1] = 0


if __name__ == '__main__':
    gen = ToFeatures()
    gen.generator()
