#
# Copyright (c) 2018 CountStarlight
# Licensed under The MIT License (MIT)
# See: LICENSE
#
from AmazonEdge.ai import GreedyPolicyPlayer
from AmazonEdge.models.policy import CNNPolicy

MODEL = 'output/model.json'
WEIGHTS = 'output/weights.00000.hdf5'

policy = CNNPolicy.load_model(MODEL)
policy.model.load_weights(WEIGHTS)

player = GreedyPolicyPlayer(policy)