#! /bin/bash

# Create model file
python2.7 -m build/create_model model.json output
python -m AmazonEdge.training.supervised_policy_trainer output/model.json data/hdf5/very_weak.hdf5 output/training_results/ --epochs 5 --minibatch 10 --learning-rate 0.01
