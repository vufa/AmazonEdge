#! /bin/bash

# Create model file
python2.7 -m build/create_model model.json output
python2.7 -m build/supervised_policy_training
