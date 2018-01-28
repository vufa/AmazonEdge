import os
from AmazonEdge.models.policy import CNNPolicy

def CreateModel(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list
    """
    import argparse
    parser = argparse.ArgumentParser(description='Create a convolutional neural network model')
    parser.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicy.save_model())")
    parser.add_argument("out_directory", help="directory where model will be saved")
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)
    arch = {'filters_per_layer': 128, 'layers': 12}  # args to CNNPolicy.create_network()
    features = ['board', 'ones']  # must match args to game_converter
    policy = CNNPolicy(features, **arch)
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)
    out_path = args.out_directory+'/'+args.model
    if os.path.exists(out_path):
        print("\033[0;33m%s\033[0m" %"[WARRING]model file %s exists. any previous data will be overwritten" %
              out_path)
    policy.save_model(out_path)
    print("\033[0;32m%s\033[0m" % "[SUCCESS]Model file created at %s successfully" %
          out_path)

if __name__ == '__main__':
    CreateModel()