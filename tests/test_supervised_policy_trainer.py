import os
from AmazonEdge.training.supervised_policy_trainer import run_training
import unittest


class TestSupervisedPolicyTrainer(unittest.TestCase):
    def testTrain(self):
        model = 'tests/test_data/model.json'
        data = 'tests/test_data/hdf5/very_weak.hdf5'
        output = 'tests/test_data/.tmp.training/'
        args = [model, data, output, '--verbose', '--epochs', '1']
        run_training(args)

        os.remove(os.path.join(output, 'metadata.json'))
        os.remove(os.path.join(output, 'shuffle.npz'))
        os.remove(os.path.join(output, 'weights.00000.hdf5'))
        os.rmdir(output)


if __name__ == '__main__':
    unittest.main()
