from AmazonEdge.training.supervised_policy_trainer import run_training
import unittest


class TestSupervisedPolicyTrainer(unittest.TestCase):
    def testTrain(self):
        model = 'output/model.json'
        data = 'data/hdf5/very_weak.hdf5'
        output = 'output/'
        args = [model, data, output, '--verbose', '--epochs', '1']
        run_training(args)


if __name__ == '__main__':
    unittest.main()
