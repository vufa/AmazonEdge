# Amazon Edge

An Ai for Amazon game, AmazonEdge base on neural networks with supervised learning and reinforcement learning.

## Environment
* python 2.7
* anaconda3(recommend)

#### 1.Create an anaconda environment for Amazon Edge(recommend)

* Download Anaconda: https://www.anaconda.com/download/#linux
* Install:

```shell
conda create -n AmazonEdge python=2.7
source activate AmazonEdge
```
#### 2.Install dependency packages
```shell
pip install -r requirements.txt
```
#### 3.Use tensorflow as Keras backend
```shell
pip install tensorflow
```
Edit `~/.keras/keras.json` to
```json
{
    "image_dim_ordering": "tf", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "tensorflow"
}
```
## Phase 1: supervised learning of policy networks

### Supervised training script

To see what arguments are available, use

```shell
python2.7 -m AmazonEdge.training.supervised_policy_trainer --help
```

#### 1.Get a model file(a json specifying the policy network's architecture)

```shell
python2.7 -m build/create_model MODEL_NAME.json MODEL_PATH
```

#### 2.Running tests
```shell
python -m tests.test_supervised_policy_trainer
```