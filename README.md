# AmazonEdge

Build Status: [![Build Status](https://travis-ci.org/countstarlight/AmazonEdge.svg?branch=master)](https://travis-ci.org/countstarlight/AmazonEdge)

[简体中文](docs/zh_CN/README.md)

AmazonEdge is an AI for Game of Amazons, based on neural networks with supervised learning and reinforcement learning.

## Environment
* python 2.7
* Anaconda3(Recommend)

## For Linux:
#### 1.Create an `Anaconda` environment for `AmazonEdge`(Recommend)

* Download `Anaconda`: https://www.anaconda.com/download/#linux

* Installing `Anaconda` follow [Document](https://conda.io/docs/user-guide/install/linux.html)

* Create an environment for `AmazonEdge`:

```shell
conda create -n AmazonEdge python=2.7	# Create an environment named AmazonEdge with python2.7
source activate AmazonEdge	# Enter this environment
```
#### 2.Install dependency packages
```shell
pip install -r requirements.txt
```
#### 3.Use `tensorflow` as `Keras` backend
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
## Phase 1: Supervised Learning and Policy Networks

### Generate hdf5 file from actions file
```shell
python -m tools.actions_to_feature_layers
```
The input file at `data/actions/actions.txt` and the output file at `data/hdf5/`, you can edit `tools/actions_to_feature_layers` as needed.
### Supervised training

To see what arguments are available, use

```shell
python -m AmazonEdge.training.supervised_policy_trainer --help
```

#### 1.Get a model file(A json specifying the policy network's architecture)

```shell
python -m build/create_model MODEL_NAME.json MODEL_PATH
```

#### 2.Running Supervised training test
```shell
python -m tests.test_supervised_policy_trainer
```
