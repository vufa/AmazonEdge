# AmazonEdge

构建状态: [![Build Status](https://travis-ci.org/countstarlight/AmazonEdge.svg?branch=master)](https://travis-ci.org/countstarlight/AmazonEdge)

AmazonEdge 是一个亚马逊棋AI, 基于神经网路，借助监督式学习和增强学习。

## 环境要求
* python 2.7
* Anaconda3(建议)

## 在Linux系统上的配置
#### 1.用`Anaconda`为`AmazonEdge`创建一个环境(建议)

* 下载 Anaconda: https://www.anaconda.com/download/#linux

* 根据[文档](https://conda.io/docs/user-guide/install/linux.html)安装 Anaconda

* 为`AmazonEdge`创建一个环境:

```shell
conda create -n AmazonEdge python=2.7   #创建一个python版本为2.7，名称为AmazonEdge的环境
source activate AmazonEdge      #进入这个环境
```
#### 2.安装依赖包
```shell
pip install -r requirements.txt
```
#### 3.使用 `tensorflow` 作为 `Keras` 的后端
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
## 第1阶段: 监督式学习建立决策网络

### 用步法文件生成训练所需的hdf5文件
```shell
python -m tools.actions_to_feature_layers
```
输入的步法文件为 `data/actions/actions.txt` ，输出的文件在 `data/hdf5/`, 你可以修改 `tools/actions_to_feature_layers` 中的输入输出路径及文件名。
### 监督式训练

要查看提供了哪些参数，使用：

```shell
python -m AmazonEdge.training.supervised_policy_trainer --help
```

#### 1.获得一个模型文件（用于描述网络结构的json格式文件）

```shell
python -m build/create_model MODEL_NAME.json MODEL_PATH
```

#### 2.运行监督式训练测试
```shell
python -m tests.test_supervised_policy_trainer
```
