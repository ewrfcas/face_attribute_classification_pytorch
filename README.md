# Face attribute classification based on pytorch

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![pytorch 1.3.0](https://img.shields.io/badge/pytorch-1.3.0-green.svg?style=plastic)

In this repository, we implement the face attribute classification method based on pytorch according to **Adaptively Weighted Multi-task Deep Network for Person Attribute Classific**

The official caffe code is https://github.com/qiexing/adaptive_weighted_attribute

[[Paper](https://dl.acm.org/doi/10.1145/3123266.3123424)]

## Results

| methods | test mean accuracy | 
| ------ | ------ |
| resnet50 | 90.479 |
| resnet50+adaptive weight | **90.845** |

## Download trained model

Coming soon

## Test

Coming soon

## Train

1. Download img_celeba_aligned dataset

2. python train.py -input_path {img_celeba_aligned_path}
