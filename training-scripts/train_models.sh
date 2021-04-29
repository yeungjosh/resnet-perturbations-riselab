#!/bin/bash

nohup python3 resnet_training.py --optimizer sgd &
nohup python3 resnet_training.py --optimizer adam &
nohup python3 resnet_training.py --optimizer adahessian &

nohup python3 resnet_training.py --optimizer sgd --dataset cifar10 &
nohup python3 resnet_training.py --optimizer adam --dataset cifar10 &
nohup python3 resnet_training.py --optimizer adahessian --dataset cifar10 &