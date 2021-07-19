# Resnet Perturbations Riselab Mahoney Group
Performing perturbation analysis on resnets trained on various optimizers

Run resnet training experiments for perturbation analysis

# Train models using resnet_training.py in training-scripts: 

Supports

## MNIST:
python3 resnet_training.py --optimizer sgd 

## CIFAR10:
python3 resnet_training.py --optimizer sgd --dataset cifar10 

Params:
• weight_decay
• learning_rate
• dataset (either mnist or cifar10)
• epochs
• optimizer
• l2_lambda

resnet_cifar10_frank_wolfe.py trains resnets on frank wolfe optimizer.

Add perturbations to models and plot on perturbation_analysis.ipynb 

