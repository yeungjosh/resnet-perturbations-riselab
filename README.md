# Resnet Perturbations Riselab Mahoney Group
Performing perturbation analysis on resnets trained on various optimizers

Run resnet training experiments for perturbation analysis

# Train models using resnet_training.py in training-scripts: 

## MNIST:
python3 resnet_training.py --optimizer sgd 

## CIFAR10:
python3 resnet_training.py --optimizer sgd --dataset cifar10 

## Add perturbations to models and plot on perturbation_analysis.ipynb 

