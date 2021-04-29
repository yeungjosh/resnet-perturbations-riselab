# -*- coding: utf-8 -*-
"""
# ResNet training in PyTorch
"""
# !pip install resnet_pytorch

import adahessian.image_classification.optim_adahessian

import argparse
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import CIFAR10
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time

import torch
import torch.nn as nn
import torch.nn.init as init
from resnet_pytorch import ResNet 

from torch import nn, optim
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from copy import copy, deepcopy
import numpy as np
import os

# Helper functions
import tools
import dataset_helpers

# We define all the classes and function regarding the ResNet architecture in this code cell
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
 
def _weights_init(m):
    """
        Initialization of CNN weights
    """
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    """
      Identity mapping between ResNet blocks with diffrenet size feature map
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# A basic block as shown in Fig.3 (right) in the paper consists of two convolutional blocks, each followed by a Bach-Norm layer. 
# Every basic block is shortcuted in ResNet architecture to construct f(x)+x module. 
# Expansion for option 'A' in the paper is equal to identity with extra zero entries padded
# for increasing dimensions between layers with different feature map size. This option introduces no extra parameter. 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 experiment, ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Stack of 3 times 2*n (n is the number of basic blocks) layers are used for making the ResNet model, 
# where each 2n layers have feature maps of size {16,32,64}, respectively. 
# The subsampling is performed by convolutions with a stride of 2.
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay", help="set weight decay")
    parser.add_argument("--learning_rate", help="set learning_rate")
    parser.add_argument("--dataset", help="set dataset")
    # parser.add_argument("--lr_step_size", help="set lr_step_size")
    # parser.add_argument("--momentum", help="set momentum")
    parser.add_argument("--epochs", help="set epochs")
    parser.add_argument("--optimizer", help="set optimizer")
    parser.add_argument("--l2_lambda", help="set lambda for l2")

    args = parser.parse_args()

    # To get the current working directory 
    cwd = os.getcwd()

    start_ts = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 110
    if args.epochs:
        print("epochs: ",args.epochs)
        epochs=int(args.epochs)

    # if args.dataset == 'mnist':
    # default dataset is mnist
    model = dataset_helpers.MnistResNet().to(device)
    train_loader, val_loader = dataset_helpers.getData(name='mnist', train_bs=128, test_bs=1000)

    if args.dataset == 'cifar10':
        model = resnet20().to(device)
        train_loader, val_loader = dataset_helpers.getData(name='cifar10', train_bs=128, test_bs=1000)

    losses = []
    loss_function = nn.CrossEntropyLoss()

    if args.weight_decay:
        weight_decay = args.weight_decay
    else:
        weight_decay = 5e-4

    l2_lambda=3e-4
    if args.l2_lambda:
        l2_lambda = args.l2_lambda

    # Default Optimizer adadelta    
    learning_rate = 1.0 #@param {type:"number"}
    if args.learning_rate:
        print("learning_rate: ",args.learning_rate)
        learning_rate=(args.learning_rate)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if args.optimizer == 'sgd':
        learning_rate = 0.01 #@param {type:"number"}
        if args.learning_rate:
            print("learning_rate: ",args.learning_rate)
            learning_rate=(args.learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if args.optimizer == 'adam':
        learning_rate = 0.001
        if args.learning_rate:
            print("learning_rate: ",args.learning_rate)
            learning_rate=(args.learning_rate)
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)

    if args.optimizer == 'adahessian':
        optimizer = adahessian.image_classification.optim_adahessian.Adahessian(model.parameters())


    batches = len(train_loader)
    val_batches = len(val_loader)
    # keep best model
    accuracies=[]
    best_accuracy = 0
    best_model = deepcopy(model)

    # training loop + eval loop
    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        model.train()
        # # lr decay
        optimizer = tools.exp_lr_scheduler(epoch, optimizer, decay_eff=0.1, decayEpoch=[30,60,90])

        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)
            
            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y) + tools.L2reg(model, l2_lambda)
            
            if args.optimizer == 'adahessian':
                loss.backward(create_graph=True)  # You need this line for Hessian backprop
            else:
                loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
            
        torch.cuda.empty_cache()
        
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)
                outputs = model(X)
                val_losses += loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1]
                
                # this gives an error when running on AWS
                # for acc, metric in zip((precision, recall, f1, accuracy), 
                #                        (precision_score, recall_score, f1_score, accuracy_score)):
                #     acc.append(
                #         calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                #     )
                
                # temporarily fixes error
                accuracy.append(tools.calculate_metric(accuracy_score, y.cpu(), predicted_classes.cpu()))

        
        current_model_accuracy = sum(accuracy)/val_batches
        accuracies.append(current_model_accuracy)

        if current_model_accuracy > best_accuracy:
            best_model = deepcopy(model)
            best_accuracy=current_model_accuracy
            
        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
        tools.print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss/batches)
        print('current_model_accuracy: ',current_model_accuracy)
        print('best_accuracy: ',best_accuracy)

    optimizer_name = 'adadelta'
    if args.optimizer:
        optimizer_name = str(args.optimizer)

    dataset_name = 'mnist'
    if args.dataset == 'cifar10':
        dataset_name = 'cifar10'
    model_save_name = '{0}_resnet_{1}_best.pkl'.format(dataset_name, optimizer_name)
    path = os.path.join(cwd,model_save_name)
    torch.save(best_model.state_dict(), path)

    print(losses)

    print(f"Training time: {time.time()-start_ts}s")

    print('Best accuracy: ',best_accuracy)


