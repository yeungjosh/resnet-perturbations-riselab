# -*- coding: utf-8 -*-
"""resnet_cifar10_frank_wolfe.ipynb
# ResNet for CIFAR10 in PyTorch
"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('resnet_pytorch')
install('barbar')

# !pip install resnet_pytorch
# !pip install --upgrade --force-reinstall --quiet git+https://github.com/ZIB-IOL/StochasticFrankWolfe.git@arXiv-2010.07243v2
# !pip install --quiet barbar
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

# Helper functions
import tools
import dataset_helpers

import frankwolfe.pytorch as fw

# from google.colab import drive
# drive.mount('/content/gdrive')

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


def test(net):
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


# if __name__ == "__main__":
#     for net_name in __all__:
#         if net_name.startswith('resnet'):
#             print(net_name)
#             test(globals()[net_name]())
#             print()

#  class MyResNetArgs:
#    """
#     Passing the hyperparameters to the model
#    """
#    def __init__(self, arch='resnet20' ,epochs=200, start_epoch=0, batch_size=128, lr=0.1, momentum=0.9, weight_decay=1e-4, print_freq=55,
#                  evaluate=0, pretrained=0, half=0, save_dir='save_temp', save_every=10):
#         self.save_every = save_every #Saves checkpoints at every specified number of epochs
#         self.save_dir = save_dir #The directory used to save the trained models
#         self.half = half #use half-precision(16-bit)
#         self.evaluate = evaluate #evaluate model on the validation set
#         self.pretrained = pretrained #evaluate the pretrained model on the validation set
#         self.print_freq = print_freq #print frequency 
#         self.weight_decay = weight_decay
#         self.momentum = momentum 
#         self.lr = lr #Learning rate
#         self.batch_size = batch_size 
#         self.start_epoch = start_epoch
#         self.epochs = epochs
#         self.arch = arch #ResNet model

# from torchsummary import summary
# args=MyResNetArgs('resnet20',pretrained=0)
# #model = resnet.__dict__[args.arch]()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# model = resnet.__dict__[args.arch]().to(device)
# summary(model, (3,32,32))
# best_prec1 = 0

class RetractionLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Retracts the learning rate as follows. Two running averages are kept, one of length n_close, one of n_far. Adjust
    the learning_rate depending on the relation of far_average and close_average. Decrease by 1-retraction_factor.
    Increase by 1/(1 - retraction_factor*growth_factor)
    """
    def __init__(self, optimizer, retraction_factor=0.3, n_close=5, n_far=10, lowerBound=1e-5, upperBound=1, growth_factor=0.2, last_epoch=-1):
        self.retraction_factor = retraction_factor
        self.n_close = n_close
        self.n_far = n_far
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.growth_factor = growth_factor

        assert (0 <= self.retraction_factor < 1), "Retraction factor must be in [0, 1[."
        assert (0 <= self.lowerBound < self.upperBound <= 1), "Bounds must be in [0, 1]"
        assert (0 < self.growth_factor <= 1), "Growth factor must be in ]0, 1]"

        self.closeAverage = RunningAverage(self.n_close)
        self.farAverage = RunningAverage(self.n_far)

        super(RetractionLR, self).__init__(optimizer, last_epoch)

    def update_averages(self, loss):
        self.closeAverage(loss)
        self.farAverage(loss)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        factor = 1
        if self.farAverage.is_complete() and self.closeAverage.is_complete():
            if self.closeAverage.result() > self.farAverage.result():
                # Decrease the learning rate
                factor = 1 - self.retraction_factor
            elif self.farAverage.result() > self.closeAverage.result():
                # Increase the learning rate
                factor = 1./(1 - self.retraction_factor*self.growth_factor)

        return [max(self.lowerBound, min(factor * group['lr'], self.upperBound)) for group in self.optimizer.param_groups]

class RunningAverage(object):
    """Tracks the running average of n numbers"""
    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.entries = []

    def result(self):
        return self.avg

    def get_count(self):
        return len(self.entries)

    def is_complete(self):
        return len(self.entries) == self.n

    def __call__(self, val):
        if len(self.entries) == self.n:
            l = self.entries.pop(0)
            self.sum -= l
        self.entries.append(val)
        self.sum += val
        self.avg = self.sum / len(self.entries)

    def __str__(self):
        return str(self.avg)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def result(self):
        return self.avg

    def __call__(self, val, n=1):
        """val is an average over n samples. To compute the overall average, add val*n to sum and increase count by n"""
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay", help="set weight decay")
    parser.add_argument("--learning_rate", help="set learning_rate")
    parser.add_argument("--lr_scheduler_active", help="set lr_scheduler_active")
    parser.add_argument("--lr_decrease_factor", help="set lr_decrease_factor")
    parser.add_argument("--lr_step_size", help="set lr_step_size")
    parser.add_argument("--ord", help="set ord")
    parser.add_argument("--value", help="set value")
    parser.add_argument("--momentum", help="set momentum")
    parser.add_argument("--retraction", help="enable retraction of the learning rate")
    parser.add_argument("--epochs", help="set epochs")

    args = parser.parse_args()

    weight_decay=5e-4
    if args.weight_decay:
        print("weight decay: ",args.weight_decay)
        weight_decay=args.weight_decay
    # Init model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet20().to(device)

    #@title Choosing Lp-Norm constraints
    #@markdown The following cell allows you to set Lp-norm constraints for the chosen network. For exact parameters both for the constraints and the optimizer see the last cell of this notebook.
    ord =  "1" #@param [1, 2, 5, 'inf']
    if args.ord:
        print("ord: ",args.ord)
        ord=str(args.ord)
    ord = float(ord)
    value = 1000000 #@param {type:"number"}
    if args.value:
        print("value: ",args.value)
        value=(args.value)
    mode = 'initialization' #@param ['initialization', 'radius', 'diameter']
    assert value > 0

    # Select constraints
    constraints = fw.constraints.create_lp_constraints(model, ord=ord, value=value, mode=mode)

    #@title Configuring the Frank-Wolfe Algorithm
    #@markdown Choose momentum and learning rate rescaling, see Section 3.1 of [arXiv:2010.07243](https://arxiv.org/pdf/2010.07243.pdf).
    momentum = 0.9 #@param {type:"number"}
    if args.momentum:
        print("momentum: ",args.momentum)
        momentum=(args.momentum)
    rescale = 'gradient' #@param ['gradient', 'diameter', 'None']
    rescale = None if rescale == 'None' else rescale

    #@markdown Choose a learning rate for SFW. You can activate the learning rate scheduler which automatically multiplies the current learning rate by `lr_decrease_factor` every `lr_step_size epochs`
    learning_rate = 0.1 #@param {type:"number"}
    if args.learning_rate:
        print("learning_rate: ",args.learning_rate)
        learning_rate=(args.learning_rate)
    lr_scheduler_active = True #@param {type:"boolean"}
    if args.lr_scheduler_active:
        print("lr_scheduler_active: ",args.lr_scheduler_active)
        lr_scheduler_active=(args.lr_scheduler_active)
    lr_decrease_factor =  0.1#@param {type:"number"}
    if args.lr_decrease_factor:
        print("lr_decrease_factor: ",args.lr_decrease_factor)
        lr_decrease_factor=(args.lr_decrease_factor)
    lr_step_size = 50 #@param {type:"integer"}
    if args.lr_step_size:
        print("lr_step_size: ",args.lr_step_size)
        lr_step_size=(args.lr_step_size)

    #@markdown You can also enable retraction of the learning rate, i.e., if enabled the learning rate is increased and decreased automatically depending on the two moving averages of different length of the train loss over the epochs.
    retraction = True #@param {type:"boolean"}
    if args.retraction:
        print("retraction: ",args.retraction)
        retraction=(args.retraction)

    assert learning_rate > 0
    assert 0 <= momentum <= 1
    assert lr_decrease_factor > 0
    assert lr_step_size > 0


    # Select optimizer
    optimizer = fw.optimizers.SFW(params=model.parameters(), learning_rate=learning_rate, momentum=momentum, rescale=rescale)

    start_ts = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 150
    if args.epochs:
        print("epochs: ",args.epochs)
        epochs=args.epochs
    

    train_loader, val_loader = dataset_helpers.getData(name='cifar10', train_bs=128, test_bs=1000)


    losses = []
    loss_function = nn.CrossEntropyLoss()

    # f_w
    # initialize some necessary metrics objects
    train_loss, train_accuracy = AverageMeter(), AverageMeter()
    test_loss, test_accuracy = AverageMeter(), AverageMeter()

    if lr_scheduler_active:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_size, gamma=lr_decrease_factor)

    if retraction:
        retractionScheduler = RetractionLR(optimizer=optimizer)

    # function to reset metrics
    def reset_metrics():
        train_loss.reset()
        train_accuracy.reset()

        test_loss.reset()
        test_accuracy.reset()

    # https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951/1
    def L2reg(model, lmbda):
        reg = 0. # torch.zeros(1)
        for param in model.parameters():
            reg += torch.linalg.norm(param)**2
        return reg * lmbda

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

        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)
            
            model.zero_grad()
            outputs = model(X)
            # loss = loss_function(outputs, y)
            loss = loss_function(outputs, y) + tools.L2reg(model, 3e-4)

            loss.backward()
            optimizer.step(constraints=constraints)
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))

        if lr_scheduler_active:
            scheduler.step()  

        if retraction:
            # Learning rate retraction
            retractionScheduler.update_averages(train_loss.result())
            retractionScheduler.step()    

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
                
                for acc, metric in zip((precision, recall, f1, accuracy), 
                                    (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        tools.calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )
        
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

    model_save_name = 'cifar10_resnet_frank_wolfe_L1_norm_best.pkl'
    path = F"/content/gdrive/My Drive/{model_save_name}" 
    torch.save(best_model.state_dict(), path)

    print(losses)
    print(f"Training time: {time.time()-start_ts}s")

    print('best_accuracy: ',best_accuracy)

    import matplotlib.pyplot as plt
    plt.plot(accuracies, linestyle = 'dotted')

