
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
from datetime import datetime
import time
import os

import torch

from torch import nn, optim
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from copy import copy, deepcopy
import numpy as np


def get_model_logpath(path, args):
    filename, extension = os.path.splitext(path)
    time = datetime.strftime(datetime.now(), "%y%m%d_%H%M%S")
    sp = args.l1_constraint_size
    lr = args.nuc_constraint_size
    depth = args.resnet_depth
    if args.arch != 'resnet':
        depth = ''
    path = f'{filename}{args.arch}{depth}_lr:{lr}_sp:{sp}_{time}{extension}'
    return path


class LMOConv(nn.Module):
    def __init__(self, lmo_fun):
        super().__init__()
        self.lmo_fun = lmo_fun

    def forward(self, u, v):
        b, N, C, m, n = u.shape
        # print(u.shape, v.shape)
        update_dir, max_step_size = self.lmo_fun(
            u.permute(0, 3, 4, 1, 2), v.permute(0, 3, 4, 1, 2))
        return update_dir.reshape(b, N, C, m, n), max_step_size


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


class RetractionLR(optim.lr_scheduler._LRScheduler):
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

        assert (0 <= self.retraction_factor <
                1), "Retraction factor must be in [0, 1[."
        assert (0 <= self.lowerBound < self.upperBound <=
                1), "Bounds must be in [0, 1]"
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


# add input model_save_name
def save_model(input_model, model_save_name):
  path = F"/content/gdrive/My Drive/{model_save_name}" 
  torch.save(input_model, path)

def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

def exp_lr_scheduler(epoch, optimizer, strategy='normal', decay_eff=0.1, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if strategy == 'normal':
        if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
            print('New learning rate is: ', param_group['lr'])
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')
    return optimizer

# https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951/1
def L2reg(model, lmbda):
    reg = 0. # torch.zeros(1)
    for param in model.parameters():
        reg += torch.linalg.norm(param)**2
    return reg * lmbda


import torch
import chop
from resnet import ResNet


def load(checkpoint_path, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path)
    if args is None:
        args = checkpoint['args']
    
    model = ResNet(depth=args.resnet_depth, num_classes=10).to(device)

    if args.no_splitting:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
        bias_opt = None
        bias_scheduler = None
    else:
        print("Make constraints...")
        constraints_sparsity = chop.constraints.make_model_constraints(model,
                                                                       ord=1,
                                                                       value=args.l1_constraint_size,
                                                                       constrain_bias=False)
        constraints_low_rank = chop.constraints.make_model_constraints(model,
                                                                       ord='nuc',
                                                                       value=args.nuc_constraint_size,
                                                                       constrain_bias=False)
        proxes = [constraint.prox if constraint else None
                  for constraint in constraints_sparsity]
        lmos = [constraint.lmo if constraint else None
                for constraint in constraints_low_rank]

        proxes_lr = [constraint.prox if constraint else None
                     for constraint in constraints_low_rank]

        # Unconstrain downsampling layers
        for k, (name, param) in enumerate(model.named_parameters()):
            if 'downsample' in name:
                try:
                    *_, m, n = param.shape
                except ValueError:
                    continue
                if m == n == 1:
                    proxes[k], lmos[k], proxes_lr[k] = None, None, None

        print("Initialize optimizer...")
        optimizer = chop.stochastic.SplittingProxFW(model.parameters(), lmos,
                                                    proxes,
                                                    lr=args.lr,
                                                    lipschitz=args.lipschitz,
                                                    momentum=args.momentum,
                                                    weight_decay=args.weight_decay,
                                                    normalization=args.grad_norm)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

        if args.retraction:
            retractionScheduler = RetractionLR(optimizer=optimizer)
        else:
            retractionScheduler = None

        bias_params = (param for param, lmo in zip(model.parameters(), lmos)
                       if lmo is not None)
        bias_opt = torch.optim.SGD(
            bias_params, lr=args.lr_bias, momentum=args.momentum)
        bias_scheduler = torch.optim.lr_scheduler.StepLR(
            bias_opt, step_size=args.lr_decay_step, gamma=args.lr_decay)

    epoch = checkpoint['epoch']
    for name, thing in zip(['model_state_dict', 'optimizer_state_dict', 'opt_scheduler_state_dict',
                            'opt_bias_state_dict', 'bias_opt_scheduler_state_dict',
                            'retraction_scheduler_state_dict'],
                            [model, optimizer, scheduler, bias_opt,
                             bias_scheduler, retractionScheduler]):
        thing.load_state_dict(checkpoint[name])

    model.eval()

    return model, optimizer, scheduler, bias_opt, bias_scheduler, retractionScheduler, epoch
