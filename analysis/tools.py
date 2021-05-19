import torch.nn as nn
import math
import torch
import numpy as np
from easydict import EasyDict
import chop
import warnings
from os.path import join
warnings.filterwarnings("ignore")

import seaborn as sns

from tqdm import tqdm
import torch.nn.functional as F

from torch.nn.utils import prune
import re

import copy

import pandas as pd
import matplotlib.pyplot as plt

### CONSTANTS ###
base_path = 'Low Rank + Sparse Models'
MODEL_PATHS = {
    'old-LR+SP': join(base_path, 'Pre-bug fix/run210507_015518 -- best performing LR + S model.chkpt'),
    'old-LR7'  : join(base_path, 'Pre-bug fix/run210509_234008 -- low rank 7.chkpt'),
    'old-LR70' : join(base_path, 'Pre-bug fix/run210509_234008 -- low rank only 70.chkpt'),
    'old-SP'   : join(base_path, 'Pre-bug fix/run210509_234915 -- sparse only.chkpt'),
    'SGD'      : join(base_path, 'run210510_235114 -- SGD.chkpt'),
    'LR_old'   : join(base_path, 'run210513_020753 -- nuc 150 l1 0.chkpt'),
    'LR'       : join(base_path, 'run210514_130300 -- Best Perf Conv reshape.chkpt'),
    'SP'       : join(base_path, 'run210513_020757 -- nuc 0 l1 80.chkpt'),
    'LR+SP'    : join(base_path, 'run210513_020800 -- nuc 100 l1 40.chkpt')
}

### BASIC TOOLS ###
class LMOConv(nn.Module):
    def __init__(self, lmo_fun):
        super().__init__()
        self.lmo_fun = lmo_fun

    def forward(self, u, v):
        b, N, C, m, n = u.shape
        update_dir, max_step_size = self.lmo_fun(u.reshape(b, N*C, m*n), v.reshape(b, N*C, m*n))
        return update_dir.reshape(b, N, C, m, n), max_step_size

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def NoSequential(*args):
    """Filters Nones as no-ops when making ann.Sequential to allow for architecture toggling."""
    net = [arg for arg in args if arg is not None]
    return nn.Sequential(*net)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 with_bn=True):
        super(BasicBlock, self).__init__()

        self.with_bn = with_bn

        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        if self.with_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        if self.with_bn:
            out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        if self.with_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 with_bn=True):
        super(Bottleneck, self).__init__()

        self.with_bn = with_bn

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        if self.with_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if self.with_bn:
            self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.with_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_bn:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.with_bn:
            out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


ALPHA_ = 1


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=10, use_batch_norms=True):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 500 else BasicBlock

        self.with_bn = use_batch_norms
        self.inplanes = 16 * ALPHA_
        self.conv1 = nn.Conv2d(3,
                               16 * ALPHA_,
                               kernel_size=3,
                               padding=1,
                               bias=False)

        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(16 * ALPHA_)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,
                                       16 * ALPHA_,
                                       n,
                                       with_bn=self.with_bn)
        self.layer2 = self._make_layer(block,
                                       32 * ALPHA_,
                                       n,
                                       stride=2,
                                       with_bn=self.with_bn)
        self.layer3 = self._make_layer(block,
                                       64 * ALPHA_,
                                       n,
                                       stride=2,
                                       with_bn=self.with_bn)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * ALPHA_ * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.mode = 'normal'

    def _make_layer(self, block, planes, blocks, stride=1, with_bn=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = NoSequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion) if with_bn else None,
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                      stride, downsample, with_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, with_bn=with_bn))

        return nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)

        x = self.conv1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = self.relu(x)  # 32x32

        for i in range(len(self.layer1)):
            x = self.layer1[i](x)

        for i in range(len(self.layer2)):
            x = self.layer2[i](x)

        for i in range(len(self.layer3)):
            x = self.layer3[i](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def test(model, device, test_loader, opt, epoch, splitting=True):
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output,
                                         target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

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

device = torch.device('cpu')


def load(checkpoint_path, args=None, keep_sgd_optimizer=True):
    try:
        checkpoint = torch.load(checkpoint_path)
    except RuntimeError:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if args is None:
        args = checkpoint['args']
    
    model = ResNet(depth=args.resnet_depth, num_classes=10).to(device)

    if args.no_splitting and keep_sgd_optimizer:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
        bias_opt = None
        bias_scheduler = None
        retractionScheduler = None
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
            if 'conv' in name:
                if lmos[k]:
                    lmos[k] = LMOConv(lmos[k])

        print("Initialize optimizer...")
        optimizer = chop.stochastic.SplittingProxFW(model.parameters(),
                                                    lmo=lmos,
                                                    prox1=proxes,
                                                    prox2=proxes_lr,
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
                       if lmo is None)
        bias_opt = torch.optim.SGD(
            bias_params, lr=args.lr_bias, momentum=args.momentum)
        bias_scheduler = torch.optim.lr_scheduler.StepLR(
            bias_opt, step_size=args.lr_decay_step, gamma=args.lr_decay)

    epoch = checkpoint['epoch']
    print('Loading data')
    for name, thing in zip(['model_state_dict', 'optimizer_state_dict', 'opt_scheduler_state_dict',
                            'opt_bias_state_dict', 'bias_opt_scheduler_state_dict',
                            'retraction_scheduler_state_dict'],
                           [model, optimizer, scheduler, bias_opt,
                            bias_scheduler, retractionScheduler]):

        excl_list = ['optimizer_state_dict', 'opt_bias_state_dict', 'bias_opt_scheduler_state_dict', 'retraction_scheduler_state_dict']
        if thing is not None and not ((name in excl_list) and (keep_sgd_optimizer==False)):
            thing.load_state_dict(checkpoint[name])

    model.eval()

    return model, optimizer, scheduler, bias_opt, bias_scheduler, retractionScheduler, epoch

# This example is for the best performing model we have, on Ben's ResNet

args = EasyDict({
    'epochs': 120,
    'grad_norm': 'gradient',
    'l1_constraint_size': 23.070341001763246,
    'lipschitz': 72.92194132407582,
    'lr': 0.09510827642645917,
    'lr_bias': 0.017271296934580724,
    'lr_decay': 0.28233417011324874,
    'lr_decay_step': 34,
    'momentum': 0.9758785351079742,
    'nuc_constraint_size': 11.222103455264056,
    'seed': 1483,
    'weight_decay': 0.002110655336688122,
    'resnet_depth': 20,
    'no_splitting': True,
    'retraction': True
})


### RANK AND WEIGHT DISTRIBUTION ###
@torch.no_grad()
def get_model_optimizer_and_copies(model_filepath, keep_sgd_optimizer=True):
    # does not support args
    model, optimizer, _, _, _, _, _ = load(model_filepath, keep_sgd_optimizer=keep_sgd_optimizer)
    model_copy, optimizer_copy, _, _, _, _, _ = load(model_filepath, keep_sgd_optimizer=keep_sgd_optimizer)

    return model, optimizer, model_copy, optimizer_copy


@torch.no_grad()
def get_sparse_and_lr_components(optimizer):
    sparse_comp, lr_comp = [], []
    for p in optimizer.param_groups[0]['params']:
        sparse_comp.append(optimizer.state[p]['x'])
        lr_comp.append(optimizer.state[p]['y'])
    return sparse_comp, lr_comp


@torch.no_grad()
def svd_on_parameter(p):
    if p.ndim == 4:
        p = p.permute((2, 3, 1, 0))

    U, S, V = None, None, None
    if p.ndim >= 2:
        U, S, V = torch.svd(p)
        return U, S, V


@torch.no_grad()
def get_singular_values_from_parameter_list(parameters):
    lr_sing_values = []
    for p in parameters:
        _, S, _ = svd_on_parameter(p)
        if S is not None:
            lr_sing_values.append(S.flatten())

    lr_sing_values = torch.cat(lr_sing_values).flatten()
    return lr_sing_values


@torch.no_grad()
def analyze_lr_and_sp(model_name='LR'):
    (model, optimizer, scheduler, bias_opt, 
     bias_scheduler, retractionScheduler, epoch) = load(model_paths[model_name])

    if 'SGD' not in model_name:
        sparse_comp, lr_comp = get_sparse_and_lr_components(optimizer)
    else:
        params = [p[1].detach().cpu() for p in model.parameters()]
        sparse_comp, lr_comp = params, params

    sp_values = torch.cat([x.flatten() for x in sparse_comp]).flatten()
    lr_sing_values = get_singular_values_from_parameter_list(lr_comp)

    return sp_values, lr_sing_values


@torch.no_grad()
def plot_sp_and_lr(sp_values, lr_sing_values, model_name):
    fig, axs = plt.subplots(1, 2, figsize=(10,3))
  # s = sns.histplot(sp_values.numpy(), log_scale=True, ax=axs[0]).set_title('Sparse Component - Hist')
    s = sns.boxplot(sp_values.numpy(), ax=axs[0]).set_title(f'{model_name} - Sparse Component')
    g = sns.histplot(lr_sing_values, log_scale=True, ax=axs[1], bins=40)
    axs[1].set_title(f'{model_name} - Singular Values')
    g.set_yscale("log")
    plt.tight_layout()
    plt.savefig(f'{model_name}_analysis.pdf')


### GLOBAL WEIGHT PRUNING ###
def get_module_from_parameter(name, module, p, sensitivity=1e-2):
    if ((hasattr(module, 'weight')) and 
        ('downsample' not in name) and
        (module.weight.shape == p.shape) and
        (torch.allclose(module.weight, p))):
        return module

  # go through each of the children
    for name, child in module.named_children():
        module = get_module_from_parameter(name, child, p)
        if module is not None:
            return module


@torch.no_grad()
def sparsify_weights(model, optimizer, model_copy, optimizer_copy, fraction_sp=0., fraction_lr=0.):
    # get the parameters we want to prune
    parameters_to_prune = []
    for p in optimizer_copy.param_groups[0]['params']:
        module = get_module_from_parameter('model', model, p)
        parameters_to_prune.append((module, 'weight'))

    # prune the parameters
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=fraction_sp,
    )

    # for each parameter, get the mask
    for p in optimizer.param_groups[0]['params']:
        module = get_module_from_parameter('model', model, p)
        if module is None:
            continue
        mask = list(module.named_buffers())
        if len(mask) > 1:
            print('ERROR: Should not have more than one mask')
        mask = mask[0][1].long()

        if ('y' in optimizer.state[p]) and ('x' in optimizer.state[p]):    
            lr = optimizer.state[p]['y'].clone()
            sp = optimizer.state[p]['x'].clone()
            sp_masked = sp.data*mask
            p.copy_(sp_masked+lr)
        else:
            p.copy_(p.data*mask)

    return model


### LOW RANK PRUNING ###
@torch.no_grad()
def svd_on_parameter(p):
    if p.ndim == 4:
        p = p.permute((2, 3, 1, 0)).clone()

    U, S, VT = None, None, None
    if p.ndim >= 2:
        U, S, VT = torch.linalg.svd(p, full_matrices=False)
    return U, S, VT


@torch.no_grad()
def prune_sv_for_parameter(p, fraction_lr=0.):
    U, S, VT = svd_on_parameter(p)

    filtered_S = S.flatten()
    k = int(fraction_lr * len(filtered_S))
    _, idx = filtered_S.topk(k, largest=False)
    filtered_S.flatten()[idx] = 0.

    filtered_P = U @ torch.diag_embed(filtered_S.reshape(S.shape)) @ VT
    # filtered_P = U @ torch.diag_embed(S) @ VT
    if p.ndim == 4:
        filtered_P = filtered_P.permute((3, 2, 0, 1))
    return filtered_P


@torch.no_grad()
def sv_sparsify(model, optimizer, model_copy, optimizer_copy,
                fraction_sp=0., fraction_lr=0.):
    # deal with the case of having low_rank
    for p in optimizer.param_groups[0]['params']:
        if 'y' in optimizer_copy.state[p]:
            low_rank = optimizer_copy.state[p]['y'].clone()
            pruned_low_rank = prune_sv_for_parameter(low_rank, fraction_lr)
            sparse = optimizer_copy.state[p]['x'].clone()
            if sparse is None:
                p.copy_(pruned_low_rank)
            else:
                p.copy_(pruned_low_rank + sparse)
        else:
            pruned_low_rank = prune_sv_for_parameter(p.clone(), fraction_lr)
            p.copy_(pruned_low_rank)

    return model

