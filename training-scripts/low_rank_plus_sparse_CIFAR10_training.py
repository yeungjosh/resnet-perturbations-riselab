"""
Constrained Neural Network Training.
======================================
Trains a ResNet model on CIFAR10 using constraints on the weights.
This example is inspired by the official PyTorch MNIST example, which
can be found [here](https://github.com/pytorch/examples/blob/master/mnist/main.py).
"""
from __future__ import print_function
import warnings
import torch.nn.functional as F
import torch.nn as nn
import argparse

from tqdm import tqdm


import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

import chop

import wandb


# Hyperparam setup
default_config = {
    'lr': 1e-10,
    # 'lr_prox': 'sublinear',
    'momentum': .9,
    'weight_decay': 1e-5,
    'lr_bias': 0.01,
    'grad_norm': 'none',
    'l1_constraint_size': 5e2,
    'nuc_constraint_size': 5e2,
    'epochs': 2,
    'seed': 0
}

wandb.init(project='low-rank_sparse_cifar10', config=default_config)
config = wandb.config


def get_sparsity_and_rank(opt):
    nnzero = 0
    n_params = 0
    total_rank = 0
    max_rank = 0

    for group in opt.param_groups:
        for p in group['params']:
            state = opt.state[p]
            nnzero += (state['x'] != 0).sum()
            n_params += p.numel()
            ranks = torch.linalg.matrix_rank(state['y'])
            total_rank += ranks.sum()
            max_rank += min(p.shape) * ranks.numel()

    return nnzero / n_params, total_rank / max_rank


def train(args, model, device, train_loader, opt, opt_bias, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader),
                                          desc=f'Training epoch {epoch}'):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        opt_bias.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        if loss.isnan():
            warnings.warn("Train loss is nan.")
            break
        opt.step()
        opt_bias.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"Train Loss": loss.item(),
                       "Logits": F.log_softmax(output, dim=-1).cpu()})


def test(args, model, device, test_loader, opt, epoch):
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
            example_images.append(wandb.Image(
                data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    sparsity, rank = get_sparsity_and_rank(opt)
    wandb.log({
        # "Examples": example_images,
        "Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss,
        "Sparsity": sparsity,
        "Rank": rank,
        "LR": opt.param_groups[0]['lr'],
        "Epoch": epoch})


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


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr',  default=1e-2, metavar='LR',
                        help='learning rate (default: "sublinear")')
    parser.add_argument('--lipschitz', default=1., type=float, metavar='LIP',
                        help='Lipschitz estimate, used in the prox (default: 1.)')
    parser.add_argument('--lr_bias',  default=1e-2, type=float, metavar='LRB',
                        help='learning rate (default: "sublinear")')
    parser.add_argument('--lr_decay', default=.1, type=float)
    parser.add_argument('--lr_decay_step', default=25, type=int)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='W',
                        help='Optimizer weight decay (default: 0.)')
    parser.add_argument('--grad_norm', type=str, default='gradient',
                        help='Gradient normalization options')
    parser.add_argument('--nuc_constraint_size', type=float, default=1e3,
                        help='Size of the Nuclear norm Ball constraint')
    parser.add_argument('--l1_constraint_size', type=float, default=1e3,
                        help='Size of the ell-1 norm Ball constraint')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--retraction', type=bool, default=True,
                        help='enable retraction of the learning rate')

    # You can also enable retraction of the learning rate, i.e.,
    # if enabled the learning rate
    # is increased and decreased automatically depending on
    # the two moving averages of different length of the train loss
    # over the epochs.
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.lr != 'sublinear':
        args.lr = float(args.lr)

    wandb.config.update(args, allow_val_change=True)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print("Loading dataset...")
    dataset = chop.utils.data.CIFAR10("~/datasets/")
    loaders = dataset.loaders(
        args.batch_size, args.test_batch_size, num_workers=0)

    print("Preparing model...")
    model = resnet18(num_classes=10).to(device)
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

    # initialize some necessary metrics objects
    train_loss, train_accuracy = AverageMeter(), AverageMeter()
    test_loss, test_accuracy = AverageMeter(), AverageMeter()

    # function to reset metrics
    def reset_metrics():
        train_loss.reset()
        train_accuracy.reset()

        test_loss.reset()
        test_accuracy.reset()

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

    bias_params = (param for param, lmo in zip(model.parameters(), lmos)
                   if lmo is not None)
    bias_opt = torch.optim.SGD(bias_params, lr=args.lr_bias, momentum=.9)
    bias_scheduler = torch.optim.lr_scheduler.StepLR(
        bias_opt, step_size=args.lr_decay_step, gamma=args.lr_decay)

    # wandb.watch(model, log_freq=1, log='all')

    print("Training...")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, loaders.train, optimizer, bias_opt, epoch)
        test(args, model, device, loaders.test, optimizer, epoch)
        scheduler.step()
        bias_scheduler.step()

        if args.retraction:
            # Learning rate retraction
            retractionScheduler.update_averages(train_loss.result())
            retractionScheduler.step()


if __name__ == '__main__':
    main()
