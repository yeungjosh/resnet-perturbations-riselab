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
import os
from tqdm import tqdm
from datetime import datetime
import copy

import torch
from torch import nn
from torch.nn import functional as F
from resnet import ResNet18
from torchvision.models import mobilenet_v2
import torchvision.models as models

import chop

import wandb


def log_opt_state(opt, epoch, splitting=True):
    singular_values, values = [], []
    singular_values_lr, values_sparse  = [], []

    for group in opt.param_groups:
        for p in group['params']:
            state = opt.state[p]
            if p.ndim == 4:
                _, s, _ = torch.linalg.svd(p.permute(2, 3, 0, 1))
            else:
                _, s, _ = torch.linalg.svd(p)
            singular_values.append(s.flatten().detach().cpu())
            values.append(p.flatten().detach().cpu())
            if splitting:
                if p.ndim == 4:
                    _, s, _ = torch.linalg.svd(state['y'].permute(2, 3, 0, 1))
                else:
                    _, s, _ = torch.linalg.svd(state['y'])
                singular_values_lr.append(s.flatten().detach().cpu())
                values_sparse.append(state['x'].flatten().detach().cpu())

    log_dict = {
        "singular_values_of_parameter": wandb.Histogram(torch.clamp(torch.cat(singular_values), 1e-8)),
        "values": wandb.Histogram(torch.cat(values)),
        "Epoch": epoch
    }

    if splitting:
        log_dict.update({
            "sparse_component": wandb.Histogram(torch.cat(values_sparse)),
            "singular_values_of_lr_component": wandb.Histogram(torch.cat(singular_values_lr)),
        })
    wandb.log(log_dict)


def get_sparsity_and_rank(opt, splitting=True):
    threshold = 1e-8
    nnzero = 0
    n_params = 0
    total_rank = 0
    max_rank = 0

    for group in opt.param_groups:
        for p in group['params']:
            if splitting:
                state = opt.state[p]
                nnzero += (~torch.isclose(state['x'], torch.zeros_like(p))).sum()
                if p.ndim == 4:
                    ranks = torch.linalg.matrix_rank(state['y'].clone().permute((2, 3, 1, 0)))

                ranks = torch.linalg.matrix_rank(state['y'])

            else:
                nnzero += (torch.abs(p) >= threshold).sum()
                if p.ndim == 4:
                    ranks = torch.linalg.matrix_rank(p.clone().permute((2, 3, 1, 0)))
                if p.ndim > 1:
                    ranks = torch.linalg.matrix_rank(p)

            n_params += p.numel()
            
            q = p.clone()
            if p.ndim == 4:
                q = p.clone().permute((2, 3, 1, 0))
            if p.ndim > 1:
                total_rank += ranks.sum()
                max_rank += min(q.shape[-2:]) * ranks.numel()

    return nnzero / n_params, total_rank / max_rank


def train(args, model, device, train_loader, opt, opt_bias, epoch, train_loss, splitting=True):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader),
                                          desc=f'Training epoch {epoch}'):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        if splitting:
            opt_bias.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # TODO: figure out subgradient descent with L1 penalty
        # if args.penalty and not splitting:
        #     loss += ell1_penalty(parameters, scale)
        loss.backward()
        if loss.isnan():
            warnings.warn("Train loss is nan.")
            break
        opt.step()
        if splitting:
            opt_bias.step()
        train_loss(loss.item(), len(target))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"Train Loss": loss.item(),
                       "Logits": F.log_softmax(output, dim=-1).cpu()})
    return loss


def test(args, model, device, test_loader, opt, epoch, splitting=True):
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
    test_accuracy = 100 * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
         test_loss, correct, len(test_loader.dataset), test_accuracy))
    
    return test_accuracy, test_loss


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

    
def init_best_dict():
    keys = ['model', 'optimizer', 'scheduler', 'bias_scheduler', 
            'retractionScheduler', 'bias_opt', 'accuracy', 
            'loss', 'epoch', 'rank', 'sparsity', 'loss']
    values = [None, None, None, None, None, None, 0, 0, 0, 0, 0, 0]
    return {keys[i]:values[i] for i in range(len(keys))}

def update_best_dict(best_dict, updates_dict):
    for k, v in updates_dict.items():
        if k in best_dict:
            best_dict[k] = copy.deepcopy(v)
    return best_dict
                                             
def log_current_training_epoch(test_loader, accuracy, 
                               loss, sparsity, rank, optimizer, epoch):
    wandb.log({
        "Test Accuracy": accuracy,
        "Test Loss": loss,
        "Sparsity": sparsity,
        "Rank": rank,
        "LR": optimizer.param_groups[0]['lr'],
        "Epoch": epoch})
                                             
def log_new_best(best_dict):
    wandb.log({
        "Best Test Accuracy": best_dict['accuracy'],
        "Best Test Loss": best_dict['loss'],
        "Best Sparsity": best_dict['sparsity'],
        "Best Rank": best_dict['rank'],
        "Best LR": best_dict['optimizer'].param_groups[0]['lr'],
        "Best Epoch": best_dict['epoch']})
                                             
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--resnet_depth', type=int, default=20)
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--lr',  default=.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lipschitz', default=1., type=float, metavar='LIP',
                        help='Lipschitz estimate, used in the prox (default: 1.)')
    parser.add_argument('--lr_bias',  default=1e-2, type=float, metavar='LRB',
                        help='learning rate')
    parser.add_argument('--lr_decay', default=.1, type=float)
    parser.add_argument('--lr_decay_step', default=50, type=int)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='W',
                        help='Optimizer weight decay (default: 5e-4)')
    parser.add_argument('--grad_norm', type=str, default='gradient',
                        help='Gradient normalization options')
    parser.add_argument('--nuc_constraint_size', type=float, default=.1,
                        help='Size of the Nuclear norm Ball constraint')
    parser.add_argument('--l1_constraint_size', type=float, default=.1,
                        help='Size of the ell-1 norm Ball constraint')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--retraction', type=bool, default=True,
                        help='enable retraction of the learning rate')
    parser.add_argument('--penalty', default=False,
                        help='if passed, uses a penalized formulation rather than constrained.')
    parser.add_argument('--no_splitting', action='store_true', default=False)
    parser.add_argument('--log_model_interval', type=int, default=10,
                        help='how many epochs to wait before saving the state of everything')

    # You can also enable retraction of the learning rate, i.e.,
    # if enabled the learning rate
    # is increased and decreased automatically depending on
    # the two moving averages of different length of the train loss
    # over the epochs.
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.lr != 'sublinear':
        args.lr = float(args.lr)

    if args.penalty in ('false', 'False', 'f', 'F', False, ''):
        args.penalty = False
    elif args.penalty in ('true', 'True', 't', 'T', True):
        args.penalty = True
    else:
        raise ValueError("args.penalty was not understood. Please use 'True' or 'False'.")

    if args.retraction in ('false', 'False', 'f', 'F', False, ''):
        args.retraction = False
    elif args.retraction in ('true', 'True', 't', 'T', True):
        args.retraction = True
    else:
        raise ValueError("args.retraction was not understood. Please use 'True' or 'False'.")

    wandb.init(project='low-rank_sparse_cifar10', config=args)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print("Loading dataset...")
    dataset = chop.utils.data.CIFAR10("~/datasets/")
    loaders = dataset.loaders(
        args.batch_size, args.test_batch_size, num_workers=0)

    print("Preparing model...")
    if args.arch == 'resnet':
        # model = ResNet(depth=args.resnet_depth, num_classes=10).to(device)
        # TODO: MAKE THIS BETTER
        model = ResNet18(num_classes=10).to(device)
    else:
        model = mobilenet_v2(pretrained=False, num_classes=10).to(device)

    if not os.path.exists('models/'):
        os.mkdir('models/')
    LOGPATH = "models/run.chkpt"
    LOGPATH = get_model_logpath(LOGPATH, args)

    if args.no_splitting:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
        # TODO: figure out subgradient descent on penalized L1 / tracenorm here!
        bias_opt = None
        bias_scheduler = None
        retractionScheduler = None
    else:
        print("Make constraints/penalties...")
        constraints_sparsity = chop.constraints.make_model_constraints(model,
                                                                       ord=1,
                                                                       value=args.l1_constraint_size,
                                                                       constrain_bias=False,
                                                                       penalty=args.penalty)
        constraints_low_rank = chop.constraints.make_model_constraints(model,
                                                                       ord='nuc',
                                                                       value=args.nuc_constraint_size,
                                                                       constrain_bias=False,
                                                                       penalty=args.penalty)
        proxes = [constraint.prox if constraint else None
                  for constraint in constraints_sparsity]
        lmos = [constraint.lmo if constraint else None
                for constraint in constraints_low_rank]

        proxes_lr = [constraint.prox if constraint else None
                     for constraint in constraints_low_rank]

        # Unconstrain downsampling layers
        for k, (name, param) in enumerate(model.named_parameters()):
            if 'conv' in name or 'downsample' in name:
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
                                                    normalization=args.grad_norm,
                                                    generalized_lmo=args.penalty)

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

    print("Training...")

    # initialize some necessary metrics objects
    train_loss, train_accuracy = AverageMeter(), AverageMeter()
    test_loss, test_accuracy = AverageMeter(), AverageMeter()
    
    best_dict = init_best_dict()
    splitting = not args.no_splitting

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, loaders.train, optimizer,
                     bias_opt, epoch, train_loss, not args.no_splitting)
        if loss.isnan():
            break
        accuracy, loss = test(args, model, device, loaders.test,
             optimizer, epoch, splitting)
        sparsity, rank = get_sparsity_and_rank(optimizer, splitting)
        log_current_training_epoch(loaders.test, accuracy, 
                                   loss, sparsity, rank, optimizer, epoch)
                                             
        if accuracy > best_dict['accuracy']:
            best_dict = update_best_dict(best_dict,
                                         {'model':model,
                                          'optimizer': optimizer,
                                          'scheduler': scheduler, 
                                          'bias_scheduler': bias_scheduler,
                                          'retractionScheduler': retractionScheduler,
                                          'bias_opt': bias_opt,
                                          'accuracy': accuracy,
                                          'loss': loss,
                                          'sparsity': sparsity,
                                          'rank': rank
                                         })
            log_new_best(best_dict)

        log_opt_state(optimizer, epoch, splitting)
                                  
        scheduler.step()
        if not args.no_splitting:
            bias_scheduler.step()

        if args.retraction and not args.no_splitting:
            # Learning rate retraction
            retractionScheduler.update_averages(train_loss.result())
            retractionScheduler.step()

        if epoch % args.log_model_interval == 1:
            torch.save(best_dict, LOGPATH)


if __name__ == '__main__':
    main()
