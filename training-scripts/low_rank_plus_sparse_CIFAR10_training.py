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
import copy

import torch
from torch import nn
from torch.nn import functional as F
from resnet import ResNet18
from torchvision.models import mobilenet_v2
import torchvision.models as models

import chop

import wandb

from tools import RetractionLR, RunningAverage, AverageMeter, LMOConv, get_model_logpath 


ACC_THRESHOLDS = {
    'cifar' : .4,
    'imagenet' : .15
}


def log_opt_state(opt, epoch, splitting=True):
    singular_values, values = [], []
    singular_values_lr, values_sparse  = [], []

    for group in opt.param_groups:
        for p in group['params']:
            state = opt.state[p]
            if p.ndim == 4:
                try:
                    _, s, _ = torch.linalg.svd(p.permute(2, 3, 0, 1))
                except:
                    s = torch.zeros(1)
            elif p.ndim == 2:
                try:
                    _, s, _ = torch.linalg.svd(p)
                except:
                    s = torch.zeros(1)
            singular_values.append(s.flatten().detach().cpu())
            values.append(p.flatten().detach().cpu())
            if splitting:
                if p.ndim == 4:
                    try:
                        _, s, _ = torch.linalg.svd(state['y'].permute(2, 3, 0, 1))
                    except:
                        s = torch.zeros(1)
                elif p.ndim == 2:
                    try:
                        _, s, _ = torch.linalg.svd(state['y'])
                    except:
                        s = torch.zeros(1)
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
                    try:
                        ranks = torch.linalg.matrix_rank(state['y'].clone().permute((2, 3, 1, 0)))
                    except:
                        ranks = torch.zeros(1)
                elif p.ndim > 1:
                    try:
                        ranks = torch.linalg.matrix_rank(state['y'])
                    except:
                        ranks = torch.zeros(1)

            else:
                nnzero += (torch.abs(p) >= threshold).sum()
                if p.ndim == 4:
                    try:
                        ranks = torch.linalg.matrix_rank(p.clone().permute((2, 3, 1, 0)))
                    except:
                        ranks = torch.zeros(1)
                elif p.ndim > 1:
                    try:
                        ranks = torch.linalg.matrix_rank(p)
                    except:
                        ranks = torch.zeros(1)

            n_params += p.numel()
            
            q = p.clone()
            if p.ndim == 4:
                q = p.clone().permute((2, 3, 1, 0))
            elif p.ndim > 1:
                total_rank += ranks.sum()
                max_rank += min(q.shape[-2:]) * ranks.numel()
            elif p.ndim == 1:
                total_rank = 0
                max_rank = 1

    return nnzero / n_params, total_rank / max_rank


def train(args, model, device, train_loader, opt, opt_bias, epoch,
          train_loss, sparse_penalties=None, low_rank_penalties=None,
          splitting=True):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader),
                                          desc=f'Training epoch {epoch}'):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        if splitting:
            opt_bias.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        if args.optimizer=='sgd':
            for param, sp_penalty, lr_penalty in zip(model.parameters(), sparse_penalties, low_rank_penalties):
                if args.isL1Penalty and sp_penalty and sp_penalty.alpha > 0.:
                    loss += sp_penalty(param).sum()
                if args.isNucPenalty and lr_penalty and lr_penalty.alpha > 0.:
                    q = param.clone()
                    if param.ndim == 4:
                        q = param.permute(2, 3, 0, 1)
                    loss += lr_penalty(q).sum()
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
    # TODO: ADD THRESHOLDING HERE?
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


def init_best_dict(args):
    keys = ['args', 'model_state_dict', 'optimizer_state_dict', 
            'opt_scheduler_state_dict', 'bias_opt_scheduler_state_dict', 
            'retraction_scheduler_state_dict', 'opt_bias_state_dict', 
            'accuracy', 'loss', 'epoch', 'rank', 'sparsity', 'loss']
    values = [args, None, None, None, None, None, None, 0, 0, 0, 0, 0, 0]
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
                                             
def log_new_best(best_dict, optimizer):
    wandb.log({
        "Best Test Accuracy": best_dict['accuracy'],
        "Best Test Loss": best_dict['loss'],
        "Best Sparsity": best_dict['sparsity'],
        "Best Rank": best_dict['rank'],
        "Best LR": optimizer.param_groups[0]['lr'],
        "Best Epoch": best_dict['epoch']})

def make_str_arg_to_bool(args, attribute):
    str_attribute = getattr(args, attribute)
    if str_attribute in ('false', 'False', 'f', 'F', False, ''):
        setattr(args, attribute, False)
    elif str_attribute in ('true', 'True', 't', 'T', True):
        setattr(args, attribute, True)
    else:
        raise ValueError(f"args.{attribute} was not understood. Please use 'True' or 'False'.")
    return args

    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--dataset', type=str, default='cifar',
                        choices=['cifar', 'imagenet'])
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet20', 'resnet50',
                                 'resnet50_2x', 'resnet50_4x', 'squeezenet'])
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training')
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
    parser.add_argument('--nuc_scale', type=float, default=1e-4,
                        help='Size of the Nuclear norm Ball constraint')
    parser.add_argument('--l1_scale', type=float, default=1e-4,
                        help='Size of the ell-1 norm Ball constraint')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--retraction', type=str, default=False,
                        help='enable retraction of the learning rate')
    parser.add_argument('--isL1Penalty', type=str, default='True',
                        help='if passed, uses a penalized formulation rather than constrained.')
    parser.add_argument('--isNucPenalty', type=str, default='True',
                        help='if passed, uses a penalized formulation rather than constrained.')
    parser.add_argument('--lmo', type=str, default='l1',
                        choices=['l1', 'nuc'],
                        help='which component uses LMO.')
    parser.add_argument('--optimizer', type=str, default='splitting',
                        choices=['sgd', 'fw', 'splitting'])
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

    attributes = ['isL1Penalty', 'isNucPenalty', 'retraction']
    for att in attributes:
        args = make_str_arg_to_bool(args, att)

    wandb.init(project='low-rank_sparse_cifar10', config=args)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print("Loading dataset...")
    dataset = chop.utils.data.CIFAR10("~/datasets/")
    loaders = dataset.loaders(
        args.batch_size, args.test_batch_size, num_workers=0)

    # TODO: ADD ALL ARCHITECTURES
    print("Preparing model...")
    if args.arch == 'resnet18':
        # model = ResNet(depth=args.resnet_depth, num_classes=10).to(device)
        # TODO: MAKE THIS BETTER
        model = ResNet18(num_classes=10).to(device)
    else:
        model = mobilenet_v2(pretrained=False, num_classes=10).to(device)

    print(model)
    if not os.path.exists('models/'):
        os.mkdir('models/')
    LOGPATH = "models/run.chkpt"
    LOGPATH = get_model_logpath(LOGPATH, args)

    print("Make constraints/penalties...")
    constraints_sparsity = chop.constraints.make_model_constraints(model,
                                                                   ord=1,
                                                                   value=args.l1_scale,
                                                                   constrain_bias=False,
                                                                   penalty=args.isL1Penalty)
    constraints_low_rank = chop.constraints.make_model_constraints(model,
                                                                   ord='nuc',
                                                                   value=args.nuc_scale,
                                                                   constrain_bias=False,
                                                                   penalty=args.isNucPenalty)
    proxes_sparse = [constraint.prox if constraint else None
                     for constraint in constraints_sparsity]
    lmos_sparse = [constraint.lmo if constraint else None
                   for constraint in constraints_sparsity]

    lmos_low_rank = [constraint.lmo if constraint else None
               for constraint in constraints_low_rank]
    proxes_low_rank = [constraint.prox if constraint else None
                    for constraint in constraints_low_rank]

    if args.lmo == 'l1':
        lmos = lmos_sparse
        proxes = proxes_low_rank
        proxes_y = proxes_sparse
    else:
        lmos = lmos_low_rank
        proxes = proxes_sparse
        proxes_y = proxes_low_rank
    

    # Unconstrain downsampling layers
    for k, (name, param) in enumerate(model.named_parameters()):
        if 'conv' in name or 'shortcut' in name:
            if lmos[k]:
                lmos[k] = LMOConv(lmos[k])

    print("Initialize optimizer...")
    if args.optimizer == 'fw':
        args.no_splitting = True
        optimizer = chop.stochastic.FrankWolfe(model.parameters(),
                                                lmo=lmos,
                                                prox=proxes_y,
                                                lr=args.lr,
                                                momentum=args.momentum,
                                                weight_decay=args.weight_decay,
                                                normalization=args.grad_norm)


        bias_params = (param for param, lmo in zip(model.parameters(), lmos)
                       if lmo is None)
        bias_opt = torch.optim.SGD(
            bias_params, lr=args.lr_bias, momentum=args.momentum)
        bias_scheduler = torch.optim.lr_scheduler.StepLR(
            bias_opt, step_size=args.lr_decay_step, gamma=args.lr_decay)

        if args.retraction:
            retractionScheduler = RetractionLR(optimizer=optimizer)
        else:
            retractionScheduler = None

    elif args.optimizer == 'splitting':
        args.no_splitting = False
        optimizer = chop.stochastic.SplittingProxFW(model.parameters(),
                                                    lmo=lmos,
                                                    prox1=proxes,
                                                    prox2=proxes_y,
                                                    lr=args.lr,
                                                    lipschitz=args.lipschitz,
                                                    momentum=args.momentum,
                                                    weight_decay=args.weight_decay,
                                                    normalization=args.grad_norm,
                                                    generalized_lmo=False)

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

    elif args.optimizer == 'sgd':
        args.no_splitting = True
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)
        bias_opt = None
        bias_scheduler = None
        retractionScheduler = None

    else:
        raise ValueError("Unknown optimizer")

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

    print("Training...")

    # initialize some necessary metrics objects
    train_loss, train_accuracy = AverageMeter(), AverageMeter()
    test_loss, test_accuracy = AverageMeter(), AverageMeter()
    
    best_dict = init_best_dict(args)
    splitting = not args.no_splitting

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, loaders.train, optimizer,
                     bias_opt, epoch, train_loss, constraints_sparsity, 
                     constraints_low_rank, splitting)
        if loss.isnan():
            break
        accuracy, loss = test(args, model, device, loaders.test,
             optimizer, epoch, splitting)
        sparsity, rank = get_sparsity_and_rank(optimizer, splitting)
        log_current_training_epoch(loaders.test, accuracy, 
                                   loss, sparsity, rank, optimizer, epoch)

        retractionSchedulerStateDict = None 
        if hasattr(retractionScheduler, 'state_dict'):
            retractionSchedulerStateDict = retractionScheduler.state_dict()
        biasOptSchedulerStateDict = None 
        if hasattr(bias_scheduler, 'state_dict'):
            biasOptSchedulerStateDict = bias_scheduler.state_dict()
        biasOptStateDict = None 
        if hasattr(bias_opt, 'state_dict'):
            biasOptStateDict = bias_opt.state_dict()
        
        if accuracy > best_dict['accuracy']:
            best_dict = update_best_dict(best_dict,
                                         {'model_state_dict':model.state_dict(),
                                          'optimizer_state_dict': optimizer.state_dict(),
                                          'opt_scheduler_state_dict': scheduler.state_dict(), 
                                          'bias_opt_scheduler_state_dict': biasOptSchedulerStateDict,
                                          'retraction_scheduler_state_dict': retractionSchedulerStateDict,
                                          'opt_bias_state_dict': biasOptStateDict,
                                          'accuracy': accuracy,
                                          'loss': loss,
                                          'sparsity': sparsity,
                                          'rank': rank,
                                          'epoch':epoch
                                         })
            log_new_best(best_dict, optimizer)

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
