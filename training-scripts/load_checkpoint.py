import torch
import chop
from resnet import Resnet


def load(checkpoint_path, args=None):

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
        thing.load[checkpoint[name]]

    model.eval()

    return model, optimizer, scheduler, bias_opt, bias_scheduler, retractionScheduler, epoch