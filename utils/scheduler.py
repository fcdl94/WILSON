from torch.optim.lr_scheduler import _LRScheduler, StepLR
import torch


def get_scheduler(opts, optim):
    if opts.lr_policy == 'poly':
        scheduler = PolyLR(optim, max_iters=opts.max_iters, power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=opts.lr_decay_step,
                                                    gamma=opts.lr_decay_factor)
    elif opts.lr_policy == 'warmup':
        scheduler = WarmUpPolyLR(optim, max_iters=opts.max_iters, power=opts.lr_power, start_decay=opts.start_decay)
    elif opts.lr_policy == 'none':
        scheduler = NoScheduler(optim)
    else:
        raise NotImplementedError
    return scheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.power = power
        self.max_iters = max_iters
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iters) ** self.power
                for base_lr in self.base_lrs]


class NoScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(NoScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lrs


class WarmUpPolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, start_decay=20, last_epoch=-1):
        self.power = power
        self.max_iters = max_iters
        self.start_decay = start_decay
        super(WarmUpPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.start_decay:
            return [base_lr * (1 - self.last_epoch / self.max_iters) ** self.power
                    for base_lr in self.base_lrs]
        else:
            return self.base_lrs
