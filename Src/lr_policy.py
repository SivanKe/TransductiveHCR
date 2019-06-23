import numpy as np


class StepLR(object):
    def __init__(self, optimizer, step_size=1000, max_iter=10000):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.step_size = step_size
        self.last_iter = -1
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, last_iter=None):
        if last_iter is not None:
            self.last_iter = last_iter
        if self.last_iter + 1 == self.max_iter:
            self.last_iter = -1
        self.last_iter = (self.last_iter + 1) % self.max_iter
        for ids, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[ids] * 0.1 ** ( self.last_iter // self.step_size )

class StepLrOld(object):
    def __init__(self, optimizer, step_size=1000, max_iter=10000):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.step_size = step_size
        self.last_iter = -1
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, last_iter=None):
        if last_iter is not None:
            self.last_iter = last_iter
        if self.last_iter + 1 == self.max_iter:
            self.last_iter = -1
        self.last_iter = (self.last_iter + 1) % self.max_iter
        for ids, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[ids] * 0.1 ** ( self.last_iter // self.step_size )

class DannLR(object):
    def __init__(self, optimizer, step_size=1000, max_iter=120000):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.step_size = step_size
        self.last_iter = -1
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def update(self, p):
        self.last_iter += 1
        lr_factor = 1./(1. + 10 * p)**0.75
        for ids, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[ids] * lr_factor
