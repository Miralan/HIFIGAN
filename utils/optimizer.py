import math
import torch
import numpy as np


class Optimizer(object):
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self,
                 optimizer,
                 init_lr,
                 epoch=0,
                 decay_learning_rate=0.999):

        self.optimizer = optimizer
        self.lr = init_lr
        self.epoch = epoch
        self.decay_learning_rate = decay_learning_rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step_and_update_lr(self, epoch):
        self.update_learning_rate(epoch)
        self.optimizer.step()

    def get_lr_scale(self, epoch):
        if self.epoch != epoch:
           lr_scale = self.decay_learning_rate
           self.epoch = epoch
        else:
           lr_scale = 1
        return lr_scale

    def update_learning_rate(self, epoch):

        self.lr = self.lr * self.get_lr_scale(epoch)
        self.lr = np.maximum(1e-6, self.lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

