import numpy as np


class Optimizer:
    def __init__(self, lr=0.01):
        pass

    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, param, dparam):
        if self.v is None:
            self.v = np.zeros(param.shape)

        self.v = self.momentum * self.v - (self.lr) * dparam
        param = param + self.v
        return param



class RMSProp(Optimizer):
    def __init__(self, lr=0.01, decay_rate=0.99, eps=1e-6):
        self.lr = lr
        self.decay_rate = decay_rate
        self.eps = eps
        self.v = None

    def step(self, param, dparam):
        if self.v is None:
            self.v = np.zeros(dparam.shape)

        self.v = self.decay_rate * self.v + (1 - self.decay_rate) * dparam ** 2
        param = param - ((self.lr * dparam) / (np.sqrt(self.v) + self.eps))
        return param


class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, param, dparam):
        self.t += 1
        if self.m is None:
            self.m = np.zeros(param.shape)
            self.v = np.zeros(param.shape)

        self.m = self.beta1 * self.m + (1 - self.beta1) * dparam
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dparam ** 2)
        mt = self.m / (1 - self.beta1 ** self.t)
        vt = self.v / (1 - self.beta2 ** self.t)
        param = param - ((self.lr * mt) / (np.sqrt(vt) + self.eps))
        return param


