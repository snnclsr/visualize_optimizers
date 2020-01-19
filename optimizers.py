import numpy as np


class Optimizer:

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, param, dparam):
        pass


class SGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0.0):
        super().__init__(lr=lr)
        self.momentum = momentum
        self.v = None

    def step(self, param, dparam):
        if self.v is None:
            self.v = np.zeros_like(dparam)

        self.v = self.momentum * self.v + self.lr * dparam
        param = param - self.v
        return param


class Adagrad(Optimizer):

    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__(lr=lr)
        self.eps = eps
        self.squared_grad = None

    def step(self, param, dparam):
        if self.squared_grad is None:
            self.squared_grad = np.zeros_like(dparam)

        self.squared_grad = self.squared_grad + dparam ** 2
        param = param - (self.lr * dparam / (np.sqrt(self.squared_grad) + self.eps))
        return param


class Adadelta(Optimizer):

    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        super().__init__(lr=lr)
        self.rho = rho
        self.eps = eps
        self.acc_update = None
        self.squared_avg = None

    def step(self, param, dparam):
        if self.acc_update is None:
            self.acc_update = np.zeros_like(dparam)
            self.squared_avg = np.zeros_like(dparam)

        # Accumulated gradient
        self.squared_avg = self.rho * self.squared_avg + (1 - self.rho) * dparam * dparam
        std = np.sqrt(self.squared_avg + self.eps)
        delta = (np.sqrt(self.acc_update + self.eps) / std) * dparam
        param = param - delta
        self.acc_update = self.rho * self.acc_update + (1 - self.rho) * delta * delta
        return param


class RMSProp(Optimizer):

    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        super().__init__(lr=lr)
        self.rho = rho
        self.eps = eps
        self.squared_avg = None

    def step(self, param, dparam):
        if self.squared_avg is None:
            self.squared_avg = np.zeros_like(dparam)

        self.squared_avg = self.rho * self.squared_avg + (1 - self.rho) * dparam * dparam
        param = param - ((self.lr * dparam) / (np.sqrt(self.squared_avg) + self.eps))
        return param


class Adam(Optimizer):

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, param, dparam):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(dparam)
            self.v = np.zeros_like(dparam)

        self.m = self.beta1 * self.m + (1 - self.beta1) * dparam
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dparam * dparam)
        mt = self.m / (1 - self.beta1 ** self.t)
        vt = self.v / (1 - self.beta2 ** self.t)
        param = param - ((self.lr * mt) / (np.sqrt(vt) + self.eps))
        return param


class AdaMax(Optimizer):

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.u = None

    def step(self, param, dparam):

        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(dparam)
            self.u = np.zeros_like(dparam)

        self.m = self.beta1 * self.m + (1 - self.beta1) * dparam
        self.u = np.maximum((self.beta2 * self.u), np.abs(dparam))
        mt = self.m / (1 - self.beta1 ** self.t)
        param = param - (self.lr * mt) / (self.u + self.eps)
        return param
