"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        for ind, param in enumerate(self.params):
            self.u[ind] = (self.momentum * (self.u.get(ind) or 0) + (1-self.momentum) * (param.grad + self.weight_decay * param)).data
            param.data = ndl.Tensor(param.data - self.lr * self.u[ind], dtype = param.data.dtype)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.t += 1
        for ind, param in enumerate(self.params):
            self.m[ind] = (self.beta1 * (self.m.get(ind) or 0) + (1 - self.beta1) * param.grad).data
            self.v[ind] = (self.beta2 * (self.v.get(ind) or 0) + (1 - self.beta2) * param.grad ** 2).data
            new_m = self.m[ind] / (1 - self.beta1 ** self.t)
            new_v = self.v[ind] / (1 - self.beta2 ** self.t)
            param.data = ndl.Tensor(param.data - self.lr * new_m / (new_v ** 0.5 + self.eps), dtype=param.dtype)
        ### END YOUR SOLUTION
