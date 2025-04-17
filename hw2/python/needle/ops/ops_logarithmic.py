from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        max_z = array_api.max(Z, axis=1, keepdims=True)
        z_minus_max_z = Z - array_api.broadcast_to(max_z, Z.shape)
        sum_exp_z_minus_max_z = array_api.broadcast_to(array_api.sum(array_api.exp(z_minus_max_z), axis=1, keepdims=True), Z.shape)
        return  z_minus_max_z - array_api.log(sum_exp_z_minus_max_z)

       ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        z = node.inputs[0]
        tiled_shape = (z.shape[0], z.shape[1], z.shape[1])
        exp_z = exp(z)
        sum_exp_z = exp_z.sum(axes=1).reshape([z.shape[0], 1]).broadcast_to(z.shape)
        tiled_normalized_expz = (exp_z / sum_exp_z).reshape((z.shape[0], z.shape[1], 1)).broadcast_to(tiled_shape)
        tiled_eyes = Tensor(array_api.identity(z.shape[1])).broadcast_to(tiled_shape)
        tiled_out_grad = out_grad.reshape((z.shape[0], 1, z.shape[1])).broadcast_to(tiled_shape)
        return ( tiled_out_grad * (tiled_eyes - tiled_normalized_expz)).sum(axes=2).reshape(z.shape)

        ## END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_ = array_api.max(max_z, axis=self.axes)
        return array_api.log(
            array_api.sum(array_api.exp(Z - array_api.broadcast_to(max_z, Z.shape)), axis=self.axes)) + max_z_
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        z = node.inputs[0]
        axes = self.axes or range(len(z.shape))
        new_shape = [1 if i in axes else dim for i, dim in enumerate(z.shape)]
        max_z = z.realize_cached_data().max(axis=self.axes)
        exp_z = exp(z - Tensor(max_z).reshape(new_shape).broadcast_to(z.shape))
        return exp_z * (out_grad / exp_z.sum(axes= self.axes)).reshape(tuple(new_shape)).broadcast_to(z.shape)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

