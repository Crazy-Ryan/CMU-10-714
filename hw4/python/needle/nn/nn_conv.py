"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        fan_in = in_channels * kernel_size ** 2
        self.weight = Parameter(
            init.kaiming_uniform(fan_in=fan_in, fan_out=None,
                                 shape=(kernel_size, kernel_size, in_channels, out_channels)), device=device)
        if bias:
            interval = 1 / np.sqrt(fan_in).astype(dtype)
            self.bias = Parameter(init.rand( out_channels, low=-interval, high=interval, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        res = ops.conv(x.transpose((1,2)).transpose((2,3)), self.weight, padding=(self.kernel_size - 1)//2 , stride=self.stride)
        if self.bias:
            res += self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(res.shape)
        return res.transpose((2,3)).transpose((1,2))
        ### END YOUR SOLUTION