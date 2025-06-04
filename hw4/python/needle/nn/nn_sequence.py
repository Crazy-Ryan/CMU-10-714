"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return (ops.exp(-x) + 1) ** -1
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        bound = 1 / hidden_size ** 0.5
        self.nonlinearity = ops.tanh if nonlinearity == 'tanh' else ops.relu
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low = -bound, high = bound, device=device))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low = -bound, high = bound, device=device))
        self.bias = bias
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low = -bound, high = bound, device=device))
            self.bias_hh = Parameter(init.rand(hidden_size, low = -bound, high = bound, device=device))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        res = X @ self.W_ih
        if h:
            res += h @ self.W_hh
        if self.bias:
            res += (self.bias_ih + self.bias_hh).reshape((1, res.shape[1])).broadcast_to(res.shape)
        return self.nonlinearity(res)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.rnn_cells = [RNNCell(input_size if ind == 0 else hidden_size, hidden_size, bias, nonlinearity, device ) for ind in range(num_layers) ]
        self.num_layers = num_layers
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        h_seq = list(ops.split(h0, axis=0)) if h0 is not None else [None] * self.num_layers
        X_splitted = ops.split(X, axis=0)
        output = []
        for seq_ind in range(X.shape[0]):
            for layer_ind, cell in enumerate(self.rnn_cells):
                input = X_splitted[seq_ind]  if layer_ind == 0 else h_seq[layer_ind - 1]
                h_seq[layer_ind] = self.rnn_cells[layer_ind](input, h_seq[layer_ind])
            output.append(h_seq[-1])
        return ops.stack(output, axis=0), ops.stack(h_seq, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        bound = 1 / hidden_size ** 0.5
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low = -bound, high = bound, device=device))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low = -bound, high = bound, device=device))
        self.bias = bias
        self.hidden_size = hidden_size
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low = -bound, high = bound, device=device))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low = -bound, high = bound, device=device))
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        res = X @ self.W_ih
        if h:
            h0, c0 = h
            res += h0 @ self.W_hh
        if self.bias:
            res += (self.bias_ih + self.bias_hh).reshape((1, res.shape[1])).broadcast_to(res.shape)
        res_arr = list(ops.split(res, axis=1))
        i_g = self.sigmoid(ops.stack(res_arr[:self.hidden_size], axis=1))
        f_g = self.sigmoid(ops.stack(res_arr[self.hidden_size: 2*self.hidden_size], axis=1))
        g_g = ops.tanh(ops.stack(res_arr[2*self.hidden_size: 3*self.hidden_size], axis=1))
        o_g = self.sigmoid(ops.stack(res_arr[3*self.hidden_size:], axis=1))
        c_prime = i_g * g_g
        if h:
            c_prime += f_g * c0
        h_prime = o_g * ops.tanh(c_prime)
        return h_prime, c_prime
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.num_layers = num_layers
        self.lstm_cells = [LSTMCell(input_size if ind == 0 else hidden_size, hidden_size, bias, device ) for ind in range(num_layers) ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if h:
            h_seq = list(ops.split(h[0], axis=0))
            c_seq = list(ops.split(h[1], axis=0))
        else:
            h_seq = [None] * self.num_layers
            c_seq = [None] * self.num_layers
        X_splitted = ops.split(X, axis=0)
        output = []
        for seq_ind in range(X.shape[0]):
            for layer_ind, cell in enumerate(self.lstm_cells):
                input = X_splitted[seq_ind] if layer_ind == 0 else h_seq[layer_ind - 1]
                h = (h_seq[layer_ind], c_seq[layer_ind]) if h_seq[layer_ind] is not None else None
                h_seq[layer_ind], c_seq[layer_ind] = self.lstm_cells[layer_ind](input, h)
            output.append(h_seq[-1])
        return ops.stack(output, axis=0), (ops.stack(h_seq, axis=0), ops.stack(c_seq, axis=0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION