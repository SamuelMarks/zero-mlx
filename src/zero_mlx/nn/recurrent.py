"""Recurrent neural network layers."""

from typing import Callable, Optional, Union, Tuple

import numpy as np

from .. import array
from .layers import Module


class RNN(Module):
    """An Elman recurrent layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: Optional[bool] = True,
        nonlinearity: Optional[Callable] = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        self.weight_ih = np.random.randn(hidden_size, input_size)
        self.weight_hh = np.random.randn(hidden_size, hidden_size)
        if self.bias:
            self.bias_ih = np.zeros(hidden_size)
            self.bias_hh = np.zeros(hidden_size)

    def __call__(
        self, x: array, h0: Optional[array] = None
    ) -> Union[array, Tuple[array, array]]:
        """Forward pass."""
        d = x.data
        if d.ndim == 2:
            d = d[np.newaxis, ...]
        N, L, _ = d.shape

        h = np.zeros((N, self.hidden_size), dtype=d.dtype) if h0 is None else h0.data
        if h.ndim == 1:
            h = h[np.newaxis, :]

        out = np.zeros((N, L, self.hidden_size), dtype=d.dtype)

        act = self.nonlinearity if self.nonlinearity else np.tanh

        for t in range(L):
            x_t = d[:, t, :]
            ih = np.dot(x_t, self.weight_ih.T)
            hh = np.dot(h, self.weight_hh.T)
            if self.bias:
                ih += self.bias_ih
                hh += self.bias_hh

            h = act(ih + hh)
            if hasattr(h, "data"):
                h = h.data
            out[:, t, :] = h

        if x.data.ndim == 2:
            out = out[0]
            # When working with pure numpy arrays, shape slicing works fine.
            # If h somehow became an object with no simple slice, we use np.squeeze
            h = (
                np.squeeze(h, axis=0)
                if (hasattr(h, "shape") and len(h.shape) > 0 and h.shape[0] == 1)
                else h
            )

        return array(out), array(h)


class GRU(Module):
    """A gated recurrent unit (GRU) RNN layer."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        """Initialize."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih = np.random.randn(3 * hidden_size, input_size)
        self.weight_hh = np.random.randn(3 * hidden_size, hidden_size)
        if self.bias:
            self.bias_ih = np.zeros(3 * hidden_size)
            self.bias_hh = np.zeros(3 * hidden_size)

    def __call__(
        self, x: array, h0: Optional[array] = None
    ) -> Union[array, Tuple[array, array]]:
        """Forward pass."""
        d = x.data
        if d.ndim == 2:
            d = d[np.newaxis, ...]
        N, L, _ = d.shape

        h = np.zeros((N, self.hidden_size), dtype=d.dtype) if h0 is None else h0.data
        if h.ndim == 1:
            h = h[np.newaxis, :]

        out = np.zeros((N, L, self.hidden_size), dtype=d.dtype)

        def sigmoid(v):
            """Sigmoid activation."""
            return 1 / (1 + np.exp(-v))

        for t in range(L):
            x_t = d[:, t, :]

            # W_ih: (3*H, I), W_hh: (3*H, H)
            ih = np.dot(x_t, self.weight_ih.T)
            hh = np.dot(h, self.weight_hh.T)

            if self.bias:
                ih += self.bias_ih
                hh += self.bias_hh

            # Split into r, z, n
            # Order in MLX/PyTorch usually is reset, update, new (r, z, n)
            r_ih, z_ih, n_ih = np.split(ih, 3, axis=-1)
            r_hh, z_hh, n_hh = np.split(hh, 3, axis=-1)

            r = sigmoid(r_ih + r_hh)
            z = sigmoid(z_ih + z_hh)
            n = np.tanh(n_ih + r * n_hh)

            h = (1 - z) * n + z * h
            out[:, t, :] = h

        if x.data.ndim == 2:
            out = out[0]
            # When working with pure numpy arrays, shape slicing works fine.
            # If h somehow became an object with no simple slice, we use np.squeeze
            h = (
                np.squeeze(h, axis=0)
                if (hasattr(h, "shape") and len(h.shape) > 0 and h.shape[0] == 1)
                else h
            )

        return array(out), array(h)


class LSTM(Module):
    """An LSTM recurrent layer."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        """Initialize."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih = np.random.randn(4 * hidden_size, input_size)
        self.weight_hh = np.random.randn(4 * hidden_size, hidden_size)
        if self.bias:
            self.bias_ih = np.zeros(4 * hidden_size)
            self.bias_hh = np.zeros(4 * hidden_size)

    def __call__(
        self, x: array, hx: Optional[Tuple[array, array]] = None
    ) -> Union[Tuple[array, Tuple[array, array]], array]:
        """Forward pass."""
        d = x.data
        if d.ndim == 2:
            d = d[np.newaxis, ...]
        N, L, _ = d.shape

        if hx is None:
            h = np.zeros((N, self.hidden_size), dtype=d.dtype)
            c = np.zeros((N, self.hidden_size), dtype=d.dtype)
        else:
            h = hx[0].data
            c = hx[1].data

        if h.ndim == 1:
            h = h[np.newaxis, :]
            c = c[np.newaxis, :]

        out = np.zeros((N, L, self.hidden_size), dtype=d.dtype)

        def sigmoid(v):
            """Sigmoid activation."""
            return 1 / (1 + np.exp(-v))

        for t in range(L):
            x_t = d[:, t, :]

            # W_ih: (4*H, I), W_hh: (4*H, H)
            ih = np.dot(x_t, self.weight_ih.T)
            hh = np.dot(h, self.weight_hh.T)

            if self.bias:
                ih += self.bias_ih
                hh += self.bias_hh

            # Split into i, f, g, o
            # MLX/PyTorch order is input, forget, cell (g), output
            i_ih, f_ih, g_ih, o_ih = np.split(ih, 4, axis=-1)
            i_hh, f_hh, g_hh, o_hh = np.split(hh, 4, axis=-1)

            i = sigmoid(i_ih + i_hh)
            f = sigmoid(f_ih + f_hh)
            g = np.tanh(g_ih + g_hh)
            o = sigmoid(o_ih + o_hh)

            c = f * c + i * g
            h = o * np.tanh(c)
            out[:, t, :] = h

        if x.data.ndim == 2:
            out = out[0]
            if hasattr(h, "shape") and len(h.shape) > 0 and h.shape[0] == 1:
                h = np.squeeze(h, axis=0)
            if hasattr(c, "shape") and len(c.shape) > 0 and c.shape[0] == 1:
                c = np.squeeze(c, axis=0)

        return array(out), (array(h), array(c))
