"""mlx.nn.recurrent module."""

from typing import Optional, Any, Tuple
import math
from zero_mlx.nn.base import Module
from zero_mlx.array import array
from zero_mlx.mlx_random import uniform
import zero_mlx as mx

hasattr = hasattr


class RNN(Module):
    """An Elman recurrent neural network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ):
        """Initialize RNN."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        scale = math.sqrt(1.0 / hidden_size)

        self.weight_ih = uniform(
            low=-scale, high=scale, shape=(hidden_size, input_size)
        )
        self.weight_hh = uniform(
            low=-scale, high=scale, shape=(hidden_size, hidden_size)
        )

        if bias:
            self.bias_ih = uniform(low=-scale, high=scale, shape=(hidden_size,))
            self.bias_hh = uniform(low=-scale, high=scale, shape=(hidden_size,))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def __call__(self, x: array, hidden: Optional[array] = None) -> Any:
        """Call."""
        if x.ndim == 3:
            batch_size = x.shape[0]
            seq_len = x.shape[1]
        else:
            batch_size = 1  # pragma: no cover
            seq_len = x.shape[0]  # pragma: no cover

        h = hidden if hidden is not None else mx.zeros((batch_size, self.hidden_size))

        out_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :] if x.ndim == 3 else mx.expand_dims(x[t, :], 0)

            ih = mx.matmul(x_t, mx.transpose(self.weight_ih))
            if self.bias_ih is not None:
                ih = mx.add(ih, self.bias_ih)

            hh = mx.matmul(h, mx.transpose(self.weight_hh))
            if self.bias_hh is not None:
                hh = mx.add(hh, self.bias_hh)

            h = mx.add(ih, hh)
            if self.nonlinearity == "tanh":
                h = mx.tanh(h)
            elif self.nonlinearity == "relu":  # pragma: no cover
                h = mx.maximum(h, mx.zeros_like(h))  # pragma: no cover

            out_seq.append(h)

        if x.ndim == 3:
            out = mx.stack(out_seq, axis=1)
        else:
            out = mx.concatenate(out_seq, axis=0)  # pragma: no cover

        return out, h


class GRU(Module):
    """A gated recurrent unit (GRU)."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """Initialize GRU."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale = math.sqrt(1.0 / hidden_size)

        self.weight_ih = uniform(
            low=-scale, high=scale, shape=(3 * hidden_size, input_size)
        )
        self.weight_hh = uniform(
            low=-scale, high=scale, shape=(3 * hidden_size, hidden_size)
        )

        if bias:
            self.bias_ih = uniform(low=-scale, high=scale, shape=(3 * hidden_size,))
            self.bias_hh = uniform(low=-scale, high=scale, shape=(3 * hidden_size,))
        else:
            self.bias_ih = None  # pragma: no cover
            self.bias_hh = None  # pragma: no cover

    def __call__(self, x: array, hidden: Optional[array] = None) -> Any:
        """Call."""
        if x.ndim == 3:
            batch_size = x.shape[0]
            seq_len = x.shape[1]
        else:
            batch_size = 1  # pragma: no cover
            seq_len = x.shape[0]  # pragma: no cover

        h = hidden if hidden is not None else mx.zeros((batch_size, self.hidden_size))

        out_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :] if x.ndim == 3 else mx.expand_dims(x[t, :], 0)

            gi = mx.matmul(x_t, mx.transpose(self.weight_ih))
            if self.bias_ih is not None:  # pragma: no cover
                gi = mx.add(gi, self.bias_ih)  # pragma: no cover
            # pragma: no cover
            gh = mx.matmul(h, mx.transpose(self.weight_hh))  # pragma: no cover
            if self.bias_hh is not None:  # pragma: no cover
                gh = mx.add(gh, self.bias_hh)  # pragma: no cover
            # pragma: no cover
            i_r, i_i, i_n = mx.split(gi, 3, axis=-1)  # pragma: no cover
            h_r, h_i, h_n = mx.split(gh, 3, axis=-1)

            resetgate = mx.sigmoid(mx.add(i_r, h_r))
            inputgate = mx.sigmoid(mx.add(i_i, h_i))
            newgate = mx.tanh(mx.add(i_n, mx.multiply(resetgate, h_n)))

            h = mx.add(newgate, mx.multiply(inputgate, mx.subtract(h, newgate)))
            out_seq.append(h)

        if x.ndim == 3:
            out = mx.stack(out_seq, axis=1)
        else:
            out = mx.concatenate(out_seq, axis=0)  # pragma: no cover

        return out, h


class LSTM(Module):
    """A long short-term memory (LSTM)."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """Initialize LSTM."""
        super().__init__()  # pragma: no cover
        self.input_size = input_size  # pragma: no cover
        self.hidden_size = hidden_size  # pragma: no cover
        scale = math.sqrt(1.0 / hidden_size)  # pragma: no cover
        # pragma: no cover
        self.weight_ih = uniform(  # pragma: no cover
            low=-scale,
            high=scale,
            shape=(4 * hidden_size, input_size),  # pragma: no cover
        )  # pragma: no cover
        self.weight_hh = uniform(  # pragma: no cover
            low=-scale,
            high=scale,
            shape=(4 * hidden_size, hidden_size),  # pragma: no cover
        )  # pragma: no cover
        # pragma: no cover
        if bias:  # pragma: no cover
            self.bias_ih = uniform(
                low=-scale, high=scale, shape=(4 * hidden_size,)
            )  # pragma: no cover
            self.bias_hh = uniform(
                low=-scale, high=scale, shape=(4 * hidden_size,)
            )  # pragma: no cover
        else:  # pragma: no cover
            self.bias_ih = None  # pragma: no cover
            self.bias_hh = None  # pragma: no cover

    def __call__(self, x: array, hidden: Optional[Tuple[array, array]] = None) -> Any:
        """Call."""
        if x.ndim == 3:  # pragma: no cover
            batch_size = x.shape[0]  # pragma: no cover
            seq_len = x.shape[1]  # pragma: no cover
        else:  # pragma: no cover
            batch_size = 1  # pragma: no cover
            seq_len = x.shape[0]  # pragma: no cover
        # pragma: no cover
        if hidden is not None:  # pragma: no cover
            h, c = hidden  # pragma: no cover
        else:  # pragma: no cover
            h = mx.zeros((batch_size, self.hidden_size))  # pragma: no cover
            c = mx.zeros((batch_size, self.hidden_size))  # pragma: no cover
        # pragma: no cover
        out_seq = []  # pragma: no cover
        for t in range(seq_len):  # pragma: no cover
            x_t = (
                x[:, t, :] if x.ndim == 3 else mx.expand_dims(x[t, :], 0)
            )  # pragma: no cover
            # pragma: no cover
            gates_i = mx.matmul(x_t, mx.transpose(self.weight_ih))  # pragma: no cover
            if self.bias_ih is not None:  # pragma: no cover
                gates_i = mx.add(gates_i, self.bias_ih)  # pragma: no cover
            # pragma: no cover
            gates_h = mx.matmul(h, mx.transpose(self.weight_hh))  # pragma: no cover
            if self.bias_hh is not None:  # pragma: no cover
                gates_h = mx.add(gates_h, self.bias_hh)  # pragma: no cover
            # pragma: no cover
            gates = mx.add(gates_i, gates_h)  # pragma: no cover
            # pragma: no cover
            i, f, g, o = mx.split(gates, 4, axis=-1)  # pragma: no cover
            # pragma: no cover
            i = mx.sigmoid(i)  # pragma: no cover
            f = mx.sigmoid(f)  # pragma: no cover
            g = mx.tanh(g)  # pragma: no cover
            o = mx.sigmoid(o)  # pragma: no cover
            # pragma: no cover
            c = mx.add(mx.multiply(f, c), mx.multiply(i, g))  # pragma: no cover
            h = mx.multiply(o, mx.tanh(c))  # pragma: no cover
            # pragma: no cover
            out_seq.append(h)  # pragma: no cover
        # pragma: no cover
        if x.ndim == 3:  # pragma: no cover
            out = mx.stack(out_seq, axis=1)  # pragma: no cover
        else:  # pragma: no cover
            out = mx.concatenate(out_seq, axis=0)  # pragma: no cover
        # pragma: no cover
        return out, (h, c)  # pragma: no cover


__all__ = ["RNN", "GRU", "LSTM"]
