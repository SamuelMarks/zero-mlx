"""Recurrent neural network layers (`RNN`, `GRU`, `LSTM`) for `mlx.nn`."""

from __future__ import annotations

import math

from zero_mlx.array import array
from zero_mlx.mlx_random import uniform
from zero_mlx.nn.base import Module
import zero_mlx.ops as mx


class RNN(Module):
    """An Elman recurrent neural network (`RNN`) layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ) -> None:
        """Initialize `RNN`.

        Args:
            input_size: Number of expected features in the input `x`.
            hidden_size: Number of features in the hidden state `h`.
            bias: If False, the layer does not use bias weights `bias_ih` and `bias_hh`.
            nonlinearity: Nonlinearity to use (`'tanh'` or `'relu'`).

        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.nonlinearity: str = nonlinearity
        scale: float = math.sqrt(1.0 / float(hidden_size))

        self.weight_ih: array = uniform(low=-scale, high=scale, shape=(hidden_size, input_size))
        self.weight_hh: array = uniform(low=-scale, high=scale, shape=(hidden_size, hidden_size))
        self.bias_ih: array | None = (
            uniform(low=-scale, high=scale, shape=(hidden_size,)) if bias else None
        )
        self.bias_hh: array | None = (
            uniform(low=-scale, high=scale, shape=(hidden_size,)) if bias else None
        )

    def __call__(self, x: array, hidden: array | None = None) -> tuple[array, array]:
        """Run the `RNN` over input sequence `x`.

        Args:
            x: Input sequence of shape `(B, L, input_size)` or `(L, input_size)`.
            hidden: Optional initial hidden state `(B, hidden_size)` or `(hidden_size,)`.

        Returns:
            Tuple `(output_sequence, final_hidden_state)`.

        """
        is_batched: bool = x.ndim == 3
        batch_size: int = x.shape[0] if is_batched else 1
        seq_len: int = x.shape[1] if is_batched else x.shape[0]

        h: array = (
            (hidden if hidden.ndim == 2 else mx.expand_dims(hidden, axis=0))
            if hidden is not None
            else mx.zeros((batch_size, self.hidden_size))
        )

        out_seq: list[array] = []
        for t in range(seq_len):
            x_t = x[:, t, :] if is_batched else mx.expand_dims(x[t, :], axis=0)
            ih = mx.matmul(x_t, mx.transpose(self.weight_ih))
            if self.bias_ih is not None:
                ih = mx.add(ih, self.bias_ih)

            hh = mx.matmul(h, mx.transpose(self.weight_hh))
            if self.bias_hh is not None:
                hh = mx.add(hh, self.bias_hh)

            h = mx.add(ih, hh)
            if self.nonlinearity == "relu":
                h = mx.maximum(h, 0.0)
            else:
                h = mx.tanh(h)
            out_seq.append(h)

        if is_batched:
            out = mx.stack(out_seq, axis=1)
        else:
            out = mx.concatenate(out_seq, axis=0)
        return out, h


class GRU(Module):
    """A Gated Recurrent Unit (`GRU`) layer."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        """Initialize `GRU`.

        Args:
            input_size: Number of expected features in the input `x`.
            hidden_size: Number of features in the hidden state `h`.
            bias: Whether to use learnable bias vectors `bias_ih` and `bias_hh`.

        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        scale: float = math.sqrt(1.0 / float(hidden_size))

        self.weight_ih: array = uniform(low=-scale, high=scale, shape=(3 * hidden_size, input_size))
        self.weight_hh: array = uniform(
            low=-scale, high=scale, shape=(3 * hidden_size, hidden_size)
        )
        self.bias_ih: array | None = (
            uniform(low=-scale, high=scale, shape=(3 * hidden_size,)) if bias else None
        )
        self.bias_hh: array | None = (
            uniform(low=-scale, high=scale, shape=(3 * hidden_size,)) if bias else None
        )

    def __call__(self, x: array, hidden: array | None = None) -> tuple[array, array]:
        """Run the `GRU` over input sequence `x`.

        Args:
            x: Input sequence of shape `(B, L, input_size)` or `(L, input_size)`.
            hidden: Optional initial hidden state.

        Returns:
            Tuple `(output_sequence, final_hidden_state)`.

        """
        is_batched: bool = x.ndim == 3
        batch_size: int = x.shape[0] if is_batched else 1
        seq_len: int = x.shape[1] if is_batched else x.shape[0]

        h: array = (
            (hidden if hidden.ndim == 2 else mx.expand_dims(hidden, axis=0))
            if hidden is not None
            else mx.zeros((batch_size, self.hidden_size))
        )

        out_seq: list[array] = []
        for t in range(seq_len):
            x_t = x[:, t, :] if is_batched else mx.expand_dims(x[t, :], axis=0)
            gi = mx.matmul(x_t, mx.transpose(self.weight_ih))
            if self.bias_ih is not None:
                gi = mx.add(gi, self.bias_ih)

            gh = mx.matmul(h, mx.transpose(self.weight_hh))
            if self.bias_hh is not None:
                gh = mx.add(gh, self.bias_hh)

            i_r, i_i, i_n = mx.split(gi, 3, axis=-1)
            h_r, h_i, h_n = mx.split(gh, 3, axis=-1)

            resetgate = mx.sigmoid(mx.add(i_r, h_r))
            inputgate = mx.sigmoid(mx.add(i_i, h_i))
            newgate = mx.tanh(mx.add(i_n, mx.multiply(resetgate, h_n)))

            h = mx.add(newgate, mx.multiply(inputgate, mx.subtract(h, newgate)))
            out_seq.append(h)

        if is_batched:
            out = mx.stack(out_seq, axis=1)
        else:
            out = mx.concatenate(out_seq, axis=0)
        return out, h


class LSTM(Module):
    """A Long Short-Term Memory (`LSTM`) recurrent neural network layer."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        """Initialize `LSTM`.

        Args:
            input_size: Number of expected features in the input `x`.
            hidden_size: Number of features in the hidden state `h` and cell state `c`.
            bias: Whether to use learnable bias vectors `bias_ih` and `bias_hh`.

        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        scale: float = math.sqrt(1.0 / float(hidden_size))

        self.weight_ih: array = uniform(low=-scale, high=scale, shape=(4 * hidden_size, input_size))
        self.weight_hh: array = uniform(
            low=-scale, high=scale, shape=(4 * hidden_size, hidden_size)
        )
        self.bias_ih: array | None = (
            uniform(low=-scale, high=scale, shape=(4 * hidden_size,)) if bias else None
        )
        self.bias_hh: array | None = (
            uniform(low=-scale, high=scale, shape=(4 * hidden_size,)) if bias else None
        )

    def __call__(
        self,
        x: array,
        hidden: array | tuple[array, array] | None = None,
        cell: array | None = None,
    ) -> tuple[array, tuple[array, array]]:
        """Run the `LSTM` over input sequence `x`.

        Args:
            x: Input sequence of shape `(B, L, input_size)` or `(L, input_size)`.
            hidden: Optional initial hidden state `h` or `(h, c)` tuple.
            cell: Optional initial cell state `c` when `hidden` is a single `array`.

        Returns:
            Tuple `(output_sequence, (final_h, final_c))`.

        """
        is_batched: bool = x.ndim == 3
        batch_size: int = x.shape[0] if is_batched else 1
        seq_len: int = x.shape[1] if is_batched else x.shape[0]

        if isinstance(hidden, tuple):
            h, c = hidden
        else:
            h = (
                (hidden if hidden.ndim == 2 else mx.expand_dims(hidden, axis=0))
                if hidden is not None
                else mx.zeros((batch_size, self.hidden_size))
            )
            c = (
                (cell if cell.ndim == 2 else mx.expand_dims(cell, axis=0))
                if cell is not None
                else mx.zeros((batch_size, self.hidden_size))
            )

        out_seq: list[array] = []
        for t in range(seq_len):
            x_t = x[:, t, :] if is_batched else mx.expand_dims(x[t, :], axis=0)
            gates_i = mx.matmul(x_t, mx.transpose(self.weight_ih))
            if self.bias_ih is not None:
                gates_i = mx.add(gates_i, self.bias_ih)

            gates_h = mx.matmul(h, mx.transpose(self.weight_hh))
            if self.bias_hh is not None:
                gates_h = mx.add(gates_h, self.bias_hh)

            gates = mx.add(gates_i, gates_h)
            i, f, g, o = mx.split(gates, 4, axis=-1)

            i = mx.sigmoid(i)
            f = mx.sigmoid(f)
            g = mx.tanh(g)
            o = mx.sigmoid(o)

            c = mx.add(mx.multiply(f, c), mx.multiply(i, g))
            h = mx.multiply(o, mx.tanh(c))
            out_seq.append(h)

        if is_batched:
            out = mx.stack(out_seq, axis=1)
        else:
            out = mx.concatenate(out_seq, axis=0)
        return out, (h, c)


__all__ = ["GRU", "LSTM", "RNN"]
