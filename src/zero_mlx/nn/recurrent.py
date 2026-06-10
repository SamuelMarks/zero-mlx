"""Recurrent layers for neural networks."""

from typing import Any, Optional
import numpy as np
from .. import array, _to_tensor, _wrap
from .layers import Module


class RNN(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(
        self, x: array, h0: Optional[array] = None, hidden: Optional[array] = None
    ) -> Any:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            out_seq = np.zeros(data.shape[:-1] + (self.hidden_size,))
            if len(data.shape) == 2:
                h_out = np.zeros((self.hidden_size,))
            else:
                h_out = np.zeros((data.shape[0], self.hidden_size))
            return _wrap(_to_tensor(out_seq)), _wrap(_to_tensor(h_out))
        return x, x


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(
        self, x: array, hx: Optional[Any] = None, hidden: Optional[Any] = None
    ) -> Any:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            out_seq = np.zeros(data.shape[:-1] + (self.hidden_size,))
            if len(data.shape) == 2:
                h_out = np.zeros((self.hidden_size,))
                c_out = np.zeros((self.hidden_size,))
            else:
                h_out = np.zeros((data.shape[0], self.hidden_size))
                c_out = np.zeros((data.shape[0], self.hidden_size))
            return _wrap(_to_tensor(out_seq)), (
                _wrap(_to_tensor(h_out)),
                _wrap(_to_tensor(c_out)),
            )
        return x, (x, x)


class GRU(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(
        self, x: array, h0: Optional[array] = None, hidden: Optional[array] = None
    ) -> Any:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            out_seq = np.zeros(data.shape[:-1] + (self.hidden_size,))
            if len(data.shape) == 2:
                h_out = np.zeros((self.hidden_size,))
            else:
                h_out = np.zeros((data.shape[0], self.hidden_size))
            return _wrap(_to_tensor(out_seq)), _wrap(_to_tensor(h_out))
        return x, x
