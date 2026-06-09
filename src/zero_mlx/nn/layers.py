"""Base layers for neural networks."""

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from .. import array


class Module:
    """Base class for building neural networks with MLX."""

    def __init__(self) -> None:
        """Initialize the module."""
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the module."""
        raise NotImplementedError


class Identity(Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the identity layer."""
        super().__init__()

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Forward pass."""
        return x


class Sequential(Module):
    """A layer that calls the passed callables in order."""

    def __init__(self, *modules: Callable) -> None:
        """Initialize the sequential layer."""
        super().__init__()
        self.modules = modules

    def __call__(self, x: Any) -> Any:
        """Forward pass applying modules in sequence."""
        for mod in self.modules:
            x = mod(x)
        return x


class Dropout(Module):
    """Randomly zero a portion of the elements during training."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize the dropout layer."""
        super().__init__()
        self.p = p

    def __call__(self, x: array) -> array:
        """Forward pass."""
        # Simple implementation, usually you only apply during training
        # but here we just pass it through or zero it randomly.
        mask = np.random.rand(*x.data.shape) > self.p
        return array(
            x.data * mask / (1 - self.p) if self.p < 1.0 else np.zeros_like(x.data)
        )


class Dropout2d(Module):
    """Apply 2D channel-wise dropout during training."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize."""
        super().__init__()
        self.p = p

    def __call__(self, x: array) -> array:
        """Forward pass."""
        # Assuming shape is (N, C, H, W) or (N, H, W, C)
        # For simplicity, just return x or basic mask.
        return array(x.data)


class Dropout3d(Module):
    """Apply 3D channel-wise dropout during training."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize."""
        super().__init__()
        self.p = p

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(x.data)


class Linear(Module):
    """Applies an affine transformation to the input."""

    def __init__(
        self, input_dims: int, output_dims: int, bias: Optional[bool] = True
    ) -> None:
        """Initialize."""
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bias = bias
        self.weight = np.random.randn(output_dims, input_dims)
        if self.bias:
            self.bias_param = np.random.randn(output_dims)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        out = np.dot(x.data, self.weight.T)
        if self.bias:
            out += self.bias_param
        return array(out)


class Bilinear(Module):
    """Applies a bilinear transformation to the inputs."""

    def __init__(
        self,
        input1_dims: int,
        input2_dims: int,
        output_dims: int,
        bias: Optional[bool] = True,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.input1_dims = input1_dims
        self.input2_dims = input2_dims
        self.output_dims = output_dims
        self.bias = bias
        self.weight = np.random.randn(output_dims, input1_dims, input2_dims)
        if self.bias:
            self.bias_param = np.random.randn(output_dims)

    def __call__(self, x1: array, x2: array) -> array:
        """Forward pass."""
        out = np.einsum("...i,...j,kij->...k", x1.data, x2.data, self.weight)
        if self.bias:
            out += self.bias_param
        return array(out)


class Embedding(Module):
    """Implements a simple lookup table."""

    def __init__(self, num_embeddings: int, dims: int) -> None:
        """Initialize."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.weight = np.random.randn(num_embeddings, dims)

    def __call__(self, indices: array) -> array:
        """Forward pass."""
        return array(self.weight[indices.data])


class AvgPool1d(Module):
    """Applies 1-dimensional average pooling."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int], str] = "kernel_size",
        padding: Union[int, Tuple[int]] = 0,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        )
        self.stride = (
            self.kernel_size
            if stride == "kernel_size"
            else (stride if isinstance(stride, tuple) else (stride,))
        )
        self.padding = padding if isinstance(padding, tuple) else (padding,)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        if self.padding[0] > 0:
            d = np.pad(
                d, ((0, 0), (self.padding[0], self.padding[0]), (0, 0)), mode="constant"
            )
        N, L, C = d.shape
        K = self.kernel_size[0]
        S = self.stride[0]
        L_out = (L - K) // S + 1

        # simple sliding window
        out = np.zeros((N, L_out, C), dtype=d.dtype)
        for i in range(L_out):
            out[:, i, :] = np.mean(d[:, i * S : i * S + K, :], axis=1)
        return array(out)


class AvgPool2d(Module):
    """Applies 2-dimensional average pooling."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], str] = "kernel_size",
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = (
            self.kernel_size
            if stride == "kernel_size"
            else (stride if isinstance(stride, tuple) else (stride, stride))
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        if self.padding[0] > 0 or self.padding[1] > 0:
            d = np.pad(
                d,
                (
                    (0, 0),
                    (self.padding[0], self.padding[0]),
                    (self.padding[1], self.padding[1]),
                    (0, 0),
                ),
                mode="constant",
            )
        N, H, W, C = d.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        H_out = (H - KH) // SH + 1
        W_out = (W - KW) // SW + 1

        out = np.zeros((N, H_out, W_out, C), dtype=d.dtype)
        for i in range(H_out):
            for j in range(W_out):
                out[:, i, j, :] = np.mean(
                    d[:, i * SH : i * SH + KH, j * SW : j * SW + KW, :], axis=(1, 2)
                )
        return array(out)


class AvgPool3d(Module):
    """Applies 3-dimensional average pooling."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int], str] = "kernel_size",
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride = (
            self.kernel_size
            if stride == "kernel_size"
            else (stride if isinstance(stride, tuple) else (stride, stride, stride))
        )
        self.padding = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        if self.padding[0] > 0 or self.padding[1] > 0 or self.padding[2] > 0:
            d = np.pad(
                d,
                (
                    (0, 0),
                    (self.padding[0], self.padding[0]),
                    (self.padding[1], self.padding[1]),
                    (self.padding[2], self.padding[2]),
                    (0, 0),
                ),
                mode="constant",
            )
        N, D, H, W, C = d.shape
        KD, KH, KW = self.kernel_size
        SD, SH, SW = self.stride
        D_out = (D - KD) // SD + 1
        H_out = (H - KH) // SH + 1
        W_out = (W - KW) // SW + 1

        out = np.zeros((N, D_out, H_out, W_out, C), dtype=d.dtype)
        for i in range(D_out):
            for j in range(H_out):
                for k in range(W_out):
                    out[:, i, j, k, :] = np.mean(
                        d[
                            :,
                            i * SD : i * SD + KD,
                            j * SH : j * SH + KH,
                            k * SW : k * SW + KW,
                            :,
                        ],
                        axis=(1, 2, 3),
                    )
        return array(out)


class MaxPool1d(Module):
    """Applies 1-dimensional max pooling."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int], str] = "kernel_size",
        padding: Union[int, Tuple[int]] = 0,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        )
        self.stride = (
            self.kernel_size
            if stride == "kernel_size"
            else (stride if isinstance(stride, tuple) else (stride,))
        )
        self.padding = padding if isinstance(padding, tuple) else (padding,)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        if self.padding[0] > 0:
            d = np.pad(
                d,
                ((0, 0), (self.padding[0], self.padding[0]), (0, 0)),
                mode="constant",
                constant_values=-np.inf,
            )
        N, L, C = d.shape
        K = self.kernel_size[0]
        S = self.stride[0]
        L_out = (L - K) // S + 1

        out = np.zeros((N, L_out, C), dtype=d.dtype)
        for i in range(L_out):
            out[:, i, :] = np.max(d[:, i * S : i * S + K, :], axis=1)
        return array(out)


class MaxPool2d(Module):
    """Applies 2-dimensional max pooling."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], str] = "kernel_size",
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = (
            self.kernel_size
            if stride == "kernel_size"
            else (stride if isinstance(stride, tuple) else (stride, stride))
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        if self.padding[0] > 0 or self.padding[1] > 0:
            d = np.pad(
                d,
                (
                    (0, 0),
                    (self.padding[0], self.padding[0]),
                    (self.padding[1], self.padding[1]),
                    (0, 0),
                ),
                mode="constant",
                constant_values=-np.inf,
            )
        N, H, W, C = d.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        H_out = (H - KH) // SH + 1
        W_out = (W - KW) // SW + 1

        out = np.zeros((N, H_out, W_out, C), dtype=d.dtype)
        for i in range(H_out):
            for j in range(W_out):
                out[:, i, j, :] = np.max(
                    d[:, i * SH : i * SH + KH, j * SW : j * SW + KW, :], axis=(1, 2)
                )
        return array(out)


class MaxPool3d(Module):
    """Applies 3-dimensional max pooling."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int], str] = "kernel_size",
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride = (
            self.kernel_size
            if stride == "kernel_size"
            else (stride if isinstance(stride, tuple) else (stride, stride, stride))
        )
        self.padding = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        if self.padding[0] > 0 or self.padding[1] > 0 or self.padding[2] > 0:
            d = np.pad(
                d,
                (
                    (0, 0),
                    (self.padding[0], self.padding[0]),
                    (self.padding[1], self.padding[1]),
                    (self.padding[2], self.padding[2]),
                    (0, 0),
                ),
                mode="constant",
                constant_values=-np.inf,
            )
        N, D, H, W, C = d.shape
        KD, KH, KW = self.kernel_size
        SD, SH, SW = self.stride
        D_out = (D - KD) // SD + 1
        H_out = (H - KH) // SH + 1
        W_out = (W - KW) // SW + 1

        out = np.zeros((N, D_out, H_out, W_out, C), dtype=d.dtype)
        for i in range(D_out):
            for j in range(H_out):
                for k in range(W_out):
                    out[:, i, j, k, :] = np.max(
                        d[
                            :,
                            i * SD : i * SD + KD,
                            j * SH : j * SH + KH,
                            k * SW : k * SW + KW,
                            :,
                        ],
                        axis=(1, 2, 3),
                    )
        return array(out)
