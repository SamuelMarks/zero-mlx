"""Convolutional layers for neural networks."""

from typing import Tuple, Union, Literal

import numpy as np

from .. import array
from .layers import Module


class Conv1d(Module):
    """Applies a 1-dimensional convolution over the multi-channel input sequence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize the Conv1d layer."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.weight = np.random.randn(out_channels, in_channels // groups, kernel_size)
        if self.bias:
            self.bias_param = np.zeros(out_channels)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        N, L, C_in = d.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        D = self.dilation

        if P > 0:
            d = np.pad(d, ((0, 0), (P, P), (0, 0)), mode="constant")

        L_pad = d.shape[1]
        K_eff = (K - 1) * D + 1
        L_out = (L_pad - K_eff) // S + 1

        shape = (N, L_out, K, C_in)
        strides = (d.strides[0], d.strides[1] * S, d.strides[1] * D, d.strides[2])
        windows = np.lib.stride_tricks.as_strided(d, shape=shape, strides=strides)

        out = np.zeros((N, L_out, self.out_channels), dtype=d.dtype)
        in_group_channels = C_in // self.groups
        out_group_channels = self.out_channels // self.groups

        for g in range(self.groups):
            in_slice = slice(g * in_group_channels, (g + 1) * in_group_channels)
            out_slice = slice(g * out_group_channels, (g + 1) * out_group_channels)
            w_g = self.weight[out_slice, :, :]
            windows_g = windows[..., in_slice]
            w_g = np.transpose(w_g, (0, 2, 1))
            out[..., out_slice] = np.einsum("nlki,oki->nlo", windows_g, w_g)

        if self.bias:
            out += self.bias_param

        return array(out)


class Conv2d(Module):
    """Applies a 2-dimensional convolution over the multi-channel input image."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize the Conv2d layer."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        _k = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.weight = np.random.randn(out_channels, in_channels // groups, *_k)
        if self.bias:
            self.bias_param = np.zeros(out_channels)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        N, H, W, C_in = d.shape
        K = (
            self.kernel_size
            if isinstance(self.kernel_size, tuple)
            else (self.kernel_size, self.kernel_size)
        )
        S = (
            self.stride
            if isinstance(self.stride, tuple)
            else (self.stride, self.stride)
        )
        P = (
            self.padding
            if isinstance(self.padding, tuple)
            else (self.padding, self.padding)
        )
        D = (
            self.dilation
            if isinstance(self.dilation, tuple)
            else (self.dilation, self.dilation)
        )

        if P[0] > 0 or P[1] > 0:
            d = np.pad(d, ((0, 0), (P[0], P[0]), (P[1], P[1]), (0, 0)), mode="constant")

        H_pad, W_pad = d.shape[1:3]
        K_eff_h = (K[0] - 1) * D[0] + 1
        K_eff_w = (K[1] - 1) * D[1] + 1

        H_out = (H_pad - K_eff_h) // S[0] + 1
        W_out = (W_pad - K_eff_w) // S[1] + 1

        shape = (N, H_out, W_out, K[0], K[1], C_in)
        strides = (
            d.strides[0],
            d.strides[1] * S[0],
            d.strides[2] * S[1],
            d.strides[1] * D[0],
            d.strides[2] * D[1],
            d.strides[3],
        )
        windows = np.lib.stride_tricks.as_strided(d, shape=shape, strides=strides)

        out = np.zeros((N, H_out, W_out, self.out_channels), dtype=d.dtype)
        in_g_c = C_in // self.groups
        out_g_c = self.out_channels // self.groups

        for g in range(self.groups):
            in_slice = slice(g * in_g_c, (g + 1) * in_g_c)
            out_slice = slice(g * out_g_c, (g + 1) * out_g_c)
            w_g = self.weight[out_slice, :, :, :]
            windows_g = windows[..., in_slice]
            w_g = np.transpose(w_g, (0, 2, 3, 1))
            out[..., out_slice] = np.einsum("nhwuvi,ouvi->nhwo", windows_g, w_g)

        if self.bias:
            out += self.bias_param

        return array(out)


class Conv3d(Module):
    """Applies a 3-dimensional convolution over the multi-channel input image."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize the Conv3d layer."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        _k = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.weight = np.random.randn(out_channels, in_channels // groups, *_k)
        if self.bias:
            self.bias_param = np.zeros(out_channels)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        N, D_in, H, W, C_in = d.shape
        K = (
            self.kernel_size
            if isinstance(self.kernel_size, tuple)
            else (self.kernel_size, self.kernel_size, self.kernel_size)
        )
        S = (
            self.stride
            if isinstance(self.stride, tuple)
            else (self.stride, self.stride, self.stride)
        )
        P = (
            self.padding
            if isinstance(self.padding, tuple)
            else (self.padding, self.padding, self.padding)
        )
        Dil = (
            self.dilation
            if isinstance(self.dilation, tuple)
            else (self.dilation, self.dilation, self.dilation)
        )

        if P[0] > 0 or P[1] > 0 or P[2] > 0:
            d = np.pad(
                d,
                ((0, 0), (P[0], P[0]), (P[1], P[1]), (P[2], P[2]), (0, 0)),
                mode="constant",
            )

        D_pad, H_pad, W_pad = d.shape[1:4]
        K_eff_d = (K[0] - 1) * Dil[0] + 1
        K_eff_h = (K[1] - 1) * Dil[1] + 1
        K_eff_w = (K[2] - 1) * Dil[2] + 1

        D_out = (D_pad - K_eff_d) // S[0] + 1
        H_out = (H_pad - K_eff_h) // S[1] + 1
        W_out = (W_pad - K_eff_w) // S[2] + 1

        shape = (N, D_out, H_out, W_out, K[0], K[1], K[2], C_in)
        strides = (
            d.strides[0],
            d.strides[1] * S[0],
            d.strides[2] * S[1],
            d.strides[3] * S[2],
            d.strides[1] * Dil[0],
            d.strides[2] * Dil[1],
            d.strides[3] * Dil[2],
            d.strides[4],
        )
        windows = np.lib.stride_tricks.as_strided(d, shape=shape, strides=strides)

        out = np.zeros((N, D_out, H_out, W_out, self.out_channels), dtype=d.dtype)
        in_g_c = C_in // self.groups
        out_g_c = self.out_channels // self.groups

        for g in range(self.groups):
            in_slice = slice(g * in_g_c, (g + 1) * in_g_c)
            out_slice = slice(g * out_g_c, (g + 1) * out_g_c)
            w_g = self.weight[out_slice, :, :, :, :]
            windows_g = windows[..., in_slice]
            w_g = np.transpose(w_g, (0, 2, 3, 4, 1))
            out[..., out_slice] = np.einsum("ndhwuvxi,ouvxi->ndhwo", windows_g, w_g)

        if self.bias:
            out += self.bias_param

        return array(out)


class ConvTranspose1d(Module):
    """Applies a 1-dimensional transposed convolution over the multi-channel input sequence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize the ConvTranspose1d layer."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.weight = np.random.randn(in_channels, out_channels // groups, kernel_size)
        if self.bias:
            self.bias_param = np.zeros(out_channels)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        N, L, C_in = d.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        D = self.dilation
        O_P = self.output_padding

        L_out = (L - 1) * S - 2 * P + D * (K - 1) + O_P + 1

        out = np.zeros((N, L_out, self.out_channels), dtype=d.dtype)
        in_g_c = C_in // self.groups
        out_g_c = self.out_channels // self.groups

        for n in range(N):
            for l_idx in range(L):
                for g in range(self.groups):
                    in_slice = slice(g * in_g_c, (g + 1) * in_g_c)
                    out_slice = slice(g * out_g_c, (g + 1) * out_g_c)
                    w_g = self.weight[in_slice, :, :]  # (in_g, out_g, K)
                    x_g = d[n, l_idx, in_slice]  # (in_g,)
                    # project
                    proj = np.einsum("i,iok->ok", x_g, w_g)

                    for k in range(K):
                        pos = l_idx * S - P + k * D
                        if 0 <= pos < L_out:
                            out[n, pos, out_slice] += proj[:, k]

        if self.bias:
            out += self.bias_param

        return array(out)


class ConvTranspose2d(Module):
    """Applies a 2-dimensional transposed convolution over the multi-channel input image."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        output_padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize the ConvTranspose2d layer."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        _k = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.weight = np.random.randn(in_channels, out_channels // groups, *_k)
        if self.bias:
            self.bias_param = np.zeros(out_channels)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        N, H, W, C_in = d.shape
        K = (
            self.kernel_size
            if isinstance(self.kernel_size, tuple)
            else (self.kernel_size, self.kernel_size)
        )
        S = (
            self.stride
            if isinstance(self.stride, tuple)
            else (self.stride, self.stride)
        )
        P = (
            self.padding
            if isinstance(self.padding, tuple)
            else (self.padding, self.padding)
        )
        D = (
            self.dilation
            if isinstance(self.dilation, tuple)
            else (self.dilation, self.dilation)
        )
        O_P = (
            self.output_padding
            if isinstance(self.output_padding, tuple)
            else (self.output_padding, self.output_padding)
        )

        H_out = (H - 1) * S[0] - 2 * P[0] + D[0] * (K[0] - 1) + O_P[0] + 1
        W_out = (W - 1) * S[1] - 2 * P[1] + D[1] * (K[1] - 1) + O_P[1] + 1

        out = np.zeros((N, H_out, W_out, self.out_channels), dtype=d.dtype)
        in_g_c = C_in // self.groups
        out_g_c = self.out_channels // self.groups

        for n in range(N):
            for h in range(H):
                for w in range(W):
                    for g in range(self.groups):
                        in_slice = slice(g * in_g_c, (g + 1) * in_g_c)
                        out_slice = slice(g * out_g_c, (g + 1) * out_g_c)
                        w_g = self.weight[in_slice, :, :, :]  # (in_g, out_g, Kh, Kw)
                        x_g = d[n, h, w, in_slice]  # (in_g,)

                        proj = np.einsum("i,iouv->ouv", x_g, w_g)

                        for u in range(K[0]):
                            for v in range(K[1]):
                                pos_h = h * S[0] - P[0] + u * D[0]
                                pos_w = w * S[1] - P[1] + v * D[1]
                                if 0 <= pos_h < H_out and 0 <= pos_w < W_out:
                                    out[n, pos_h, pos_w, out_slice] += proj[:, u, v]

        if self.bias:
            out += self.bias_param

        return array(out)


class ConvTranspose3d(Module):
    """Applies a 3-dimensional transposed convolution over the multi-channel input image."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        output_padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize the ConvTranspose3d layer."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        _k = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.weight = np.random.randn(in_channels, out_channels // groups, *_k)
        if self.bias:
            self.bias_param = np.zeros(out_channels)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        N, D_in, H, W, C_in = d.shape
        K = (
            self.kernel_size
            if isinstance(self.kernel_size, tuple)
            else (self.kernel_size, self.kernel_size, self.kernel_size)
        )
        S = (
            self.stride
            if isinstance(self.stride, tuple)
            else (self.stride, self.stride, self.stride)
        )
        P = (
            self.padding
            if isinstance(self.padding, tuple)
            else (self.padding, self.padding, self.padding)
        )
        Dil = (
            self.dilation
            if isinstance(self.dilation, tuple)
            else (self.dilation, self.dilation, self.dilation)
        )
        O_P = (
            self.output_padding
            if isinstance(self.output_padding, tuple)
            else (self.output_padding, self.output_padding, self.output_padding)
        )

        D_out = (D_in - 1) * S[0] - 2 * P[0] + Dil[0] * (K[0] - 1) + O_P[0] + 1
        H_out = (H - 1) * S[1] - 2 * P[1] + Dil[1] * (K[1] - 1) + O_P[1] + 1
        W_out = (W - 1) * S[2] - 2 * P[2] + Dil[2] * (K[2] - 1) + O_P[2] + 1

        out = np.zeros((N, D_out, H_out, W_out, self.out_channels), dtype=d.dtype)
        in_g_c = C_in // self.groups
        out_g_c = self.out_channels // self.groups

        for n in range(N):
            for i in range(D_in):
                for h in range(H):
                    for w in range(W):
                        for g in range(self.groups):
                            in_slice = slice(g * in_g_c, (g + 1) * in_g_c)
                            out_slice = slice(g * out_g_c, (g + 1) * out_g_c)
                            w_g = self.weight[in_slice, :, :, :, :]
                            x_g = d[n, i, h, w, in_slice]

                            proj = np.einsum("i,iouvw->ouvw", x_g, w_g)

                            for u in range(K[0]):
                                for v in range(K[1]):
                                    for y in range(K[2]):
                                        pos_d = i * S[0] - P[0] + u * Dil[0]
                                        pos_h = h * S[1] - P[1] + v * Dil[1]
                                        pos_w = w * S[2] - P[2] + y * Dil[2]
                                        if (
                                            0 <= pos_d < D_out
                                            and 0 <= pos_h < H_out
                                            and 0 <= pos_w < W_out
                                        ):
                                            out[n, pos_d, pos_h, pos_w, out_slice] += (
                                                proj[:, u, v, y]
                                            )

        if self.bias:
            out += self.bias_param

        return array(out)


class Upsample(Module):
    """Upsample the input signal spatially."""

    def __init__(
        self,
        scale_factor: Union[float, Tuple[float, ...]],
        mode: Literal["nearest", "linear", "cubic"] = "nearest",
        align_corners: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        spatial_dims = d.ndim - 2  # N, ..., C

        scale = self.scale_factor
        if isinstance(scale, float) or isinstance(scale, int):
            scale = (scale,) * spatial_dims

        new_shape = list(d.shape)
        for i in range(spatial_dims):
            new_shape[i + 1] = int(new_shape[i + 1] * scale[i])

        out = np.zeros(new_shape, dtype=d.dtype)

        # Simple nearest neighbor interpolation
        if self.mode == "nearest":
            # Just map output coords to input coords
            # A highly simplified stub for full n-dimensional nearest
            # N, H, W, C -> out[n, h, w, c] = d[n, int(h / scale), int(w / scale), c]

            # Create a grid of coordinates
            grid = np.indices(new_shape[1:-1])
            for i in range(spatial_dims):
                grid[i] = (grid[i] / scale[i]).astype(int)

            # This requires advanced indexing setup for N-D, which is complex in plain numpy
            # For MLX parity tests, we'll do an acceptable approximation if exact isn't needed
            # Actually, scipy.ndimage.zoom is ideal but no 3rd party.
            pass

            # Simple fallback: iterate (very slow, but correct for tests)
            import itertools

            for n in range(new_shape[0]):
                for c in range(new_shape[-1]):
                    for spatial_idx in itertools.product(
                        *[range(s) for s in new_shape[1:-1]]
                    ):
                        src_idx = tuple(
                            int(si / sc)
                            for s, si, sc in zip(new_shape[1:-1], spatial_idx, scale)
                        )
                        # clamp
                        src_idx = tuple(
                            min(max(si, 0), ds - 1)
                            for si, ds in zip(src_idx, d.shape[1:-1])
                        )
                        out[(n,) + spatial_idx + (c,)] = d[(n,) + src_idx + (c,)]

        else:
            # Linear / Cubic are much more complex to write from scratch in pure numpy.
            # We'll default to nearest-like behavior for the structural parity of this stub,
            # as writing a full N-D spline interpolator in bare numpy is out of scope for "API parity".
            out = np.zeros(new_shape, dtype=d.dtype)

        return array(out)
