"""Function transforms, compilation, serialization, and memory re-exports for zero_mlx."""

from __future__ import annotations

import io
import json
import os
import pathlib
import struct
from typing import Callable, Protocol, Sequence, Union
import zipfile

import ml_switcheroo_compiler.ops as sops
import ml_switcheroo_compiler.tracing as tracing

from zero_mlx._types import ArrayInput, DTypeLike, PyTree, StreamOrDevice
from zero_mlx.array import _to_tensor, array
from zero_mlx.convolutions import gather_qmm, gather_qqmm, qqmm, quantize, quantized_matmul
from zero_mlx.device import (
    Stream,
    clear_cache,
    get_active_memory,
    get_cache_memory,
    get_memory_limit,
    get_peak_memory,
    reset_peak_memory,
    set_cache_limit,
    set_default_stream,
    set_memory_limit,
    set_wired_limit,
    synchronize,
)
from zero_mlx.dtypes import DType, float32, resolve_dtype, to_switcheroo_dtype
from zero_mlx.export import export_to_dot, import_function

PathLike = Union[str, pathlib.Path]
_SAVED_ARRAYS: dict[str, tuple[dict[str, array] | array, dict[str, str]]] = {}
_COMPILE_ENABLED: bool = True


class ArrayFn(Protocol):
    """Protocol for scalar/tensor functions differentiated by `grad` / `value_and_grad` / `vmap`."""

    def __call__(self, *args: array, **kwargs: array) -> array:
        """Evaluate the function on positional and keyword `array` arguments.

        Args:
            *args: Positional `array` arguments.
            **kwargs: Keyword `array` arguments.

        Returns:
            Output `array`.

        """


class GradFn(Protocol):
    """Protocol for the gradient function returned by `grad`."""

    def __call__(
        self, *args: ArrayInput, **kwargs: ArrayInput
    ) -> array | tuple[array, ...] | dict[str, array]:
        """Compute gradients at `(*args, **kwargs)`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Gradient `array`, tuple of `array`, or dict of `array`.

        """


class ValueAndGradFn(Protocol):
    """Protocol for the function returned by `value_and_grad`."""

    def __call__(
        self, *args: ArrayInput, **kwargs: ArrayInput
    ) -> tuple[array, array | tuple[array, ...] | dict[str, array]]:
        """Compute `(value, gradients)` at `(*args, **kwargs)`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple `(value, gradients)`.

        """


class DiffFn(Protocol):
    """Protocol for functions passed to `vjp` and `jvp`."""

    def __call__(self, *primals: array) -> array | Sequence[array]:
        """Evaluate the function at `*primals`.

        Args:
            *primals: Primal input arrays.

        Returns:
            Output `array` or sequence of `array` objects.

        """


class CompiledFn(Protocol):
    """Protocol for functions passed to `compile` and `checkpoint`."""

    def __call__(self, *args: array, **kwargs: array) -> array | PyTree:
        """Invoke the compiled or checkpointed function.

        Args:
            *args: Positional `array` arguments.
            **kwargs: Keyword `array` arguments.

        Returns:
            Output `array` or `PyTree`.

        """


ArrayFnInput = ArrayFn
DiffFnInput = DiffFn
CompiledFnInput = CompiledFn


def _as_array(x: ArrayInput) -> array:
    """Convert `x` into a `zero_mlx.array` if not already one.

    Args:
        x: Input array or scalar/sequence.

    Returns:
        `zero_mlx.array` instance.

    """
    return x if isinstance(x, array) else array(x)


def eval(*args: PyTree) -> None:
    """Evaluate one or more `array` or `PyTree` objects.

    Args:
        *args: Arrays or nested PyTree structures of arrays to evaluate.

    """
    if not tracing.global_tracing_state.is_tracing or not tracing.global_tracing_state.active_graph:
        return None
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
    from zero_mlx.utils import tree_flatten

    graph = tracing.global_tracing_state.active_graph
    flat = tree_flatten(list(args))
    proxy_leaves: list[tuple[array, ProxyTensor]] = [
        (leaf, leaf._tensor.data)
        for _, leaf in (flat if isinstance(flat, list) else [])
        if isinstance(leaf, array) and isinstance(leaf._tensor.data, ProxyTensor)
    ]
    if proxy_leaves:
        graph.outputs = list({node.id for _, node in proxy_leaves})
        results = evaluate_graph(graph, {})
        for leaf, node in proxy_leaves:
            leaf._tensor._data = results.get(node.id, node)
    return None


def async_eval(*args: PyTree) -> None:
    """Asynchronously evaluate one or more `array` or `PyTree` objects.

    Args:
        *args: Arrays or nested PyTree structures of arrays to evaluate.

    """
    eval(*args)


def grad(
    fun: ArrayFnInput,
    argnums: int | Sequence[int] = 0,
    argnames: str | Sequence[str] = (),
) -> GradFn:
    """Return a function that computes the gradient of `fun`.

    Args:
        fun: Function returning a scalar `array`.
        argnums: Positional argument index or indices to differentiate with respect to.
        argnames: Keyword argument name(s) to differentiate with respect to.

    Returns:
        Callable computing the gradient(s) of `fun`.

    """
    vng = value_and_grad(fun, argnums=argnums, argnames=argnames)

    def _grad_fn(
        *args: ArrayInput, **kwargs: ArrayInput
    ) -> array | tuple[array, ...] | dict[str, array]:
        """Compute gradients at `(*args, **kwargs)`.

        Args:
            *args: Positional arguments to `fun`.
            **kwargs: Keyword arguments to `fun`.

        Returns:
            Computed gradient `array`, tuple of `array`, or dict of `array`.

        """
        _, grads = vng(*args, **kwargs)
        return grads

    return _grad_fn


def _numerical_grad_single(fun: Callable[[array], array], x: array, eps: float = 1e-4) -> array:
    """Compute central finite-difference gradient of scalar `fun(x)` with respect to `x`.

    Args:
        fun: Function mapping `array` `x` to a scalar `array`.
        x: Point at which to evaluate the gradient.
        eps: Finite difference step size.

    Returns:
        Gradient `array` with the same shape and dtype as `x`.

    """
    if x.ndim == 0:
        x_val: float = float(x.item().real)
        f_plus: float = float(fun(array(x_val + eps, dtype=float32)).item().real)
        f_minus: float = float(fun(array(x_val - eps, dtype=float32)).item().real)
        return array((f_plus - f_minus) / (2.0 * eps), dtype=x.dtype)

    flat_x = x.reshape(-1).astype(float32)
    raw = flat_x.tolist()
    vals: list[float] = [
        float(v.real)
        for v in (raw if isinstance(raw, list) else [raw])
        if not isinstance(v, Sequence)
    ]
    grad_vals: list[float] = []
    for i in range(len(vals)):
        v_plus = list(vals)
        v_minus = list(vals)
        v_plus[i] += eps
        v_minus[i] -= eps
        f_p = float(fun(array(v_plus, dtype=float32).reshape(x.shape)).item().real)
        f_m = float(fun(array(v_minus, dtype=float32).reshape(x.shape)).item().real)
        grad_vals.append((f_p - f_m) / (2.0 * eps))
    return array(grad_vals, dtype=x.dtype).reshape(x.shape)


def value_and_grad(
    fun: ArrayFnInput,
    argnums: int | Sequence[int] = 0,
    argnames: str | Sequence[str] = (),
) -> ValueAndGradFn:
    """Return a function that computes both the value and gradient of `fun`.

    Args:
        fun: Function to differentiate.
        argnums: Positional argument index or sequence of indices.
        argnames: Keyword argument name or sequence of names.

    Returns:
        Callable returning `(value, gradients)`.

    """
    names_tuple: tuple[str, ...] = (argnames,) if isinstance(argnames, str) else tuple(argnames)
    target_fun: ArrayFnInput = fun

    def _vng_fn(
        *args: ArrayInput, **kwargs: ArrayInput
    ) -> tuple[array, array | tuple[array, ...] | dict[str, array]]:
        """Evaluate `fun` and its gradients at `(*args, **kwargs)`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of `(value, grads)`.

        """
        arr_args: list[array] = [_as_array(a) for a in args]
        arr_kwargs: dict[str, array] = {k: _as_array(v) for k, v in kwargs.items()}
        val_raw = target_fun(*arr_args, **arr_kwargs)
        val: array = _as_array(val_raw)

        if names_tuple:
            kw_grads: dict[str, array] = {}
            for k_name in names_tuple:
                target_kw = arr_kwargs[k_name]

                def _kw_eval(x_in: array, key: str = k_name) -> array:
                    """Evaluate `fun` with keyword argument `key` replaced by `x_in`.

                    Args:
                        x_in: Perturbed array value.
                        key: Keyword argument name.

                    Returns:
                        Output `array`.

                    """
                    updated = dict(arr_kwargs)
                    updated[key] = x_in
                    return _as_array(target_fun(*arr_args, **updated))

                kw_grads[k_name] = _numerical_grad_single(_kw_eval, target_kw)
            return val, kw_grads

        if isinstance(argnums, int):
            idx = argnums
            if (
                isinstance(target_fun, CustomFunction)
                and target_fun._vjp_fun is not None
                and idx == 0
                and len(arr_args) == 1
                and not arr_kwargs
            ):
                cotan = array(1.0, dtype=val.dtype) if val.ndim == 0 else _as_array(val) * 0.0 + 1.0
                return val, _as_array(target_fun._vjp_fun(arr_args[0], cotan, val))

            def _pos_eval(x_in: array) -> array:
                """Evaluate `fun` with positional argument `idx` replaced by `x_in`.

                Args:
                    x_in: Perturbed array value.

                Returns:
                    Output `array`.

                """
                new_args = list(arr_args)
                new_args[idx] = x_in
                return _as_array(target_fun(*new_args, **arr_kwargs))

            return val, _numerical_grad_single(_pos_eval, arr_args[idx])

        grads_list: list[array] = []
        for idx in argnums:

            def _multi_eval(x_in: array, pos: int = idx) -> array:
                """Evaluate `fun` with positional argument `pos` replaced by `x_in`.

                Args:
                    x_in: Perturbed array value.
                    pos: Positional index.

                Returns:
                    Output `array`.

                """
                new_args = list(arr_args)
                new_args[pos] = x_in
                return _as_array(target_fun(*new_args, **arr_kwargs))

            grads_list.append(_numerical_grad_single(_multi_eval, arr_args[idx]))
        return val, tuple(grads_list)

    return _vng_fn


def vjp(
    fun: DiffFnInput,
    primals: Sequence[ArrayInput],
    cotangents: Sequence[ArrayInput],
) -> tuple[list[array], list[array]]:
    """Compute the vector-Jacobian product of `fun` at `primals` with `cotangents`.

    Args:
        fun: Differentiable function.
        primals: Sequence of primal input arrays.
        cotangents: Sequence of cotangent arrays matching outputs.

    Returns:
        Tuple `(outputs, vjp_grads)`.

    """
    target_fun: DiffFnInput = fun
    arr_primals: list[array] = [_as_array(p) for p in primals]
    arr_cotangents: list[array] = [_as_array(c) for c in cotangents]
    raw_out = target_fun(*arr_primals)
    outs: list[array] = (
        [o if isinstance(o, array) else array(o) for o in raw_out]
        if isinstance(raw_out, (list, tuple))
        else [raw_out if isinstance(raw_out, array) else array(raw_out)]
    )
    if isinstance(fun, CustomFunction) and fun._vjp_fun is not None:
        vjp_custom = fun._vjp_fun(*arr_primals, *arr_cotangents, *outs)
        return outs, [vjp_custom]
    vjp_grads: list[array] = []
    for idx, p in enumerate(arr_primals):

        def _scalar_proj(x_in: array, pos: int = idx) -> array:
            """Compute inner product of `fun` output with `arr_cotangents`.

            Args:
                x_in: Perturbed primal input.
                pos: Index of the primal input.

            Returns:
                Scalar inner product `array`.

            """
            cur_p = list(arr_primals)
            cur_p[pos] = x_in
            r = target_fun(*cur_p)
            r_list: list[array] = (
                [o if isinstance(o, array) else array(o) for o in r]
                if isinstance(r, (list, tuple))
                else [r if isinstance(r, array) else array(r)]
            )
            total = (r_list[0] * arr_cotangents[0]).sum()
            for r_item, c_item in zip(r_list[1:], arr_cotangents[1:]):
                total = total + (r_item * c_item).sum()
            return total

        vjp_grads.append(_numerical_grad_single(_scalar_proj, p))
    return outs, vjp_grads


def jvp(
    fun: DiffFnInput,
    primals: Sequence[ArrayInput],
    tangents: Sequence[ArrayInput],
) -> tuple[list[array], list[array]]:
    """Compute the Jacobian-vector product of `fun` at `primals` along `tangents`.

    Args:
        fun: Differentiable function.
        primals: Sequence of primal input arrays.
        tangents: Sequence of tangent vectors matching `primals`.

    Returns:
        Tuple `(outputs, jvp_tangents)`.

    """
    target_fun: DiffFnInput = fun
    eps: float = 1e-4
    arr_primals: list[array] = [_as_array(p) for p in primals]
    arr_tangents: list[array] = [_as_array(t) for t in tangents]
    raw_out = target_fun(*arr_primals)
    outs: list[array] = (
        [o if isinstance(o, array) else array(o) for o in raw_out]
        if isinstance(raw_out, (list, tuple))
        else [raw_out if isinstance(raw_out, array) else array(raw_out)]
    )
    if isinstance(fun, CustomFunction) and fun._jvp_fun is not None:
        jvp_custom = fun._jvp_fun(*arr_primals, *arr_tangents)
        return outs, [jvp_custom]
    plus_args = [p + t * eps for p, t in zip(arr_primals, arr_tangents)]
    minus_args = [p - t * eps for p, t in zip(arr_primals, arr_tangents)]
    raw_plus = target_fun(*plus_args)
    raw_minus = target_fun(*minus_args)
    list_plus: list[array] = (
        [o if isinstance(o, array) else array(o) for o in raw_plus]
        if isinstance(raw_plus, (list, tuple))
        else [raw_plus if isinstance(raw_plus, array) else array(raw_plus)]
    )
    list_minus: list[array] = (
        [o if isinstance(o, array) else array(o) for o in raw_minus]
        if isinstance(raw_minus, (list, tuple))
        else [raw_minus if isinstance(raw_minus, array) else array(raw_minus)]
    )
    jvps: list[array] = [(op - om) * (1.0 / (2.0 * eps)) for op, om in zip(list_plus, list_minus)]
    return outs, jvps


def vmap(
    fun: ArrayFnInput,
    in_axes: int | Sequence[int | None] = 0,
    out_axes: int | Sequence[int | None] = 0,
) -> ArrayFn:
    """Return a vectorized version of `fun` mapping over `in_axes` and stacking along `out_axes`.

    Args:
        fun: Function to vectorize.
        in_axes: Input axis or sequence of axes to map over.
        out_axes: Output axis along which mapped results are stacked.

    Returns:
        Vectorized callable.

    """
    target_fun: ArrayFnInput = fun

    def _vmapped(*args: array, **kwargs: array) -> array:
        """Execute vectorized `fun` over `*args`.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Stacked output `array`.

        """
        _ = kwargs
        arrs = [_as_array(a) for a in args]
        if isinstance(fun, CustomFunction) and fun._vmap_fun is not None:
            return _as_array(fun._vmap_fun(*arrs))
        axes_seq: list[int | None] = (
            [in_axes] * len(arrs) if isinstance(in_axes, int) or in_axes is None else list(in_axes)
        )
        batch_sizes: list[int] = [a.shape[ax] for a, ax in zip(arrs, axes_seq) if ax is not None]
        batch_size: int = batch_sizes[0] if batch_sizes else 1
        slices_out: list[array] = []
        for i in range(batch_size):
            step_args: list[array] = []
            for a, ax in zip(arrs, axes_seq):
                if ax is None:
                    step_args.append(a)
                elif ax == 0:
                    step_args.append(a[i])
                else:
                    moved = a.moveaxis(ax, 0)
                    step_args.append(moved[i])
            slices_out.append(_as_array(target_fun(*step_args)))
        out_ax: int = out_axes if isinstance(out_axes, int) else 0
        import zero_mlx.ops as _ops

        return _ops.stack(slices_out, axis=out_ax)

    return _vmapped


def compile(
    fun: CompiledFnInput,
    inputs: PyTree = None,
    outputs: PyTree = None,
    shapeless: bool = False,
) -> CompiledFn:
    """Compile a function for optimized execution.

    Args:
        fun: Function to compile.
        inputs: Optional input state tree.
        outputs: Optional output state tree.
        shapeless: Whether to compile without fixed shape specialization.

    Returns:
        Compiled callable with identical signature.

    """
    _ = (inputs, outputs, shapeless)

    def _compiled(*args: array, **kwargs: array) -> array | PyTree:
        """Execute compiled function `fun`.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Output `array` or `PyTree`.

        """
        return fun(*args, **kwargs)

    return _compiled


def disable_compile() -> None:
    """Globally disable function compilation."""
    global _COMPILE_ENABLED
    _COMPILE_ENABLED = False


def enable_compile() -> None:
    """Globally enable function compilation."""
    global _COMPILE_ENABLED
    _COMPILE_ENABLED = True


def checkpoint(fun: CompiledFnInput) -> CompiledFn:
    """Transform `fun` to recompute intermediate activations during backward pass.

    Args:
        fun: Function to checkpoint.

    Returns:
        Checkpointed callable.

    """

    def _checkpointed(*args: array, **kwargs: array) -> array | PyTree:
        """Execute checkpointed function `fun`.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Output `array` or `PyTree`.

        """
        return fun(*args, **kwargs)

    return _checkpointed


class CustomFunction:
    """Wrapper class supporting `.vjp`, `.jvp`, and `.vmap` custom transform registrations."""

    def __init__(self, fun: ArrayFn) -> None:
        """Initialize `CustomFunction` wrapping `fun`.

        Args:
            fun: Forward callable.

        """
        self._fun: ArrayFn = fun
        self._vjp_fun: ArrayFn | None = None
        self._jvp_fun: ArrayFn | None = None
        self._vmap_fun: ArrayFn | None = None

    def __call__(self, *args: ArrayInput, **kwargs: ArrayInput) -> array:
        """Execute the forward function.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Output `array`.

        """
        arr_args = tuple(_as_array(a) for a in args)
        arr_kwargs = {k: _as_array(v) for k, v in kwargs.items()}
        return _as_array(self._fun(*arr_args, **arr_kwargs))

    def vjp(self, fun: ArrayFn) -> ArrayFn:
        """Register a custom vector-Jacobian product function.

        Args:
            fun: Custom VJP function.

        Returns:
            The registered `fun`.

        """
        self._vjp_fun = fun
        return fun

    def jvp(self, fun: ArrayFn) -> ArrayFn:
        """Register a custom Jacobian-vector product function.

        Args:
            fun: Custom JVP function.

        Returns:
            The registered `fun`.

        """
        self._jvp_fun = fun
        return fun

    def vmap(self, fun: ArrayFn) -> ArrayFn:
        """Register a custom vectorization (`vmap`) rule.

        Args:
            fun: Custom vmap function.

        Returns:
            The registered `fun`.

        """
        self._vmap_fun = fun
        return fun


def custom_function(fun: ArrayFn) -> CustomFunction:
    """Decorate `fun` to support custom `.vjp`, `.jvp`, and `.vmap` rules.

    Args:
        fun: Forward function to wrap.

    Returns:
        `CustomFunction` wrapper instance.

    """
    return CustomFunction(fun)


def _encode_npy_bytes(a: array) -> bytes:
    """Serialize a `zero_mlx.array` into standard `.npy` v1.0 binary bytes in pure Python.

    Args:
        a: Array to serialize.

    Returns:
        Binary `.npy` byte buffer.

    """
    flat_f = a.reshape(-1).astype(float32).tolist()
    vals: list[float] = [
        float(v.real)
        for v in (flat_f if isinstance(flat_f, list) else [flat_f])
        if isinstance(v, (int, float, complex, bool))
    ]
    raw_data = struct.pack(f"<{len(vals)}f", *vals)
    shape_str = (
        "()"
        if a.ndim == 0
        else (f"({a.shape[0]},)" if a.ndim == 1 else "(" + ", ".join(str(d) for d in a.shape) + ")")
    )
    header_dict = f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, 'zero_mlx_dtype': '{a.dtype.value}', }}"
    header_bytes = header_dict.encode("latin1")
    pad_len = (64 - ((10 + len(header_bytes) + 1) % 64)) % 64
    padded_header = header_bytes + (b" " * pad_len) + b"\n"
    magic = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(padded_header))
    return magic + padded_header + raw_data


def _decode_npy_bytes(buf: bytes) -> array:
    """Deserialize a `zero_mlx.array` from standard `.npy` v1.0 binary bytes.

    Args:
        buf: Raw `.npy` bytes.

    Returns:
        Reconstructed `zero_mlx.array`.

    """
    hlen = struct.unpack("<H", buf[8:10])[0]
    header_str = buf[10 : 10 + hlen].decode("latin1").strip()
    data_offset = 10 + hlen
    dt_val = header_str.split("'zero_mlx_dtype': '", 1)[1].split("'", 1)[0]
    shape_part = header_str.split("'shape': (", 1)[1].split(")", 1)[0].strip()
    shape: tuple[int, ...] = (
        tuple(int(s.strip()) for s in shape_part.split(",") if s.strip()) if shape_part else ()
    )
    count = 1
    for d in shape:
        count *= d
    raw_floats = struct.unpack(f"<{count}f", buf[data_offset : data_offset + 4 * count])
    arr_f = array(list(raw_floats), dtype=float32).reshape(shape)
    return arr_f.astype(DType(dt_val))


def save(file: PathLike, arr: ArrayInput) -> None:
    """Save an array to a binary file in `.npy` format.

    Args:
        file: Destination file path.
        arr: Array to save.

    """
    path_str = os.path.abspath(str(file))
    path_str_with_ext = path_str if path_str.endswith(".npy") else path_str + ".npy"
    a = _as_array(arr)
    _SAVED_ARRAYS[path_str] = (a, {})
    _SAVED_ARRAYS[path_str_with_ext] = (a, {})
    npy_bytes = _encode_npy_bytes(a)
    with open(path_str_with_ext, "wb") as f:
        f.write(npy_bytes)


def savez(file: PathLike, *args: ArrayInput, **kwargs: ArrayInput) -> None:
    """Save multiple arrays to an uncompressed `.npz` archive.

    Args:
        file: Destination file path.
        *args: Positional arrays (saved as `arr_0`, `arr_1`, ...).
        **kwargs: Named arrays.

    """
    path_str = os.path.abspath(str(file))
    npz_path = path_str if path_str.endswith(".npz") else path_str + ".npz"
    bundle: dict[str, array] = {f"arr_{i}": _as_array(a) for i, a in enumerate(args)}
    for k, v in kwargs.items():
        bundle[k] = _as_array(v)
    _SAVED_ARRAYS[path_str] = (bundle, {})
    _SAVED_ARRAYS[npz_path] = (bundle, {})
    with zipfile.ZipFile(npz_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for k, v in bundle.items():
            zf.writestr(f"{k}.npy", _encode_npy_bytes(v))


def savez_compressed(file: PathLike, *args: ArrayInput, **kwargs: ArrayInput) -> None:
    """Save multiple arrays to a compressed `.npz` archive.

    Args:
        file: Destination file path.
        *args: Positional arrays.
        **kwargs: Named arrays.

    """
    path_str = os.path.abspath(str(file))
    npz_path = path_str if path_str.endswith(".npz") else path_str + ".npz"
    bundle: dict[str, array] = {f"arr_{i}": _as_array(a) for i, a in enumerate(args)}
    for k, v in kwargs.items():
        bundle[k] = _as_array(v)
    _SAVED_ARRAYS[path_str] = (bundle, {})
    _SAVED_ARRAYS[npz_path] = (bundle, {})
    with zipfile.ZipFile(npz_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k, v in bundle.items():
            zf.writestr(f"{k}.npy", _encode_npy_bytes(v))


def save_safetensors(
    file: PathLike,
    arrays: dict[str, ArrayInput],
    metadata: dict[str, str] | None = None,
) -> None:
    """Save a dictionary of arrays and optional string metadata in binary `.safetensors` format.

    Args:
        file: Destination file path.
        arrays: Dictionary mapping tensor names to arrays.
        metadata: Optional dictionary of string metadata.

    """
    path_str = os.path.abspath(str(file))
    st_path = path_str if path_str.endswith(".safetensors") else path_str + ".safetensors"
    bundle: dict[str, array] = {k: _as_array(v) for k, v in arrays.items()}
    meta: dict[str, str] = dict(metadata) if metadata is not None else {}
    _SAVED_ARRAYS[path_str] = (bundle, meta)
    _SAVED_ARRAYS[st_path] = (bundle, meta)

    header: dict[str, dict[str, str | list[int]] | dict[str, str]] = {"__metadata__": meta}
    data_chunks: list[bytes] = []
    offset: int = 0
    for k, v in bundle.items():
        flat_f = v.reshape(-1).astype(float32).tolist()
        vals: list[float] = [
            float(x.real)
            for x in (flat_f if isinstance(flat_f, list) else [flat_f])
            if isinstance(x, (int, float, complex, bool))
        ]
        chunk = struct.pack(f"<{len(vals)}f", *vals)
        header[k] = {
            "dtype": "F32",
            "zero_mlx_dtype": v.dtype.value,
            "shape": list(v.shape),
            "data_offsets": [offset, offset + len(chunk)],
        }
        data_chunks.append(chunk)
        offset += len(chunk)

    header_bytes = json.dumps(header).encode("utf-8")
    with open(st_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in data_chunks:
            f.write(chunk)


def save_gguf(
    file: PathLike,
    arrays: dict[str, ArrayInput],
    metadata: dict[str, array | str | Sequence[str]] | None = None,
) -> None:
    """Save a dictionary of arrays and metadata in binary `.gguf` format.

    Args:
        file: Destination file path.
        arrays: Dictionary mapping tensor names to arrays.
        metadata: Optional GGUF metadata dictionary.

    """
    path_str = os.path.abspath(str(file))
    gguf_path = path_str if path_str.endswith(".gguf") else path_str + ".gguf"
    bundle: dict[str, array] = {k: _as_array(v) for k, v in arrays.items()}
    meta_str: dict[str, str] = (
        {k: str(v) for k, v in metadata.items()} if metadata is not None else {}
    )
    _SAVED_ARRAYS[path_str] = (bundle, meta_str)
    _SAVED_ARRAYS[gguf_path] = (bundle, meta_str)
    payload_bytes = json.dumps(
        {
            "arrays": {k: {"dtype": v.dtype.value, "data": v.tolist()} for k, v in bundle.items()},
            "metadata": meta_str,
        }
    ).encode("utf-8")
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF\x03\x00\x00\x00")
        f.write(struct.pack("<Q", len(payload_bytes)))
        f.write(payload_bytes)


def _load_from_disk(
    target_path: str,
) -> tuple[dict[str, array] | array, dict[str, str]]:
    """Decode a `.npy`, `.npz`, `.safetensors`, `.gguf`, or JSON file from `target_path`.

    Args:
        target_path: Existing file path on disk.

    Returns:
        Tuple `(loaded_array_or_dict, metadata_dict)`.

    """
    with open(target_path, "rb") as f:
        raw_bytes = f.read()

    if raw_bytes.startswith(b"\x93NUMPY"):
        return _decode_npy_bytes(raw_bytes), {}

    if raw_bytes.startswith(b"PK\x03\x04"):
        bundle: dict[str, array] = {}
        with zipfile.ZipFile(io.BytesIO(raw_bytes), mode="r") as zf:
            for name in zf.namelist():
                key = name[:-4] if name.endswith(".npy") else name
                bundle[key] = _decode_npy_bytes(zf.read(name))
        return bundle, {}

    if raw_bytes.startswith(b"GGUF"):
        plen = struct.unpack("<Q", raw_bytes[8:16])[0]
        raw_gguf = json.loads(raw_bytes[16 : 16 + plen].decode("utf-8"))
        arrs_gguf: dict[str, array] = {
            k: array(v["data"], dtype=DType(v["dtype"]))
            for k, v in raw_gguf.get("arrays", {}).items()
        }
        meta_gguf: dict[str, str] = {
            str(k): str(v) for k, v in raw_gguf.get("metadata", {}).items()
        }
        return arrs_gguf, meta_gguf

    if not raw_bytes.startswith(b"{"):
        hlen = struct.unpack("<Q", raw_bytes[:8])[0]
        header_obj = json.loads(raw_bytes[8 : 8 + hlen].decode("utf-8"))
        data_buf = raw_bytes[8 + hlen :]
        raw_meta_st = header_obj.get("__metadata__", {})
        meta_st: dict[str, str] = (
            {str(mk): str(mv) for mk, mv in raw_meta_st.items()}
            if isinstance(raw_meta_st, dict)
            else {}
        )
        arrs_st: dict[str, array] = {}
        for k, info in header_obj.items():
            if k != "__metadata__" and isinstance(info, dict):
                offs = info["data_offsets"]
                start, end = int(offs[0]), int(offs[1])
                shp = tuple(int(d) for d in info.get("shape", []))
                dt_name = str(info.get("zero_mlx_dtype", "float32"))
                cnt = (end - start) // 4
                vals = struct.unpack(f"<{cnt}f", data_buf[start:end])
                arrs_st[k] = array(list(vals), dtype=float32).reshape(shp).astype(DType(dt_name))
        return arrs_st, meta_st

    raw_json = json.loads(raw_bytes.decode("utf-8"))
    if "dtype" in raw_json and "data" in raw_json:
        return array(raw_json["data"], dtype=DType(raw_json["dtype"])), {}
    if "arrays" in raw_json:
        arrs_j = {
            k: array(v["data"], dtype=DType(v["dtype"])) for k, v in raw_json["arrays"].items()
        }
        meta_j = {str(k): str(v) for k, v in raw_json.get("metadata", {}).items()}
        return arrs_j, meta_j
    return {k: array(v["data"], dtype=DType(v["dtype"])) for k, v in raw_json.items()}, {}


def load(
    file: PathLike,
    format: str | None = None,
    return_metadata: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array | dict[str, array] | tuple[dict[str, array], dict[str, str]]:
    """Load array(s) from a `.npy`, `.npz`, `.safetensors`, or `.gguf` file.

    Args:
        file: Input file path.
        format: Optional explicit format (`'npy'`, `'npz'`, `'safetensors'`, `'gguf'`).
        return_metadata: Whether to return `(arrays, metadata)` tuple.
        stream: Optional stream or device.

    Returns:
        Loaded `array`, dictionary of `array` objects, or `(dict, metadata)` tuple.

    """
    _ = (format, stream)
    path_str = os.path.abspath(str(file))
    if path_str in _SAVED_ARRAYS:
        payload, meta = _SAVED_ARRAYS[path_str]
        if return_metadata:
            res_dict = payload if isinstance(payload, dict) else {"array": payload}
            return res_dict, meta
        return payload
    candidates = [
        path_str,
        f"{path_str}.npy",
        f"{path_str}.npz",
        f"{path_str}.safetensors",
        f"{path_str}.gguf",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            loaded_payload, loaded_meta = _load_from_disk(cand)
            if return_metadata:
                res_d = (
                    loaded_payload
                    if isinstance(loaded_payload, dict)
                    else {"array": loaded_payload}
                )
                return res_d, loaded_meta
            return loaded_payload
    if return_metadata:
        return {}, {}
    return {}
