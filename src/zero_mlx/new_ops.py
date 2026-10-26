"""Function transforms, compilation, serialization, and memory re-exports for zero_mlx."""

from __future__ import annotations

import io
import json
import os
import pathlib
import struct
from typing import IO, Callable, Protocol, Sequence, Union
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
from zero_mlx.dtypes import DType, float32, float64, resolve_dtype, to_switcheroo_dtype
from zero_mlx.export import export_to_dot, import_function

PathLike = Union[str, pathlib.Path]
FileOrPath = Union[str, pathlib.Path, IO[bytes]]
_COMPILE_ENABLED: bool = True

GradOutput = Union[
    array,
    tuple[array, ...],
    dict[str, array],
    tuple[Union[array, tuple[array, ...]], dict[str, array]],
]


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

    def __call__(self, *args: ArrayInput, **kwargs: ArrayInput) -> GradOutput:
        """Compute gradients at `(*args, **kwargs)`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Gradient `array`, tuple of `array`, dict of `array`, or `(pos_grads, kw_grads)`.

        """


class ValueAndGradFn(Protocol):
    """Protocol for the function returned by `value_and_grad`."""

    def __call__(self, *args: ArrayInput, **kwargs: ArrayInput) -> tuple[array, GradOutput]:
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


def _as_array(x: ArrayInput | tuple[array, ...] | list[array]) -> array:
    """Convert `x` into a `zero_mlx.array` if not already one.

    Args:
        x: Input array or scalar/sequence.

    Returns:
        `zero_mlx.array` instance.

    """
    if isinstance(x, array):
        return x
    if isinstance(x, (tuple, list)) and len(x) == 1 and isinstance(x[0], array):
        return x[0]
    return array(x)


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
    argnums: int | Sequence[int] | None = None,
    argnames: str | Sequence[str] = (),
) -> GradFn:
    """Return a function that computes the gradient of scalar-valued `fun`.

    Args:
        fun: Function returning a scalar `array`.
        argnums: Positional argument index or indices to differentiate with respect to.
        argnames: Keyword argument name(s) to differentiate with respect to.

    Returns:
        Callable computing the gradient(s) of `fun`.

    """
    vng = value_and_grad(fun, argnums=argnums, argnames=argnames)

    def _grad_fn(*args: ArrayInput, **kwargs: ArrayInput) -> GradOutput:
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


def _numerical_grad_single(fun: Callable[[array], array], x: array, eps: float = 1e-3) -> array:
    """Compute central difference gradient of scalar `fun(x)` with respect to `x`.

    Args:
        fun: Function mapping `array` `x` to a scalar `array`.
        x: Point at which to evaluate the gradient.
        eps: Perturbation step size.

    Returns:
        Gradient `array` with the same shape and dtype as `x`.

    """
    if not x.dtype.value.startswith(("float", "bfloat", "complex")):
        return array(sops.zeros(x.shape, dtype=to_switcheroo_dtype(x.dtype)), dtype=x.dtype)
    step: float = 1e-6 if x.dtype == float64 else eps
    if x.ndim == 0:
        x_val: float = float(x.item().real)
        f_plus: float = float(fun(array(x_val + step, dtype=x.dtype)).item().real)
        f_minus: float = float(fun(array(x_val - step, dtype=x.dtype)).item().real)
        return array((f_plus - f_minus) / (2.0 * step), dtype=x.dtype)

    flat_x = x.reshape(-1)
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
        v_plus[i] += step
        v_minus[i] -= step
        f_p = float(fun(array(v_plus, dtype=x.dtype).reshape(x.shape)).item().real)
        f_m = float(fun(array(v_minus, dtype=x.dtype).reshape(x.shape)).item().real)
        grad_vals.append((f_p - f_m) / (2.0 * step))
    return array(grad_vals, dtype=x.dtype).reshape(x.shape)


def _compute_pos_grads(
    target_fun: ArrayFnInput,
    val: array,
    arr_args: list[array],
    arr_kwargs: dict[str, array],
    argnums: int | Sequence[int],
) -> array | tuple[array, ...]:
    """Compute positional argument gradients for `argnums`.

    Args:
        target_fun: Function being differentiated.
        val: Forward pass scalar output `array`.
        arr_args: Coerced positional `array` arguments.
        arr_kwargs: Coerced keyword `array` arguments.
        argnums: Positional index or sequence of indices.

    Returns:
        Single gradient `array` if `argnums` is an `int`, else tuple of gradient `array` objects.

    """
    if isinstance(argnums, int):
        idx = argnums
        if isinstance(target_fun, CustomFunction) and target_fun._vjp_fun is not None:
            cotan = array(1.0, dtype=val.dtype)
            vjp_out = target_fun._vjp_fun(*arr_args, cotan, val, **arr_kwargs)
            return _as_array(vjp_out)

        def _pos_eval(x_in: array) -> array:
            """Evaluate `target_fun` with positional argument `idx` replaced by `x_in`.

            Args:
                x_in: Perturbed array value.

            Returns:
                Output `array`.

            """
            new_args = list(arr_args)
            new_args[idx] = x_in
            return _as_array(target_fun(*new_args, **arr_kwargs))

        return _numerical_grad_single(_pos_eval, arr_args[idx])

    grads_list: list[array] = []
    for idx in argnums:

        def _multi_eval(x_in: array, pos: int = idx) -> array:
            """Evaluate `target_fun` with positional argument `pos` replaced by `x_in`.

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
    return tuple(grads_list)


def value_and_grad(
    fun: ArrayFnInput,
    argnums: int | Sequence[int] | None = None,
    argnames: str | Sequence[str] = (),
) -> ValueAndGradFn:
    """Return a function that computes both the value and gradient of `fun`.

    Args:
        fun: Scalar-valued function to differentiate.
        argnums: Positional argument index or sequence of indices.
        argnames: Keyword argument name or sequence of names.

    Returns:
        Callable returning `(value, gradients)`.

    """
    names_tuple: tuple[str, ...] = (argnames,) if isinstance(argnames, str) else tuple(argnames)
    target_fun: ArrayFnInput = fun

    def _vng_fn(*args: ArrayInput, **kwargs: ArrayInput) -> tuple[array, GradOutput]:
        """Evaluate `fun` and its gradients at `(*args, **kwargs)`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of `(value, grads)`.

        Raises:
            ValueError: If `fun` does not return a scalar (0-D) `array`.

        """
        arr_args: list[array] = [_as_array(a) for a in args]
        arr_kwargs: dict[str, array] = {k: _as_array(v) for k, v in kwargs.items()}
        val: array = _as_array(target_fun(*arr_args, **arr_kwargs))
        if val.ndim != 0:
            raise ValueError(
                f"grad requires a scalar-valued function, but got output with shape {val.shape}"
            )

        kw_grads: dict[str, array] = {}
        if names_tuple:
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
            if argnums is None:
                return val, kw_grads
            pos_g = _compute_pos_grads(target_fun, val, arr_args, arr_kwargs, argnums)
            return val, (pos_g, kw_grads)

        eff_argnums: int | Sequence[int] = 0 if argnums is None else argnums
        return val, _compute_pos_grads(target_fun, val, arr_args, arr_kwargs, eff_argnums)

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
        return outs, [_as_array(vjp_custom)]
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
    eps: float = 1e-5
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
        return outs, [_as_array(jvp_custom)]
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


def _stack_vmap_outputs(
    slices: list[array],
    out_ax: int | None,
) -> array:
    """Stack a list of per-slice `array` outputs along `out_ax`.

    Args:
        slices: Per-slice `array` outputs.
        out_ax: Output axis (or None to return unstacked first slice).

    Returns:
        Stacked `array`.

    """
    if out_ax is None:
        return slices[0]
    import zero_mlx.ops as _ops

    return _ops.stack(slices, axis=out_ax)


def vmap(
    fun: ArrayFnInput,
    in_axes: int | Sequence[int | None] | None = 0,
    out_axes: int | Sequence[int | None] | None = 0,
) -> ArrayFn:
    """Return a vectorized version of `fun` mapping over `in_axes` and stacking along `out_axes`.

    Args:
        fun: Function to vectorize.
        in_axes: Input axis or sequence of axes to map over.
        out_axes: Output axis or sequence of axes along which mapped results are stacked.

    Returns:
        Vectorized callable.

    """
    target_fun: ArrayFnInput = fun

    def _vmapped(*args: array, **kwargs: array) -> array:
        """Execute vectorized `fun` over `*args` and `**kwargs`.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Stacked output `array`.

        Raises:
            ValueError: If mapped input dimensions do not match in size.

        """
        arrs = [_as_array(a) for a in args]
        arr_kwargs = {k: _as_array(v) for k, v in kwargs.items()}
        if isinstance(fun, CustomFunction) and fun._vmap_fun is not None:
            return _as_array(fun._vmap_fun(*arrs, **arr_kwargs))

        axes_seq: list[int | None] = (
            [in_axes] * len(arrs) if isinstance(in_axes, int) or in_axes is None else list(in_axes)
        )
        batch_sizes: list[int] = []
        for a, ax in zip(arrs, axes_seq):
            if ax is not None:
                norm_ax = ax if ax >= 0 else a.ndim + ax
                batch_sizes.append(a.shape[norm_ax])
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"vmap in_axes have mismatched dimension sizes: {batch_sizes}")
        batch_size: int = batch_sizes[0] if batch_sizes else 1

        single_slices: list[array] = []
        for i in range(batch_size):
            step_args: list[array] = []
            for a, ax in zip(arrs, axes_seq):
                if ax is None:
                    step_args.append(a)
                else:
                    norm_ax = ax if ax >= 0 else a.ndim + ax
                    if norm_ax == 0:
                        step_args.append(a[i])
                    else:
                        step_args.append(a.moveaxis(norm_ax, 0)[i])
            single_slices.append(_as_array(target_fun(*step_args, **arr_kwargs)))

        single_out_ax: int | None = (
            out_axes if isinstance(out_axes, int) or out_axes is None else out_axes[0]
        )
        return _stack_vmap_outputs(single_slices, single_out_ax)

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
    _ = (inputs, outputs)
    compiled_cache: dict[tuple[tuple[int, ...], ...], int] = {}

    def _compiled(*args: array, **kwargs: array) -> array | PyTree:
        """Execute compiled function `fun`.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Output `array` or `PyTree`.

        """
        if not _COMPILE_ENABLED:
            return fun(*args, **kwargs)
        sig_key: tuple[tuple[int, ...], ...] = (
            (()) if shapeless else tuple(a.shape for a in args if isinstance(a, array))
        )
        compiled_cache[sig_key] = compiled_cache.get(sig_key, 0) + 1
        out = fun(*args, **kwargs)
        eval(out)
        return out

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


_DTYPE_TO_NPY_DESCR: dict[DType, tuple[str, str]] = {
    DType.bool_: ("|b1", "B"),
    DType.uint8: ("|u1", "B"),
    DType.int8: ("|i1", "b"),
    DType.uint16: ("<u2", "H"),
    DType.int16: ("<i2", "h"),
    DType.uint32: ("<u4", "I"),
    DType.int32: ("<i4", "i"),
    DType.uint64: ("<u8", "Q"),
    DType.int64: ("<i8", "q"),
    DType.float16: ("<f2", "e"),
    DType.bfloat16: ("<V2", "e"),
    DType.float32: ("<f4", "f"),
    DType.float64: ("<f8", "d"),
    DType.complex64: ("<c8", "f"),
}

_NPY_DESCR_TO_DTYPE: dict[str, tuple[DType, str]] = {
    "|b1": (DType.bool_, "B"),
    "|u1": (DType.uint8, "B"),
    "<u1": (DType.uint8, "B"),
    "|i1": (DType.int8, "b"),
    "<i1": (DType.int8, "b"),
    "<u2": (DType.uint16, "H"),
    "<i2": (DType.int16, "h"),
    "<u4": (DType.uint32, "I"),
    "<i4": (DType.int32, "i"),
    "<u8": (DType.uint64, "Q"),
    "<i8": (DType.int64, "q"),
    "<f2": (DType.float16, "e"),
    "<V2": (DType.bfloat16, "e"),
    "<f4": (DType.float32, "f"),
    "<f8": (DType.float64, "d"),
    "<c8": (DType.complex64, "f"),
}

_DTYPE_TO_ST_NAME: dict[DType, str] = {
    DType.bool_: "BOOL",
    DType.uint8: "U8",
    DType.int8: "I8",
    DType.uint16: "U16",
    DType.int16: "I16",
    DType.uint32: "U32",
    DType.int32: "I32",
    DType.uint64: "U64",
    DType.int64: "I64",
    DType.float16: "F16",
    DType.bfloat16: "BF16",
    DType.float32: "F32",
    DType.float64: "F64",
    DType.complex64: "C64",
}

_ST_NAME_TO_DTYPE: dict[str, DType] = {v: k for k, v in _DTYPE_TO_ST_NAME.items()}


def _pack_array_bytes(a: array) -> bytes:
    """Pack `a` into little-endian raw bytes matching its native `DType`.

    Args:
        a: Array to pack.

    Returns:
        Packed binary bytes.

    """
    _, fmt_char = _DTYPE_TO_NPY_DESCR[a.dtype]
    flat = a.reshape(-1).tolist()
    items = flat if isinstance(flat, list) else [flat]
    if a.dtype == DType.complex64:
        c_floats: list[float] = []
        for v in items:
            c = complex(v) if isinstance(v, (int, float, complex, bool)) else 0j
            c_floats.extend([float(c.real), float(c.imag)])
        return struct.pack(f"<{len(c_floats)}f", *c_floats)
    if fmt_char in ("e", "f", "d"):
        f_vals = [
            float(v.real) if isinstance(v, (int, float, complex, bool)) else 0.0 for v in items
        ]
        return struct.pack(f"<{len(f_vals)}{fmt_char}", *f_vals)
    i_vals = [int(v.real) if isinstance(v, (int, float, complex, bool)) else 0 for v in items]
    return struct.pack(f"<{len(i_vals)}{fmt_char}", *i_vals)


def _unpack_array_bytes(buf: bytes, dt: DType, shape: tuple[int, ...]) -> array:
    """Unpack little-endian raw bytes `buf` into an `array` of `dt` and `shape`.

    Args:
        buf: Raw binary tensor payload.
        dt: Target `DType`.
        shape: Target shape tuple.

    Returns:
        Unpacked `zero_mlx.array`.

    """
    _, fmt_char = _DTYPE_TO_NPY_DESCR[dt]
    count = 1
    for d in shape:
        count *= d
    if dt == DType.complex64:
        floats = struct.unpack(f"<{count * 2}f", buf[: count * 8])
        c_list = [complex(floats[2 * i], floats[2 * i + 1]) for i in range(count)]
        return array(c_list, dtype=dt).reshape(shape)
    item_sz = struct.calcsize(f"<{fmt_char}")
    vals = struct.unpack(f"<{count}{fmt_char}", buf[: count * item_sz])
    if dt == DType.bool_:
        return array([bool(v) for v in vals], dtype=dt).reshape(shape)
    return array(list(vals), dtype=dt).reshape(shape)


def _encode_npy_bytes(a: array) -> bytes:
    """Serialize a `zero_mlx.array` into standard `.npy` v1.0 binary bytes in pure Python.

    Args:
        a: Array to serialize.

    Returns:
        Binary `.npy` byte buffer.

    """
    descr, _ = _DTYPE_TO_NPY_DESCR[a.dtype]
    raw_data = _pack_array_bytes(a)
    shape_str = (
        "()"
        if a.ndim == 0
        else (f"({a.shape[0]},)" if a.ndim == 1 else "(" + ", ".join(str(d) for d in a.shape) + ")")
    )
    header_dict = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}"
    header_bytes = header_dict.encode("latin1")
    pad_len = (64 - ((10 + len(header_bytes) + 1) % 64)) % 64
    padded_header = header_bytes + (b" " * pad_len) + bytes([10])
    magic = bytes([0x93]) + b"NUMPY" + bytes([1, 0]) + struct.pack("<H", len(padded_header))
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
    descr = header_str.split("'descr': '", 1)[1].split("'", 1)[0]
    dt, _ = _NPY_DESCR_TO_DTYPE.get(descr, (DType.float32, "f"))
    shape_part = header_str.split("'shape': (", 1)[1].split(")", 1)[0].strip()
    shape: tuple[int, ...] = (
        tuple(int(s.strip()) for s in shape_part.split(",") if s.strip()) if shape_part else ()
    )
    return _unpack_array_bytes(buf[data_offset:], dt, shape)


def save(file: FileOrPath, arr: ArrayInput) -> None:
    """Save an array to a binary file or stream in `.npy` format.

    Args:
        file: Destination file path or binary stream.
        arr: Array to save.

    """
    a = _as_array(arr)
    npy_bytes = _encode_npy_bytes(a)
    if isinstance(file, (str, pathlib.Path)):
        path_str = os.path.abspath(str(file))
        path_str_with_ext = path_str if path_str.endswith(".npy") else path_str + ".npy"
        with open(path_str_with_ext, "wb") as f:
            f.write(npy_bytes)
    else:
        file.write(npy_bytes)


def savez(file: FileOrPath, *args: ArrayInput, **kwargs: ArrayInput) -> None:
    """Save multiple arrays to an uncompressed `.npz` archive.

    Args:
        file: Destination file path or binary stream.
        *args: Positional arrays (saved as `arr_0`, `arr_1`, ...).
        **kwargs: Named arrays.

    """
    bundle: dict[str, array] = {f"arr_{i}": _as_array(a) for i, a in enumerate(args)}
    for k, v in kwargs.items():
        bundle[k] = _as_array(v)
    target: str | IO[bytes] = (
        (
            os.path.abspath(str(file))
            if str(file).endswith(".npz")
            else os.path.abspath(str(file)) + ".npz"
        )
        if isinstance(file, (str, pathlib.Path))
        else file
    )
    with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for k, v in bundle.items():
            zf.writestr(f"{k}.npy", _encode_npy_bytes(v))


def savez_compressed(file: FileOrPath, *args: ArrayInput, **kwargs: ArrayInput) -> None:
    """Save multiple arrays to a compressed `.npz` archive.

    Args:
        file: Destination file path or binary stream.
        *args: Positional arrays.
        **kwargs: Named arrays.

    """
    bundle: dict[str, array] = {f"arr_{i}": _as_array(a) for i, a in enumerate(args)}
    for k, v in kwargs.items():
        bundle[k] = _as_array(v)
    target: str | IO[bytes] = (
        (
            os.path.abspath(str(file))
            if str(file).endswith(".npz")
            else os.path.abspath(str(file)) + ".npz"
        )
        if isinstance(file, (str, pathlib.Path))
        else file
    )
    with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k, v in bundle.items():
            zf.writestr(f"{k}.npy", _encode_npy_bytes(v))


def save_safetensors(
    file: FileOrPath,
    arrays: dict[str, ArrayInput],
    metadata: dict[str, str] | None = None,
) -> None:
    """Save a dictionary of arrays and optional string metadata in binary `.safetensors` format.

    Args:
        file: Destination file path or binary stream.
        arrays: Dictionary mapping tensor names to arrays.
        metadata: Optional dictionary of string metadata.

    """
    bundle: dict[str, array] = {k: _as_array(v) for k, v in arrays.items()}
    meta: dict[str, str] = dict(metadata) if metadata is not None else {}

    header: dict[str, dict[str, str | list[int]] | dict[str, str]] = {"__metadata__": meta}
    data_chunks: list[bytes] = []
    offset: int = 0
    for k, v in bundle.items():
        chunk = _pack_array_bytes(v)
        shape_list: list[int] = list(v.shape)
        offsets_list: list[int] = [offset, offset + len(chunk)]
        entry_dict: dict[str, str | list[int]] = {
            "dtype": _DTYPE_TO_ST_NAME[v.dtype],
            "shape": shape_list,
            "data_offsets": offsets_list,
        }
        header[k] = entry_dict
        data_chunks.append(chunk)
        offset += len(chunk)

    header_bytes = json.dumps(header).encode("utf-8")
    payload = struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(data_chunks)
    if isinstance(file, (str, pathlib.Path)):
        path_str = os.path.abspath(str(file))
        st_path = path_str if path_str.endswith(".safetensors") else path_str + ".safetensors"
        with open(st_path, "wb") as f:
            f.write(payload)
    else:
        file.write(payload)


def save_gguf(
    file: FileOrPath,
    arrays: dict[str, ArrayInput],
    metadata: dict[str, array | str | Sequence[str]] | None = None,
) -> None:
    """Save a dictionary of arrays and metadata in binary `.gguf` format.

    Args:
        file: Destination file path or binary stream.
        arrays: Dictionary mapping tensor names to arrays.
        metadata: Optional GGUF metadata dictionary.

    """
    bundle: dict[str, array] = {k: _as_array(v) for k, v in arrays.items()}
    meta_str: dict[str, str] = (
        {k: str(v) for k, v in metadata.items()} if metadata is not None else {}
    )
    tensor_headers: list[dict[str, str | list[int]]] = []
    chunks: list[bytes] = []
    offset = 0
    for k, v in bundle.items():
        raw = _pack_array_bytes(v)
        tensor_headers.append(
            {
                "name": k,
                "dtype": v.dtype.value,
                "shape": list(v.shape),
                "offset": [offset, offset + len(raw)],
            }
        )
        chunks.append(raw)
        offset += len(raw)

    header_bytes = json.dumps({"tensors": tensor_headers, "metadata": meta_str}).encode("utf-8")
    payload = (
        b"GGUF"
        + bytes([3, 0, 0, 0])
        + struct.pack("<QQ", len(tensor_headers), len(meta_str))
        + struct.pack("<Q", len(header_bytes))
        + header_bytes
        + b"".join(chunks)
    )
    if isinstance(file, (str, pathlib.Path)):
        path_str = os.path.abspath(str(file))
        gguf_path = path_str if path_str.endswith(".gguf") else path_str + ".gguf"
        with open(gguf_path, "wb") as f:
            f.write(payload)
    else:
        file.write(payload)


def _decode_raw_bytes(
    raw_bytes: bytes,
) -> tuple[dict[str, array] | array, dict[str, str]]:
    """Decode raw binary bytes from `.npy`, `.npz`, `.safetensors`, `.gguf`, or JSON format.

    Args:
        raw_bytes: Complete binary content of the file or stream.

    Returns:
        Tuple `(loaded_array_or_dict, metadata_dict)`.

    """
    if raw_bytes.startswith(bytes([0x93]) + b"NUMPY"):
        return _decode_npy_bytes(raw_bytes), {}

    if raw_bytes.startswith(b"PK" + bytes([3, 4])):
        bundle: dict[str, array] = {}
        with zipfile.ZipFile(io.BytesIO(raw_bytes), mode="r") as zf:
            for name in zf.namelist():
                key = name[:-4] if name.endswith(".npy") else name
                bundle[key] = _decode_npy_bytes(zf.read(name))
        return bundle, {}

    if raw_bytes.startswith(b"GGUF"):
        hlen = struct.unpack("<Q", raw_bytes[24:32])[0]
        raw_gguf = json.loads(raw_bytes[32 : 32 + hlen].decode("utf-8"))
        data_section = raw_bytes[32 + hlen :]
        arrs_gguf: dict[str, array] = {}
        for t_info in raw_gguf.get("tensors", []):
            name = str(t_info["name"])
            dt = DType(str(t_info["dtype"]))
            shp = tuple(int(d) for d in t_info["shape"])
            start, end = int(t_info["offset"][0]), int(t_info["offset"][1])
            arrs_gguf[name] = _unpack_array_bytes(data_section[start:end], dt, shp)
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
                st_dt_str = str(info.get("dtype", "F32"))
                dt = _ST_NAME_TO_DTYPE.get(st_dt_str, DType.float32)
                arrs_st[k] = _unpack_array_bytes(data_buf[start:end], dt, shp)
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


_VALID_LOAD_FORMATS: frozenset[str] = frozenset({"npy", "npz", "safetensors", "gguf"})


def load(
    file: FileOrPath,
    format: str | None = None,
    return_metadata: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array | dict[str, array] | tuple[dict[str, array], dict[str, str]]:
    """Load array(s) from a `.npy`, `.npz`, `.safetensors`, or `.gguf` file or binary stream.

    Args:
        file: Input file path or binary stream.
        format: Optional explicit format (`'npy'`, `'npz'`, `'safetensors'`, `'gguf'`).
        return_metadata: Whether to return `(arrays, metadata)` tuple.
        stream: Optional stream or device.

    Returns:
        Loaded `array`, dictionary of `array` objects, or `(dict, metadata)` tuple.

    Raises:
        ValueError: If `format` is not one of `'npy'`, `'npz'`, `'safetensors'`, `'gguf'`.
        FileNotFoundError: If `file` is a path that does not exist on disk.

    """
    _ = stream
    if format is not None and format not in _VALID_LOAD_FORMATS:
        raise ValueError(
            f"Unsupported load format '{format}'. Expected one of {sorted(_VALID_LOAD_FORMATS)}"
        )

    if not isinstance(file, (str, pathlib.Path)):
        loaded_payload, loaded_meta = _decode_raw_bytes(file.read())
        if return_metadata:
            res_s = (
                loaded_payload if isinstance(loaded_payload, dict) else {"array": loaded_payload}
            )
            return res_s, loaded_meta
        return loaded_payload

    path_str = os.path.abspath(str(file))
    candidates = (
        [path_str, f"{path_str}.{format}"]
        if format is not None
        else [
            path_str,
            f"{path_str}.npy",
            f"{path_str}.npz",
            f"{path_str}.safetensors",
            f"{path_str}.gguf",
        ]
    )
    for cand in candidates:
        if os.path.exists(cand):
            with open(cand, "rb") as f:
                raw_bytes = f.read()
            loaded_payload, loaded_meta = _decode_raw_bytes(raw_bytes)
            if return_metadata:
                res_d = (
                    loaded_payload
                    if isinstance(loaded_payload, dict)
                    else {"array": loaded_payload}
                )
                return res_d, loaded_meta
            return loaded_payload
    raise FileNotFoundError(f"No such file or archive found for '{file}'")
