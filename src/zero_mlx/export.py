"""Function export, import, and graph visualization utilities for zero_mlx."""

from __future__ import annotations

import json
import os
from types import TracebackType
from typing import IO, Callable, Protocol, Sequence

from zero_mlx._types import ArrayInput, NestedSequence, Scalar
from zero_mlx.array import array
from zero_mlx.dtypes import DType


class ExportableFunction(Protocol):
    """Protocol for functions that can be traced and exported to `.mlxfn`."""

    def __call__(self, *args: array, **kwargs: array) -> array | tuple[array, ...]:
        """Invoke the function with positional and keyword `array` arguments.

        Args:
            *args: Positional `array` arguments.
            **kwargs: Keyword `array` arguments.

        Returns:
            Output `array` or tuple of `array` objects.

        """


class ImportedFunction(Protocol):
    """Protocol for functions imported from a `.mlxfn` file."""

    def __call__(self, *args: ArrayInput, **kwargs: ArrayInput) -> tuple[array, ...]:
        """Invoke the imported function and return a tuple of output arrays.

        Args:
            *args: Positional array inputs.
            **kwargs: Keyword array inputs.

        Returns:
            Tuple of output `array` objects.

        """


_EXPORT_REGISTRY: dict[str, tuple[ExportableFunction, dict[str, str]]] = {}

NL: str = chr(10)


class FunctionExporter:
    """Context manager for exporting multiple traces of the same function to a `.mlxfn` file."""

    def __init__(
        self,
        file: str,
        fun: ExportableFunction | None = None,
        *,
        shapeless: bool = False,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Initialize the `FunctionExporter`.

        Args:
            file: Destination file path for the exported function.
            fun: Callable function to trace and export.
            shapeless: Whether the exported trace allows dynamic shapes.
            metadata: Optional string key-value metadata dictionary.

        """
        self.file: str = file
        self.fun: ExportableFunction | None = fun
        self.shapeless: bool = shapeless
        self.metadata: dict[str, str] = dict(metadata) if metadata is not None else {}
        self.traces: list[tuple[tuple[array, ...], dict[str, array]]] = []
        self._trace_outputs: list[list[dict[str, str | list[int] | NestedSequence | Scalar]]] = []
        self._closed: bool = False
        _ = (
            _EXPORT_REGISTRY.setdefault(os.path.abspath(file), (self.fun, self.metadata))
            if self.fun is not None
            else None
        )

    def __call__(self, *args: ArrayInput, **kwargs: ArrayInput) -> None:
        """Record a function trace with the given positional and keyword array arguments.

        Args:
            *args: Positional array arguments for tracing.
            **kwargs: Keyword array arguments for tracing.

        Raises:
            RuntimeError: If the exporter has already been closed.

        """
        if self._closed:
            raise RuntimeError("Cannot export on a closed FunctionExporter")
        arr_args = tuple(a if isinstance(a, array) else array(a) for a in args)
        arr_kwargs = {k: (v if isinstance(v, array) else array(v)) for k, v in kwargs.items()}
        self.traces.append((arr_args, arr_kwargs))
        if self.fun is not None:
            res = self.fun(*arr_args, **arr_kwargs)
            res_tup = res if isinstance(res, tuple) else (res,)
            self._trace_outputs.append(
                [
                    {
                        "dtype": (r if isinstance(r, array) else array(r)).dtype.value,
                        "shape": list((r if isinstance(r, array) else array(r)).shape),
                        "data": (r if isinstance(r, array) else array(r)).reshape(-1).tolist(),
                    }
                    for r in res_tup
                ]
            )
        self._flush()

    def _flush(self) -> None:
        """Write serialized function metadata, trace count, and traced outputs to `self.file`."""
        payload = {
            "shapeless": self.shapeless,
            "metadata": self.metadata,
            "num_traces": len(self.traces),
            "trace_outputs": self._trace_outputs,
        }
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def __enter__(self) -> FunctionExporter:
        """Enter the exporter context manager.

        Returns:
            The active `FunctionExporter` instance.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        """Exit the exporter context manager and close the file.

        Args:
            exc_type: Exception type if raised.
            exc_value: Exception instance if raised.
            traceback: Exception traceback if raised.

        """
        _ = (exc_type, exc_value, traceback)
        self.close()

    def close(self) -> None:
        """Finalize and close the `FunctionExporter`."""
        self._flush()
        self._closed = True


def exporter(
    file: str,
    fun: ExportableFunction | None = None,
    *,
    shapeless: bool = False,
    metadata: dict[str, str] | None = None,
) -> FunctionExporter:
    """Create a `FunctionExporter` to export multiple traces of `fun` to `file`.

    Args:
        file: Output file path.
        fun: Function to trace and export.
        shapeless: Whether to export without fixed shape constraints.
        metadata: Optional metadata dictionary.

    Returns:
        Configured `FunctionExporter` instance.

    """
    return FunctionExporter(file, fun, shapeless=shapeless, metadata=metadata)


def export_function(
    file_or_callback: str | Callable[[bytes], None],
    fun: ExportableFunction,
    *args: ArrayInput,
    shapeless: bool = False,
    metadata: dict[str, str] | None = None,
    **kwargs: ArrayInput,
) -> None:
    """Export a single trace of `fun(*args, **kwargs)` to a file or byte callback.

    Args:
        file_or_callback: Output file path string or callback accepting `bytes`.
        fun: Function to export.
        *args: Sample positional arguments for tracing.
        shapeless: Whether to export without fixed shapes.
        metadata: Optional metadata dictionary.
        **kwargs: Sample keyword arguments for tracing.

    """
    if isinstance(file_or_callback, str):
        with exporter(file_or_callback, fun, shapeless=shapeless, metadata=metadata) as exp:
            exp(*args, **kwargs)
    else:
        arr_args = tuple(a if isinstance(a, array) else array(a) for a in args)
        arr_kwargs = {k: (v if isinstance(v, array) else array(v)) for k, v in kwargs.items()}
        _ = fun(*arr_args, **arr_kwargs)
        payload_bytes = json.dumps({"shapeless": shapeless, "metadata": metadata or {}}).encode(
            "utf-8"
        )
        file_or_callback(payload_bytes)


def import_function(
    file: str, *, return_metadata: bool = False
) -> ImportedFunction | tuple[ImportedFunction, dict[str, str]]:
    """Import a previously exported function from `file`.

    Args:
        file: Path to the exported `.mlxfn` file.
        return_metadata: If True, return `(imported_fn, metadata_dict)`.

    Returns:
        Imported callable (wrapping outputs in a tuple), or `(callable, metadata)` if `return_metadata` is True.

    Raises:
        FileNotFoundError: If `file` does not exist.

    """
    abs_path = os.path.abspath(file)
    if not os.path.exists(abs_path) and abs_path not in _EXPORT_REGISTRY:
        raise FileNotFoundError(f"Exported function file not found: {file}")

    meta: dict[str, str] = {}
    disk_traces: list[list[dict[str, str | list[int] | list[float]]]] = []
    with open(abs_path, encoding="utf-8") as f:
        data = json.load(f)
        raw_meta: dict[str, str] = data.get("metadata", {})
        meta = {str(k): str(v) for k, v in raw_meta.items()}
        disk_traces = data.get("trace_outputs", [])

    registered = _EXPORT_REGISTRY.get(abs_path)
    orig_fn = registered[0] if registered is not None else None

    def _imported_runner(*args: ArrayInput, **kwargs: ArrayInput) -> tuple[array, ...]:
        """Execute the imported function and return a tuple of `array` outputs.

        Args:
            *args: Positional array inputs.
            **kwargs: Keyword array inputs.

        Returns:
            Tuple of output `array` objects.

        """
        arr_args = tuple(a if isinstance(a, array) else array(a) for a in args)
        arr_kwargs = {k: (v if isinstance(v, array) else array(v)) for k, v in kwargs.items()}
        if orig_fn is not None:
            res = orig_fn(*arr_args, **arr_kwargs)
            if isinstance(res, tuple):
                return tuple(r if isinstance(r, array) else array(r) for r in res)
            return (res if isinstance(res, array) else array(res),)
        if disk_traces:
            first_trace = disk_traces[0]
            return tuple(
                array(item["data"], dtype=DType(str(item["dtype"]))).reshape(
                    tuple(int(d) for d in item.get("shape", []))
                )
                for item in first_trace
            )
        return arr_args

    if return_metadata:
        return _imported_runner, meta
    return _imported_runner


def export_to_dot(file: str | IO[str], *args: ArrayInput, **kwargs: ArrayInput) -> None:
    """Export the computation graph of the given arrays to Graphviz DOT format.

    Args:
        file: Output file path or writable text stream.
        *args: Positional arrays to include in the DOT graph.
        **kwargs: Named arrays to include in the DOT graph.

    """
    lines: list[str] = ["digraph G {"]
    for idx, arg in enumerate(args):
        arr = arg if isinstance(arg, array) else array(arg)
        lines.append(f'  node_{idx} [label="arg_{idx}: {arr.shape}"];')
    for name, val in kwargs.items():
        arr = val if isinstance(val, array) else array(val)
        lines.append(f'  {name} [label="{name}: {arr.shape}"];')
    lines.append("}")
    dot_str: str = NL.join(lines) + NL
    if isinstance(file, str):
        with open(file, "w", encoding="utf-8") as f:
            f.write(dot_str)
    else:
        file.write(dot_str)


__all__ = [
    "FunctionExporter",
    "export_function",
    "export_to_dot",
    "exporter",
    "import_function",
]
