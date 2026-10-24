"""Device, Stream, and memory management abstractions for zero_mlx."""

from __future__ import annotations

from enum import Enum
from types import TracebackType
from typing import Union

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device as SwitcherooDevice
from ml_switcheroo_compiler.core.device import DeviceType as SwitcherooDeviceType


class DeviceType(Enum):
    """Enumeration of supported compute device types (`cpu` and `gpu`)."""

    cpu = "cpu"
    gpu = "gpu"

    def __str__(self) -> str:
        """Return the device type name as a string.

        Returns:
            Device type name (`cpu` or `gpu`).

        """
        return self.name

    def __repr__(self) -> str:
        """Return the official representation of the device type.

        Returns:
            String representation (`DeviceType.cpu` or `DeviceType.gpu`).

        """
        return f"DeviceType.{self.name}"

    def __eq__(self, other: object) -> bool:
        """Compare equality against another `DeviceType` or `Device`.

        Args:
            other: Another `DeviceType` or `Device` instance.

        Returns:
            True if the device types match.

        """
        if isinstance(other, DeviceType):
            return self.name == other.name
        if isinstance(other, Device):
            return self.name == other.type.name
        return False

    def __hash__(self) -> int:
        """Compute hash for the `DeviceType`.

        Returns:
            Hash of the device type name.

        """
        return hash(self.name)


cpu: DeviceType = DeviceType.cpu
gpu: DeviceType = DeviceType.gpu


class Device:
    """Representation of a specific compute device and index."""

    def __init__(
        self, type: DeviceType | Device | str | int = DeviceType.cpu, index: int = 0
    ) -> None:
        """Initialize a `Device` instance.

        Args:
            type: Device type (`DeviceType.cpu`, `DeviceType.gpu`, `Device`, or string).
            index: Numeric device index. Defaults to 0.

        Raises:
            ValueError: If `type` cannot be converted to a valid `DeviceType`.

        """
        if isinstance(type, Device):
            self.type: DeviceType = type.type
            self.index: int = type.index
        elif isinstance(type, DeviceType):
            self.type = type
            self.index = index
        elif isinstance(type, str):
            clean = type.lower()
            if clean in ("cpu", "gpu"):
                self.type = DeviceType(clean)
                self.index = index
            else:
                raise ValueError(f"Invalid device type: {type}")
        else:
            raise ValueError(f"Invalid device type: {type}")

    def __eq__(self, other: object) -> bool:
        """Compare equality against another `Device` or `DeviceType`.

        Args:
            other: Other device or device type to compare with.

        Returns:
            True if both represent the same device type and index.

        """
        if isinstance(other, DeviceType):
            return self.type == other
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        return False

    def __hash__(self) -> int:
        """Compute hash value for the device.

        Returns:
            Integer hash of `(self.type, self.index)`.

        """
        return hash((self.type, self.index))

    def __str__(self) -> str:
        """Return the human-readable string representation of the device.

        Returns:
            String formatted as `Device(cpu, 0)` or `Device(gpu, 0)`.

        """
        return f"Device({self.type.name}, {self.index})"

    def __repr__(self) -> str:
        """Return the official string representation of the device.

        Returns:
            String formatted as `Device(cpu, 0)` or `Device(gpu, 0)`.

        """
        return str(self)


class Stream:
    """A stream for running operations on a given device."""

    def __init__(self, device: Device | DeviceType = cpu, index: int = 0) -> None:
        """Initialize a `Stream` bound to a specific device.

        Args:
            device: Target `Device` or `DeviceType`.
            index: Stream identifier index.

        """
        self.device: Device = device if isinstance(device, Device) else Device(device, 0)
        self.index: int = index
        self.old_device: Device = self.device

    def __enter__(self) -> Stream:
        """Enter the stream context and set its device as default.

        Returns:
            The active `Stream` instance.

        """
        self.old_device = default_device()
        set_default_device(self.device)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the stream context and restore the previous default device.

        Args:
            exc_type: Exception class if raised within the context.
            exc_val: Exception instance if raised within the context.
            exc_tb: Exception traceback if raised within the context.

        """
        _ = (exc_type, exc_val, exc_tb)
        set_default_device(self.old_device)

    def __eq__(self, other: object) -> bool:
        """Compare two streams for equality.

        Args:
            other: Another `Stream` or `Device` to compare with.

        Returns:
            True if both streams belong to the same device and index.

        """
        if isinstance(other, Stream):
            return self.device == other.device and self.index == other.index
        return False

    def __repr__(self) -> str:
        """Return string representation of the stream.

        Returns:
            Formatted stream string.

        """
        return f"Stream({self.device}, {self.index})"


class StreamContext(Stream):
    """A context manager for setting the current device and stream."""


class ThreadLocalStream(Stream):
    """A thread-local stream wrapper for running operations on a given device."""


StreamOrDeviceInput = Union[Stream, Device, DeviceType, str, None]

_default_device: Device = Device(DeviceType.gpu, 0)
_default_streams: dict[DeviceType, Stream] = {
    DeviceType.cpu: Stream(Device(DeviceType.cpu, 0), 0),
    DeviceType.gpu: Stream(Device(DeviceType.gpu, 0), 0),
}
_stream_counter: int = 1
_memory_limit: int = 8 * 1024 * 1024 * 1024
_cache_limit: int = 4 * 1024 * 1024 * 1024
_wired_limit: int = 0
_active_memory: int = 0
_peak_memory: int = 0
_cache_memory: int = 0


def default_device() -> Device:
    """Return the current default device.

    Returns:
        Current default `Device`.

    """
    return _default_device


def set_default_device(device: Device | DeviceType | str) -> None:
    """Set the global default device.

    Args:
        device: Target `Device`, `DeviceType`, or device name string.

    """
    global _default_device
    dev: Device = device if isinstance(device, Device) else Device(device)
    _default_device = dev
    config.default_device = SwitcherooDevice(SwitcherooDeviceType(dev.type.name.lower()), dev.index)


def default_stream(device: Device | DeviceType | str | None = None) -> Stream:
    """Return the default stream for the given device (or current default device).

    Args:
        device: Optional device or device type to query.

    Returns:
        The default `Stream` for that device.

    """
    dev: Device = (
        _default_device
        if device is None
        else (device if isinstance(device, Device) else Device(device))
    )
    return _default_streams.get(dev.type, Stream(dev, 0))


def set_default_stream(s: Stream) -> None:
    """Set the default stream for the stream's device.

    Args:
        s: The `Stream` to register as default.

    """
    _default_streams[s.device.type] = s


def new_stream(device: Device | DeviceType | str | None = None) -> Stream:
    """Create a new stream on the given device (or current default device).

    Args:
        device: Target device or device type.

    Returns:
        Newly created `Stream`.

    """
    global _stream_counter
    dev: Device = (
        _default_device
        if device is None
        else (device if isinstance(device, Device) else Device(device))
    )
    s = Stream(dev, _stream_counter)
    _stream_counter += 1
    return s


def new_thread_local_stream(
    d: Device | DeviceType | str | None = None,
) -> ThreadLocalStream:
    """Create a new thread-local stream on the given device.

    Args:
        d: Target device or device type.

    Returns:
        Newly created `ThreadLocalStream`.

    """
    global _stream_counter
    dev: Device = _default_device if d is None else (d if isinstance(d, Device) else Device(d))
    s = ThreadLocalStream(dev, _stream_counter)
    _stream_counter += 1
    return s


def new_thread_unsafe_stream(
    d: Device | DeviceType | str | None = None,
) -> Stream:
    """Create a new thread-unsafe stream on the given device.

    Args:
        d: Target device or device type.

    Returns:
        Newly created `Stream`.

    """
    return new_stream(d)


def stream(s: Stream | Device | DeviceType | str) -> StreamContext:
    """Create a `StreamContext` for a given stream or device.

    Args:
        s: Target `Stream`, `Device`, or `DeviceType`.

    Returns:
        Context manager activating the specified stream/device.

    """
    if isinstance(s, Stream):
        return StreamContext(s.device, s.index)
    dev = s if isinstance(s, Device) else Device(s)
    return StreamContext(dev, 0)


def synchronize(stream_arg: Stream | Device | DeviceType | None = None) -> None:
    """Synchronize the given stream (or the default stream if None).

    Args:
        stream_arg: Optional `Stream` or `Device` to synchronize.

    """
    _ = stream_arg


def clear_streams() -> None:
    """Clear all non-default cached streams."""
    global _stream_counter
    _stream_counter = 1


def is_available(device_type: Device | DeviceType | str = gpu) -> bool:
    """Check whether a given device type is available.

    Args:
        device_type: Device or device type to query.

    Returns:
        True if the device is available.

    """
    _ = device_type
    return True


def device_count(device_type: Device | DeviceType | str | None = None) -> int:
    """Return the number of available devices of the specified type.

    Args:
        device_type: Optional device type to query.

    Returns:
        Number of available devices (1).

    """
    _ = device_type
    return 1


def device_info(
    d: Device | DeviceType | str | None = None,
) -> dict[str, str | int]:
    """Return hardware metadata dictionary for the specified device.

    Args:
        d: Device or device type to query. Defaults to the default device.

    Returns:
        Dictionary containing `architecture`, `memory_size`, and `device_name`.

    """
    dev = _default_device if d is None else (d if isinstance(d, Device) else Device(d))
    return {
        "architecture": "zero_mlx_virtual",
        "memory_size": _memory_limit,
        "device_name": f"zero_mlx_{dev.type.name}",
    }


def get_active_memory() -> int:
    """Return the actively allocated memory in bytes.

    Returns:
        Active memory in bytes.

    """
    return _active_memory


def get_peak_memory() -> int:
    """Return the peak allocated memory in bytes.

    Returns:
        Peak memory in bytes.

    """
    return _peak_memory


def reset_peak_memory() -> None:
    """Reset the peak memory counter to zero."""
    global _peak_memory
    _peak_memory = 0


def get_cache_memory() -> int:
    """Return the currently cached memory in bytes.

    Returns:
        Cache memory in bytes.

    """
    return _cache_memory


def clear_cache() -> None:
    """Clear the memory cache."""
    global _cache_memory
    _cache_memory = 0


def set_memory_limit(limit: int) -> int:
    """Set the memory limit in bytes and return the previous limit.

    Args:
        limit: New memory limit in bytes.

    Returns:
        Previous memory limit in bytes.

    """
    global _memory_limit
    prev: int = _memory_limit
    _memory_limit = limit
    return prev


def get_memory_limit() -> int:
    """Return the current memory limit in bytes.

    Returns:
        Current memory limit in bytes.

    """
    return _memory_limit


def set_cache_limit(limit: int) -> int:
    """Set the free cache limit in bytes and return the previous limit.

    Args:
        limit: New cache limit in bytes.

    Returns:
        Previous cache limit in bytes.

    """
    global _cache_limit
    prev: int = _cache_limit
    _cache_limit = limit
    return prev


def set_wired_limit(limit: int) -> int:
    """Set the wired memory limit in bytes and return the previous limit.

    Args:
        limit: New wired limit in bytes.

    Returns:
        Previous wired limit in bytes.

    """
    global _wired_limit
    prev: int = _wired_limit
    _wired_limit = limit
    return prev
