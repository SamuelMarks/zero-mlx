"""Device abstractions for zero_mlx."""

from typing import Any
from ml_switcheroo.core.config import config


from enum import Enum


class DeviceType(Enum):
    """Device type enum."""

    cpu = "cpu"
    gpu = "gpu"

    def __str__(self):
        """Return the device name as string."""
        return self.name

    def __repr__(self):
        """Return the device name as representation."""
        return self.name  # pragma: no cover

    def __eq__(self, other):
        """Check equality with another device type.

        Args:
            other: The other device type.

        """
        if isinstance(other, Device):
            return self.name == other.type.name
        if isinstance(other, DeviceType):
            return self.name == other.name
        return False  # pragma: no cover


cpu = DeviceType.cpu
gpu = DeviceType.gpu


class Device:
    """Device class."""

    def __init__(self, type, index=0):
        """Initialize the device.

        Args:
            type: The device type.
            index: The device index.

        """
        if hasattr(
            type, "type"
        ):  # Handles mx.Device(mx.cpu) vs mx.Device(mx.Device(mx.cpu))
            self.type = type.type  # pragma: no cover
            self.index = type.index  # pragma: no cover
        else:
            self.type = type
            self.index = index

    def __eq__(self, other):
        """Check equality with another device type.

        Args:
            other: The other device type.

        """
        if isinstance(other, DeviceType):
            return self.type.name == other.name
        return self.type == getattr(other, "type", None) and self.index == getattr(
            other, "index", None
        )

    def __str__(self):
        """Return the device name as string."""
        return f"Device({self.type}, {self.index})"

    def __repr__(self):
        """Return the device name as representation."""
        return str(self)  # pragma: no cover


_default_device = Device(gpu)


def default_device() -> Any:
    """Get the default device."""
    return _default_device


def set_default_device(device: Any) -> None:
    """Set the default device."""
    global _default_device
    _default_device = device if isinstance(device, Device) else Device(device)


class Stream:
    """Stream context."""

    def __init__(self, device):
        """Initialize stream context.

        Args:
            device: The device to stream on.

        """
        self.device = device
        self.old_device = None

    def __enter__(self) -> "Stream":
        """Enter stream context."""
        global _default_device
        self.old_device = _default_device
        set_default_device(self.device)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit stream context.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.

        """
        global _default_device
        if self.old_device is not None:
            _default_device = self.old_device


def stream(device_or_stream) -> Stream:
    """Create a stream context."""
    if isinstance(device_or_stream, Stream):
        return device_or_stream  # pragma: no cover
    return Stream(device_or_stream)


def new_stream(device):
    """Create a new stream context.

    Args:
        device: The device to create the stream for.

    """
    return Stream(device)


def default_stream(device):
    """Get default stream for the device.

    Args:
        device: The device to query.

    """
    return Stream(device)


def clear_streams() -> None:
    """Clear cached streams."""
    pass  # pragma: no cover


def is_available(device_type):
    """Check if device type is available.

    Args:
        device_type: The device type to check.

    """
    return True


def device_count(device_type=None):
    """Get number of available devices of the given type.

    Args:
        device_type: The device type to query.

    """
    return 1


def device_info(device_type=None):
    """Get information about the devices of the given type.

    Args:
        device_type: The device type to query.

    """
    return {"architecture": "mock", "memory_size": 1024, "device_name": "mock_gpu"}
