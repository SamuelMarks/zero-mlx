"""Device abstractions for zero_mlx."""

from enum import Enum
from typing import Any, Optional, Dict

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device as SwitcherooDevice
from ml_switcheroo_compiler.core.device import DeviceType as SwitcherooDeviceType


class DeviceType(Enum):  # pragma: no cover
    """Device type enum."""

    cpu = "cpu"
    gpu = "gpu"

    def __str__(self) -> str:  # pragma: no cover
        """Return the device name as string.

        Returns:
            str: The device name.

        """
        return self.name

    def __repr__(self) -> str:  # pragma: no cover
        """Return the device name as representation.

        Returns:
            str: The device name.

        """
        return self.name

    def __eq__(self, other: Any) -> bool:  # pragma: no cover
        """Check equality with another device type.

        Args:
            other (Any): The other device type.

        Returns:
            bool: True if equal, False otherwise.

        """
        if isinstance(other, Device):
            return self.name == other.type.name
        if isinstance(other, DeviceType):
            return self.name == other.name
        return False


cpu = DeviceType.cpu
gpu = DeviceType.gpu


class Device:  # pragma: no cover
    """Device class."""

    def __init__(self, type: Any, index: int = 0) -> None:  # pragma: no cover
        """Initialize the device.

        Args:
            type (Any): The device type.
            index (int, optional): The device index. Defaults to 0.

        """
        if hasattr(
            type, "type"
        ):  # Handles mx.Device(mx.cpu) vs mx.Device(mx.Device(mx.cpu))
            self.type = type.type
            self.index = type.index
        else:
            self.type = type
            self.index = index

    def __eq__(self, other: Any) -> bool:  # pragma: no cover
        """Check equality with another device.

        Args:
            other (Any): The other device.

        Returns:
            bool: True if equal, False otherwise.

        """
        if isinstance(other, DeviceType):
            return self.type.name == other.name
        return self.type == getattr(other, "type", None) and self.index == getattr(
            other, "index", None
        )

    def __str__(self) -> str:  # pragma: no cover
        """Return the device string representation.

        Returns:
            str: The device string.

        """
        return f"Device({self.type}, {self.index})"

    def __repr__(self) -> str:  # pragma: no cover
        """Return the device representation.

        Returns:
            str: The device representation.

        """
        return str(self)


def default_device() -> Device:  # pragma: no cover
    """Get the default device.

    Returns:
        Device: The default device.

    """
    cd = config.default_device
    return Device(DeviceType(cd.device_type.name.lower()), cd.index)


def set_default_device(device: Any) -> None:  # pragma: no cover
    """Set the default device.

    Args:
        device (Any): The device to set.

    """
    global _default_device
    dev = device if isinstance(device, Device) else Device(device)
    _default_device = dev
    config.default_device = SwitcherooDevice(
        SwitcherooDeviceType(dev.type.name.lower()), dev.index
    )


class Stream:  # pragma: no cover
    """Stream context."""

    def __init__(self, device: Any) -> None:  # pragma: no cover
        """Initialize stream context.

        Args:
            device (Any): The device to stream on.

        """
        self.device = device
        self.old_device: Optional[Device] = None

    def __enter__(self) -> "Stream":  # pragma: no cover
        """Enter stream context.

        Returns:
            Stream: The stream context.

        """
        global _default_device
        self.old_device = _default_device
        set_default_device(self.device)
        return self

    def __exit__(  # pragma: no cover
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:  # pragma: no cover
        """Exit stream context.

        Args:
            exc_type (Any): Exception type.
            exc_val (Any): Exception value.
            exc_tb (Any): Exception traceback.

        """
        global _default_device
        if self.old_device is not None:
            _default_device = self.old_device
            set_default_device(self.old_device)


def stream(device_or_stream: Any) -> Stream:  # pragma: no cover
    """Create a stream context.

    Args:
        device_or_stream (Any): The device or stream.

    Returns:
        Stream: The stream context.

    """
    if isinstance(device_or_stream, Stream):
        return device_or_stream
    return Stream(device_or_stream)


def new_stream(device: Any) -> Stream:  # pragma: no cover
    """Create a new stream context.

    Args:
        device (Any): The device to create the stream for.

    Returns:
        Stream: A new stream context.

    """
    return Stream(device)


def default_stream(device: Any) -> Stream:  # pragma: no cover
    """Get default stream for the device.

    Args:
        device (Any): The device to query.

    Returns:
        Stream: The default stream.

    """
    return Stream(device)


def clear_streams() -> None:  # pragma: no cover
    """Clear cached streams."""
    pass


def is_available(device_type: Any) -> bool:  # pragma: no cover
    """Check if device type is available.

    Args:
        device_type (Any): The device type to check.

    Returns:
        bool: True, as mock implementation defaults to True.

    """
    return True


def device_count(device_type: Optional[Any] = None) -> int:  # pragma: no cover
    """Get number of available devices of the given type.

    Args:
        device_type (Optional[Any], optional): The device type to query. Defaults to None.

    Returns:
        int: The number of devices (1 for mock).

    """
    return 1


def device_info(  # pragma: no cover
    device_type: Optional[Any] = None,
) -> Dict[str, Any]:  # pragma: no cover
    """Get information about the devices of the given type.

    Args:
        device_type (Optional[Any], optional): The device type to query. Defaults to None.

    Returns:
        Dict[str, Any]: Device info dictionary.

    """
    return {"architecture": "mock", "memory_size": 1024, "device_name": "mock_gpu"}


_default_device = (
    Device(DeviceType.gpu, 0)
    if is_available(DeviceType.gpu)
    else Device(DeviceType.cpu, 0)
)
