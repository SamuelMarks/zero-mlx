"""mlx.nn.dropout module."""

from zero_mlx.nn.base import Module
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops


class Dropout(Module):
    """Applies dropout to the input."""

    def __init__(self, p: float = 0.5):
        """Initialize Dropout."""
        super().__init__()
        self.p = p

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        if not self.training or self.p == 0.0:
            return x
        # ml_switcheroo_compiler ops.dropout
        # Usually takes inputs and rate
        if hasattr(sops, "dropout"):
            out = sops.dropout(x_t, rate=self.p)
        else:
            from ml_switcheroo_compiler.ops.nn.dropout import dropout

            out = dropout(x_t, rate=self.p)
        return array(out)


class Dropout2d(Module):
    """Applies 2D dropout to the input."""

    def __init__(self, p: float = 0.5):
        """Initialize Dropout2d."""
        super().__init__()
        self.p = p

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        if not self.training or self.p == 0.0:
            return x
        if hasattr(sops, "dropout2d"):
            out = sops.dropout2d(x_t, rate=self.p)
        else:
            # Fallback to standard dropout if 2d not explicitly available
            if hasattr(sops, "dropout"):
                out = sops.dropout(x_t, rate=self.p)
            else:
                from ml_switcheroo_compiler.ops.nn.dropout import dropout

                out = dropout(x_t, rate=self.p)
        return array(out)


class Dropout3d(Module):
    """Applies 3D dropout to the input."""

    def __init__(self, p: float = 0.5):
        """Initialize Dropout3d."""
        super().__init__()
        self.p = p

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        if not self.training or self.p == 0.0:
            return x
        if hasattr(sops, "dropout3d"):
            out = sops.dropout3d(x_t, rate=self.p)
        else:
            if hasattr(sops, "dropout"):
                out = sops.dropout(x_t, rate=self.p)
            else:
                from ml_switcheroo_compiler.ops.nn.dropout import dropout

                out = dropout(x_t, rate=self.p)
        return array(out)


__all__ = ["Dropout", "Dropout2d", "Dropout3d"]
