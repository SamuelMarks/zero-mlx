"""Linear algebra operations."""

from zero_mlx.array import array
from typing import Any
import ml_switcheroo_compiler.ops as sops


def norm(a, ord=None, axis=None, keepdims=False, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.norm(
        a._tensor if hasattr(a, "_tensor") else a, ord=ord, axis=axis, keepdims=keepdims
    )
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def qr(a, stream=None):  # pragma: no cover
    """Docstring."""
    if getattr(a, "dtype", None) is not None and "int" in str(a.dtype):
        raise ValueError("QR requires float types")
    if hasattr(a, "ndim") and a.ndim < 2:
        raise ValueError("QR requires at least 2D array")

    res = sops.linalg.qr(a._tensor if hasattr(a, "_tensor") else a)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def svd(a, compute_uv=True, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.svd(a._tensor if hasattr(a, "_tensor") else a)
    if isinstance(res, tuple):
        import zero_mlx as mx

        S = array(res[1])
        if "complex" in str(S.dtype):
            S = mx.abs(S)
        if not compute_uv:
            return S
        return (array(res[0]), S, array(res[2]))
    return array(res)


def inv(a, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.inv(a._tensor if hasattr(a, "_tensor") else a)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def tri_inv(a, upper=False, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.tri_inv(
        a._tensor if hasattr(a, "_tensor") else a, lower=not upper
    )
    # Zero out the unused triangle manually to match MLX behavior
    if upper:
        res = sops.triu(res)
    else:
        res = sops.tril(res)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def cholesky(a, upper=False, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.cholesky(
        a._tensor if hasattr(a, "_tensor") else a, lower=not upper
    )
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def cholesky_inv(a, upper=False, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.cholesky_inv(
        a._tensor if hasattr(a, "_tensor") else a, lower=not upper
    )
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def pinv(a, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.pinv(a._tensor if hasattr(a, "_tensor") else a)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def cross(a, b, axis=-1, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.cross(
        a._tensor if hasattr(a, "_tensor") else a,
        b._tensor if hasattr(b, "_tensor") else b,
        axis=axis,
    )
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def eig(a, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.eig(a._tensor if hasattr(a, "_tensor") else a)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def eigvals(a, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.eigvals(a._tensor if hasattr(a, "_tensor") else a)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def eigh(a, UPLO="L", stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.eigh(a._tensor if hasattr(a, "_tensor") else a, UPLO=UPLO)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def eigvalsh(a, UPLO="L", stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.eigvalsh(a._tensor if hasattr(a, "_tensor") else a, UPLO=UPLO)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def lu(a, stream=None):  # pragma: no cover
    """Docstring."""
    if hasattr(a, "ndim") and a.ndim < 2:
        raise ValueError("LU requires at least 2D array")
    if getattr(a, "dtype", None) is not None and "int" in str(a.dtype):
        raise ValueError("LU requires float types")

    res = sops.linalg.lu(a._tensor if hasattr(a, "_tensor") else a)
    if isinstance(res, tuple):
        from zero_mlx import int32
        import zero_mlx as mx

        P_mat = array(res[0])
        if P_mat.ndim >= 2:
            # Convert permutation matrix back to indices
            P = mx.argmax(P_mat, axis=-1).astype(int32)
        else:
            P = P_mat.astype(int32)
        return (P, array(res[1]), array(res[2]))
    return array(res)


def lu_factor(a, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.lu_factor(a._tensor if hasattr(a, "_tensor") else a)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def lu_solve(lu_and_piv, b, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.lu_solve(
        lu_and_piv._tensor if hasattr(lu_and_piv, "_tensor") else lu_and_piv,
        b._tensor if hasattr(b, "_tensor") else b,
    )
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def solve(a, b, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.solve(
        a._tensor if hasattr(a, "_tensor") else a,
        b._tensor if hasattr(b, "_tensor") else b,
    )
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def solve_triangular(a, b, upper=False, stream=None):  # pragma: no cover
    """Docstring."""
    res = sops.linalg.solve_triangular(
        a._tensor if hasattr(a, "_tensor") else a,
        b._tensor if hasattr(b, "_tensor") else b,
        lower=not upper,
    )
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def det(a, stream=None):  # pragma: no cover
    """Docstring."""
    if hasattr(a, "ndim") and a.ndim < 2:
        raise ValueError("det/slogdet requires at least 2D array")
    if hasattr(a, "shape") and a.shape[-1] != a.shape[-2]:
        raise ValueError("det/slogdet requires square matrices")
    if getattr(a, "dtype", None) is not None and "complex" in str(a.dtype):
        raise ValueError("complex not supported")

    res = sops.linalg.det(a._tensor if hasattr(a, "_tensor") else a)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def slogdet(a, stream=None):  # pragma: no cover
    """Docstring."""
    if hasattr(a, "ndim") and a.ndim < 2:
        raise ValueError("det/slogdet requires at least 2D array")
    if hasattr(a, "shape") and a.shape[-1] != a.shape[-2]:
        raise ValueError("det/slogdet requires square matrices")
    if getattr(a, "dtype", None) is not None and "complex" in str(a.dtype):
        raise ValueError("complex not supported")

    res = sops.linalg.slogdet(a._tensor if hasattr(a, "_tensor") else a)
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)
