"""Linear algebra operations (`mlx.core.linalg`) for zero_mlx."""

from __future__ import annotations

from typing import Literal, Sequence

import ml_switcheroo_compiler.ops as sops
import ml_switcheroo_compiler.ops.linalg.decompositions as dec
import ml_switcheroo_compiler.ops.shape.pad_and_tile as pt

from zero_mlx._types import ArrayInput, AxesLike, StreamOrDevice
from zero_mlx.array import _to_tensor, array
from zero_mlx.dtypes import DType, float32, int32


def _as_array(x: ArrayInput) -> array:
    """Ensure `x` is a `zero_mlx.array`.

    Args:
        x: Input array or convertible object.

    Returns:
        `zero_mlx.array` instance.

    """
    return x if isinstance(x, array) else array(x)


def norm(
    a: ArrayInput,
    ord: int | float | str | None = None,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the matrix or vector norm of `a`.

    Args:
        a: Input array.
        ord: Order of the norm (`None`, `'fro'`, `'nuc'`, `inf`, `-inf`, `0`, `1`, `-1`, `2`, `-2`, or float).
        axis: Axis or 2-tuple of axes along which to compute the norm.
        keepdims: If True, the normed axes are left in the result as dimensions with size one.
        stream: Optional execution stream or device.

    Returns:
        Norm of the array(s).

    """
    _ = stream
    arr = _as_array(a)
    res = dec.norm(arr._tensor, ord=ord, axis=axis, keepdims=keepdims)
    return array(res)


def matrix_norm(
    a: ArrayInput,
    ord: int | float | str | None = "fro",
    axis: tuple[int, int] = (-2, -1),
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the matrix norm of a 2D (or batched 2D) array.

    Args:
        a: Input array with at least 2 dimensions.
        ord: Order of the matrix norm (`'fro'`, `'nuc'`, `1`, `-1`, `2`, `-2`, `inf`, `-inf`).
        axis: 2-tuple of axes specifying the 2D matrices.
        keepdims: Whether to retain reduced dimensions of size 1.
        stream: Optional execution stream or device.

    Returns:
        Matrix norm `array`.

    Raises:
        ValueError: If `a` has fewer than 2 dimensions or `ord` is invalid.

    """
    arr = _as_array(a)
    if arr.ndim < 2 or len(axis) != 2:
        raise ValueError("matrix_norm requires at least a 2D array and a 2-tuple axis")
    import zero_mlx.ops as _ops

    if ord is None or ord == "fro":
        return _ops.sqrt(
            _ops.sum(_ops.square(_ops.abs(arr)), axis=axis, keepdims=keepdims, stream=stream)
        )
    ax0 = axis[0] % arr.ndim
    ax1 = axis[1] % arr.ndim
    if ord == 1:
        col_sum = _ops.sum(_ops.abs(arr), axis=ax0, keepdims=True, stream=stream)
        res = _ops.max(col_sum, axis=ax1, keepdims=True, stream=stream)
        return res if keepdims else _ops.squeeze(res, axis=(ax0, ax1), stream=stream)
    if ord == -1:
        col_sum = _ops.sum(_ops.abs(arr), axis=ax0, keepdims=True, stream=stream)
        res = _ops.min(col_sum, axis=ax1, keepdims=True, stream=stream)
        return res if keepdims else _ops.squeeze(res, axis=(ax0, ax1), stream=stream)
    if ord == float("inf"):
        row_sum = _ops.sum(_ops.abs(arr), axis=ax1, keepdims=True, stream=stream)
        res = _ops.max(row_sum, axis=ax0, keepdims=True, stream=stream)
        return res if keepdims else _ops.squeeze(res, axis=(ax0, ax1), stream=stream)
    if ord == float("-inf"):
        row_sum = _ops.sum(_ops.abs(arr), axis=ax1, keepdims=True, stream=stream)
        res = _ops.min(row_sum, axis=ax0, keepdims=True, stream=stream)
        return res if keepdims else _ops.squeeze(res, axis=(ax0, ax1), stream=stream)
    if ord in ("nuc", 2, -2):
        other_axes = [i for i in range(arr.ndim) if i not in (ax0, ax1)]
        perm = (*other_axes, ax0, ax1)
        moved = _ops.transpose(arr, axes=perm, stream=stream)
        s_vals = svd(moved, compute_uv=False, stream=stream)
        s_arr = s_vals if isinstance(s_vals, array) else s_vals[1]
        if ord == "nuc":
            reduced = _ops.sum(s_arr, axis=-1, keepdims=False, stream=stream)
        elif ord == 2:
            reduced = _ops.max(s_arr, axis=-1, keepdims=False, stream=stream)
        else:
            reduced = _ops.min(s_arr, axis=-1, keepdims=False, stream=stream)
        if not keepdims:
            return reduced
        out_shape = [1 if i in (ax0, ax1) else arr.shape[i] for i in range(arr.ndim)]
        return reduced.reshape(tuple(out_shape))
    raise ValueError(f"Invalid norm order '{ord}' for matrix_norm")


def qr(a: ArrayInput, *, stream: StreamOrDevice = None) -> tuple[array, array]:
    """Compute the QR factorization of a matrix `a = Q @ R`.

    Args:
        a: Input matrix of shape `(..., M, N)` with floating-point dtype.
        stream: Optional execution stream or device.

    Returns:
        Tuple `(Q, R)` of orthogonal/unitary `Q` and upper-triangular `R`.

    Raises:
        ValueError: If `a` is not at least 2D or has an integer dtype.

    """
    _ = stream
    arr = _as_array(a)
    if arr.ndim < 2:
        raise ValueError("QR requires at least 2D array")
    if (
        arr.dtype.value.startswith("int")
        or arr.dtype.value.startswith("uint")
        or arr.dtype == DType.bool_
    ):
        raise ValueError("QR requires float types")
    res = dec.qr(arr._tensor)
    return (array(res[0]), array(res[1]))


def svd(
    a: ArrayInput,
    compute_uv: bool = True,
    *,
    stream: StreamOrDevice = None,
) -> array | tuple[array, array, array]:
    """Compute the Singular Value Decomposition (SVD) of `a`.

    Args:
        a: Input array of shape `(..., M, N)`.
        compute_uv: Whether to compute `U` and `Vt` in addition to `S`.
        stream: Optional execution stream or device.

    Returns:
        Tuple `(U, S, Vt)` if `compute_uv` is True, otherwise singular values `S`.

    """
    _ = stream
    arr = _as_array(a)
    res = dec.svd(arr._tensor)
    u_arr = array(res[0])
    s_arr = array(res[1]).real
    vt_arr = array(res[2])
    if not compute_uv:
        return s_arr
    return (u_arr, s_arr, vt_arr)


def inv(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the multiplicative inverse of a square matrix `a`.

    Args:
        a: Square matrix or batch of square matrices.
        stream: Optional execution stream or device.

    Returns:
        Matrix inverse `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(dec.inv(arr._tensor))


def tri_inv(
    a: ArrayInput,
    upper: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the inverse of a triangular square matrix `a`.

    Args:
        a: Lower or upper triangular matrix.
        upper: True if `a` is upper triangular, False if lower triangular.
        stream: Optional execution stream or device.

    Returns:
        Triangular inverse `array`.

    """
    _ = stream
    arr = _as_array(a)
    res = dec.tri_inv(arr._tensor, lower=not upper)
    res_tri = pt.triu(res) if upper else pt.tril(res)
    return array(res_tri)


def cholesky(
    a: ArrayInput,
    upper: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the Cholesky decomposition of a symmetric positive-definite matrix `a`.

    Args:
        a: Symmetric positive-definite matrix.
        upper: If True, return upper-triangular factor `U`; otherwise lower-triangular `L`.
        stream: Optional execution stream or device.

    Returns:
        Cholesky factor `array`.

    """
    _ = stream
    arr = _as_array(a)
    res = array(dec.cholesky(arr._tensor))
    return res.T if upper else res


def cholesky_inv(
    a: ArrayInput,
    upper: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the inverse of a symmetric positive-definite matrix from its Cholesky factor `a`.

    Args:
        a: Cholesky factor (`L` or `U`).
        upper: True if `a` is upper triangular, False if lower triangular.
        stream: Optional execution stream or device.

    Returns:
        Inverse of the original positive-definite matrix.

    """
    t_inv = tri_inv(a, upper=upper, stream=stream)
    return (t_inv @ t_inv.T) if upper else (t_inv.T @ t_inv)


def pinv(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the Moore-Penrose pseudoinverse of a matrix `a`.

    Args:
        a: Input matrix.
        stream: Optional execution stream or device.

    Returns:
        Pseudoinverse `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(dec.pinv(arr._tensor))


def cross(
    a: ArrayInput,
    b: ArrayInput,
    axis: int = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the cross product of two 3-element vectors along `axis`.

    Args:
        a: First vector array.
        b: Second vector array.
        axis: Axis defining the 3-element vectors.
        stream: Optional execution stream or device.

    Returns:
        Cross product `array`.

    """
    _ = stream
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    if axis not in (-1, arr_a.ndim - 1):
        arr_a = arr_a.moveaxis(axis, -1)
        arr_b = arr_b.moveaxis(axis, -1)
        res = array(sops.linalg.cross(arr_a._tensor, arr_b._tensor))
        return res.moveaxis(-1, axis)
    return array(sops.linalg.cross(arr_a._tensor, arr_b._tensor))


def eig(a: ArrayInput, *, stream: StreamOrDevice = None) -> tuple[array, array]:
    """Compute the eigenvalues and right eigenvectors of a square matrix `a`.

    Args:
        a: Square matrix.
        stream: Optional execution stream or device.

    Returns:
        Tuple `(eigenvalues, eigenvectors)`.

    """
    _ = stream
    arr = _as_array(a)
    res = dec.eig(arr._tensor)
    return (array(res[0]), array(res[1]))


def eigvals(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the eigenvalues of a general square matrix `a`.

    Args:
        a: Square matrix.
        stream: Optional execution stream or device.

    Returns:
        Eigenvalues `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(dec.eigvals(arr._tensor))


def eigh(
    a: ArrayInput,
    UPLO: Literal["L", "U"] = "L",
    *,
    stream: StreamOrDevice = None,
) -> tuple[array, array]:
    """Compute the eigenvalues and eigenvectors of a complex Hermitian or real symmetric matrix.

    Args:
        a: Hermitian or symmetric matrix.
        UPLO: `'L'` to use the lower triangle or `'U'` to use the upper triangle.
        stream: Optional execution stream or device.

    Returns:
        Tuple `(eigenvalues, eigenvectors)`.

    """
    _ = stream
    arr = _as_array(a)
    res = dec.eigh(arr._tensor, UPLO=UPLO)
    return (array(res[0]), array(res[1]))


def eigvalsh(
    a: ArrayInput,
    UPLO: Literal["L", "U"] = "L",
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

    Args:
        a: Hermitian or symmetric matrix.
        UPLO: `'L'` to use the lower triangle or `'U'` to use the upper triangle.
        stream: Optional execution stream or device.

    Returns:
        Real eigenvalues `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(dec.eigvalsh(arr._tensor, UPLO=UPLO))


def lu(a: ArrayInput, *, stream: StreamOrDevice = None) -> tuple[array, array, array]:
    """Compute the PLU decomposition of a matrix `a`.

    Args:
        a: Input matrix of shape `(..., M, N)` with floating-point dtype.
        stream: Optional execution stream or device.

    Returns:
        Tuple `(P, L, U)` where `P` contains pivot indices, `L` is lower triangular, and `U` is upper triangular.

    Raises:
        ValueError: If `a` is fewer than 2D or has an integer dtype.

    """
    _ = stream
    arr = _as_array(a)
    if arr.ndim < 2:
        raise ValueError("LU requires at least 2D array")
    if (
        arr.dtype.value.startswith("int")
        or arr.dtype.value.startswith("uint")
        or arr.dtype == DType.bool_
    ):
        raise ValueError("LU requires float types")
    res = dec.lu(arr._tensor)
    p_mat = array(res[0])
    p_idx = p_mat.argmax(axis=-1).astype(int32) if p_mat.ndim >= 2 else p_mat.astype(int32)
    return (p_idx, array(res[1]), array(res[2]))


def lu_factor(a: ArrayInput, *, stream: StreamOrDevice = None) -> tuple[array, array]:
    """Compute a compact LU factorization `(LU, pivots)` of a matrix `a`.

    Args:
        a: Input matrix of shape `(..., M, N)`.
        stream: Optional execution stream or device.

    Returns:
        Tuple `(LU, pivots)` containing the packed LU matrix and pivot indices.

    """
    _ = stream
    arr = _as_array(a)
    res = dec.lu_factor(arr._tensor)
    return (array(res[0]), array(res[1]).astype(int32))


def lu_solve(
    lu_and_piv: ArrayInput | tuple[ArrayInput, ArrayInput],
    b: ArrayInput,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Solve a linear system given an LU factorization and right-hand side `b`.

    Args:
        lu_and_piv: Factored LU matrix or `(lu, pivots)` tuple.
        b: Right-hand side tensor.
        stream: Optional execution stream or device.

    Returns:
        Solution `array`.

    Raises:
        ValueError: If `lu` has fewer than 2 dimensions or is not square.

    """
    _ = stream
    b_t, _ = _to_tensor(b)
    if isinstance(lu_and_piv, tuple) and len(lu_and_piv) == 2:
        lu_t, _ = _to_tensor(lu_and_piv[0])
        piv_t, _ = _to_tensor(lu_and_piv[1])
    else:
        lu_t, _ = _to_tensor(lu_and_piv)
        piv_t = sops.arange(0, lu_t.shape[-1] if len(lu_t.shape) >= 1 else 0)
    if len(lu_t.shape) < 2 or lu_t.shape[-1] != lu_t.shape[-2]:
        raise ValueError("lu_solve requires a square LU matrix of at least 2 dimensions")
    return array(dec.lu_solve((lu_t, piv_t), b_t))


def solve(
    a: ArrayInput,
    b: ArrayInput,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Solve a linear system of equations `a @ x = b` for `x`.

    Args:
        a: Coefficient square matrix `a`.
        b: Ordinate / right-hand side `b`.
        stream: Optional execution stream or device.

    Returns:
        Solution `array` `x`.

    """
    _ = stream
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    return array(dec.solve(arr_a._tensor, arr_b._tensor))


def solve_triangular(
    a: ArrayInput,
    b: ArrayInput,
    *,
    upper: bool = False,
    stream: StreamOrDevice = None,
) -> array:
    """Solve the triangular linear system `a @ x = b` for `x`.

    Args:
        a: Triangular coefficient matrix.
        b: Right-hand side array.
        upper: True if `a` is upper-triangular, False if lower-triangular.
        stream: Optional execution stream or device.

    Returns:
        Solution `array` `x`.

    """
    _ = stream
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    return array(dec.solve_triangular(arr_a._tensor, arr_b._tensor, lower=not upper))


def det(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the determinant of a square matrix `a`.

    Args:
        a: Square matrix of shape `(..., M, M)`.
        stream: Optional execution stream or device.

    Returns:
        Determinant `array`.

    Raises:
        ValueError: If `a` is not at least 2D, not square, or has complex dtype.

    """
    _ = stream
    arr = _as_array(a)
    if arr.ndim < 2:
        raise ValueError("det/slogdet requires at least 2D array")
    if arr.shape[-1] != arr.shape[-2]:
        raise ValueError("det/slogdet requires square matrices")
    if "complex" in arr.dtype.value:
        raise ValueError("complex not supported")
    out_dt = arr.dtype if "float" in arr.dtype.value else float32
    return array(dec.det(arr._tensor), dtype=out_dt)


def slogdet(a: ArrayInput, *, stream: StreamOrDevice = None) -> tuple[array, array]:
    """Compute the sign and natural logarithm of the absolute determinant of `a`.

    Args:
        a: Square matrix of shape `(..., M, M)`.
        stream: Optional execution stream or device.

    Returns:
        Tuple `(sign, logabsdet)`.

    Raises:
        ValueError: If `a` is not at least 2D, not square, or has complex dtype.

    """
    _ = stream
    arr = _as_array(a)
    if arr.ndim < 2:
        raise ValueError("det/slogdet requires at least 2D array")
    if arr.shape[-1] != arr.shape[-2]:
        raise ValueError("det/slogdet requires square matrices")
    if "complex" in arr.dtype.value:
        raise ValueError("complex not supported")
    out_dt = arr.dtype if "float" in arr.dtype.value else float32
    res = dec.slogdet(arr._tensor)
    return (array(res[0], dtype=out_dt), array(res[1], dtype=out_dt))


__all__ = [
    "cholesky",
    "cholesky_inv",
    "cross",
    "det",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "inv",
    "lu",
    "lu_factor",
    "lu_solve",
    "matrix_norm",
    "norm",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "solve_triangular",
    "svd",
    "tri_inv",
]
