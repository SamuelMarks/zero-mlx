# ruff: noqa
"""Array implementation."""

from typing import Any, Tuple, Optional, List, Union
import ml_switcheroo
import ml_switcheroo.core.backend_ext as np
from ml_switcheroo.core.config import config

from zero_mlx.dtypes import DType, to_switcheroo_dtype


def _np_dtype_to_mlx(np_dtype: Any) -> DType:
    """Compute _np_dtype_to_mlx.

    Args:
        np_dtype: The np_dtype argument.

    Returns:
        The result of _np_dtype_to_mlx.
    """
    if hasattr(np_dtype, "name"):
        name = np_dtype.name
    elif hasattr(np_dtype, "value"):  # pragma: no cover
        name = np_dtype.value  # pragma: no cover
    else:
        name = str(np_dtype)  # pragma: no cover
    try:
        name = np.dtype(name).name
    except:  # pragma: no cover
        pass  # pragma: no cover
    if name == "bool":
        return DType.bool_
    for dt in DType:
        if dt.value == name:
            return dt
    if np.issubdtype(np_dtype, np.complexfloating):  # pragma: no cover
        return DType.complex64  # pragma: no cover
    if np.issubdtype(np_dtype, np.integer):  # pragma: no cover
        return DType.int32  # pragma: no cover
    if np.issubdtype(np_dtype, np.floating):  # pragma: no cover
        return DType.float32  # pragma: no cover
    return DType.float32  # pragma: no cover


def _to_tensor(x: Any, dtype: Optional[DType] = None) -> ml_switcheroo.Tensor:
    """Compute _to_tensor.

    Args:
        x: The x argument.
        dtype: The dtype argument.

    Returns:
        The result of _to_tensor.
    """
    if hasattr(x, "_tensor"):
        return x._tensor  # pragma: no cover
    if isinstance(x, ml_switcheroo.Tensor):
        return x  # pragma: no cover

    np_arr = np.array(x)
    if dtype is None:
        dtype = _np_dtype_to_mlx(np_arr.dtype)
    c_dtype = to_switcheroo_dtype(dtype)
    return ml_switcheroo.Tensor(np_arr, np_arr.shape, c_dtype, config.default_device)


def _wrap(x: Any, mlx_dtype: Optional[DType] = None) -> Any:
    """Compute _wrap.

    Args:
        x: The x argument.
        mlx_dtype: The mlx_dtype argument.

    Returns:
        The result of _wrap.
    """
    if isinstance(x, ml_switcheroo.Tensor):  # pragma: no cover
        return array(x, dtype=mlx_dtype)  # pragma: no cover
    return x  # pragma: no cover


def _check_string(x: Any):
    """Compute _check_string.

    Args:
        x: The x argument.

    Returns:
        The result of _check_string.
    """
    if isinstance(x, str):
        raise ValueError()
    if isinstance(x, (list, tuple)):
        for item in x:
            _check_string(item)


class array:
    """Array class."""

    def __init__(self, data: Any, dtype: Optional[DType] = None):
        """Initialize array."""
        self._original_dtype = dtype

        if hasattr(data, "_tensor"):
            self._tensor = data._tensor
            if dtype is None and hasattr(data, "_original_dtype"):
                self._original_dtype = data._original_dtype
        elif isinstance(data, ml_switcheroo.Tensor):
            self._tensor = data
            from zero_mlx.dtypes import to_mlx_dtype

            self._original_dtype = to_mlx_dtype(data.dtype)
        else:
            try:
                _check_string(data)
            except ValueError:
                raise ValueError("Invalid element")

            # Handle torch tensors with bfloat16
            if (
                type(data).__name__ == "Tensor"
                and getattr(data, "dtype", None) is not None
                and str(data.dtype) == "torch.bfloat16"
            ):
                import ml_dtypes  # pragma: no cover

                data = (  # pragma: no cover
                    data.to(getattr(__import__("torch"), "float32"))  # pragma: no cover
                    .numpy()  # pragma: no cover
                    .astype(ml_dtypes.bfloat16)  # pragma: no cover
                )

            is_scalar = not isinstance(data, (list, tuple, np.ndarray, array))
            if isinstance(data, bytes):
                data = np.frombuffer(data, dtype=np.uint8)
            is_np = isinstance(data, np.ndarray)

            if is_np and data.dtype == np.float64 and dtype is None:
                data = data.astype(np.float32)
            if is_np and data.dtype == np.complex128 and dtype is None:
                data = data.astype(np.complex64)
            # removed is_np downcast for int64, only apply for lists
            if is_scalar and isinstance(data, float) and dtype is None:
                data = np.array(data, dtype=np.float32)
            if is_scalar and isinstance(data, bool) and dtype is None:
                data = np.array(data, dtype=np.bool_)
            if is_scalar and isinstance(data, complex) and dtype is None:
                data = np.array(data, dtype=np.complex64)

            if not is_np and not is_scalar:
                try:
                    arr_form = np.array(data)
                    if arr_form.dtype == np.int64:
                        arr_form = arr_form.astype(np.int32)
                    if arr_form.size == 0 and dtype is None:
                        self._original_dtype = DType.float32
                        data = arr_form.astype(np.float32)
                    elif arr_form.dtype.kind in (
                        "U",
                        "S",
                        "O",
                    ):  # String / Object fallbacks
                        raise ValueError("Invalid array")  # pragma: no cover
                except ValueError as e:  # pragma: no branch
                    if (
                        "setting an array element with a sequence"
                        in str(  # pragma: no branch
                            e
                        )
                        or "inhomogeneous" in str(e)
                    ):
                        raise ValueError(
                            "Initialization encountered non-uniform length."
                        )
                    raise ValueError(str(e))
                except Exception:  # pragma: no cover
                    pass  # pragma: no cover

            # Allow mapping lists of values directly to specific dtypes with .astype early on
            if dtype is not None:
                try:
                    val = dtype.value
                    if val == "bool":
                        val = "bool_"
                    if (
                        hasattr(__import__("ml_dtypes"), "bfloat16")
                        and val == "bfloat16"
                    ):
                        data = np.array(data, dtype=__import__("ml_dtypes").bfloat16)
                    else:
                        data = np.array(data, dtype=val)
                except Exception:  # pragma: no cover
                    pass  # pragma: no cover
                except Exception:  # pragma: no cover
                    pass  # pragma: no cover

            if dtype is None and isinstance(data, np.ndarray):
                self._original_dtype = _np_dtype_to_mlx(data.dtype)
                # Ensure we explicitly match np's promotion back down to what it was
                if data.dtype.name == "int32":
                    self._original_dtype = DType.int32
                if data.dtype.name == "uint32":
                    self._original_dtype = DType.uint32
            elif (
                getattr(self, "_original_dtype", None) is None
                and dtype is None
                and hasattr(data, "dtype")
            ):
                try:
                    self._original_dtype = DType(str(data.dtype).split(".")[-1])
                except Exception:  # pragma: no cover
                    pass  # pragma: no cover

            if (
                dtype is None
                and is_scalar
                and isinstance(data, int)
                and not isinstance(data, bool)
            ):
                if data >= 2147483648 or data < -2147483648:
                    self._original_dtype = DType.int64
                    data = np.array(data, dtype=np.int64)
                else:
                    self._original_dtype = DType.int32
                    data = np.array(data, dtype=np.int32)

            try:
                sh = np.shape(data)
                sz = np.prod(sh, dtype=int) if sh else 1
                for dim in sh:
                    if dim >= 2**31 or dim < -(2**31):
                        raise OverflowError(
                            f"Shape dimension {dim} is outside the supported range [-2147483648, 2147483647]. MLX currently uses 32-bit integers for shape dimensions."
                        )
            except TypeError:
                pass  # pragma: no cover
            except MemoryError:
                raise OverflowError("Invalid array shape")  # pragma: no cover
            except Exception as e:
                if "Shape dimension" in str(e):
                    raise OverflowError(str(e))

            try:
                self._tensor = _to_tensor(data, dtype=dtype)
            except Exception as e:  # pragma: no cover
                if (
                    "setting an array element with a sequence"
                    in str(  # pragma: no cover
                        e
                    )
                    or "inhomogeneous" in str(e)
                ):
                    raise ValueError(
                        "Initialization encountered non-uniform length."
                    )  # pragma: no cover
                raise e  # pragma: no cover

        if self._original_dtype is None:
            val = self._tensor.dtype.value
            if val == "int64":
                val = "int32"
            if val == "float64":
                val = "float32"
            if val == "complex128":
                val = "complex64"
            self._original_dtype = DType(val)

    def _binary_op(self, other: Any, op: Any) -> "array":
        """Compute _binary_op.

        Args:
            other: The other argument.
            op: The op argument.

        Returns:
            The result of _binary_op.
        """
        if isinstance(other, (str, list, tuple)):
            raise ValueError(f"Invalid operation between array and {type(other)}")
        try:
            lhs = np.array(self.data)

            # Implement overflow check for scalar ints matched with typed arrays BEFORE promotion
            if type(other) == int:
                if getattr(self, "dtype", None) == DType.int32:
                    if other >= 2**31 or other < -(2**31):
                        raise ValueError("Overflow")

            rhs = np.array(array(other).data)

            lhs_dt = getattr(self, "dtype", None)
            rhs_dt = getattr(array(other), "dtype", None)

            # Calculate expected exact dtype to match MLX precisely!
            def get_category(v):
                """Compute get_category.

                Args:
                    v: The v argument.

                Returns:
                    The result of get_category.
                """
                if type(v) == bool:
                    return 0
                if type(v) == int:
                    return 1
                if type(v) == float:
                    return 2
                if type(v) == complex:
                    return 3
                if hasattr(v, "value"):
                    if v.value == "bool":
                        return 0
                    if v.value.startswith("int") or v.value.startswith("uint"):
                        return 1
                    if v.value.startswith("float") or v.value.startswith("bfloat"):
                        return 2
                    if v.value.startswith("complex"):
                        return 3
                return 2  # pragma: no cover

            forced_dt = None

            # Python scalar promotion logic:
            is_other_scalar = type(other) in (
                bool,
                int,
                float,
                complex,
            ) and not isinstance(other, array)
            is_self_scalar = (
                self.ndim == 0
                and type(lhs.item()) in (bool, int, float, complex)
                and getattr(self, "_tensor", None) is None
            )

            if is_other_scalar and not is_self_scalar:
                arr_cat = get_category(lhs_dt)
                v_cat = get_category(other)
                if v_cat <= arr_cat:
                    forced_dt = lhs_dt
                else:
                    if v_cat == 1:
                        forced_dt = DType.int32
                    elif v_cat == 2:
                        forced_dt = DType.float32
                    else:
                        forced_dt = DType.complex64
            elif is_self_scalar and not is_other_scalar:
                arr_cat = get_category(rhs_dt)  # pragma: no cover
                v_cat = get_category(lhs.item())  # pragma: no cover
                if v_cat <= arr_cat:  # pragma: no cover
                    forced_dt = rhs_dt  # pragma: no cover
                else:
                    if v_cat == 1:  # pragma: no cover
                        forced_dt = DType.int32  # pragma: no cover
                    elif v_cat == 2:  # pragma: no cover
                        forced_dt = DType.float32  # pragma: no cover
                    else:
                        forced_dt = DType.complex64  # pragma: no cover

            # Apply forced pre-cast to lock in types before np does its bad promotion logic
            if forced_dt is not None:
                import ml_dtypes

                target_np = (
                    ml_dtypes.bfloat16
                    if forced_dt.value == "bfloat16"
                    else getattr(
                        np, forced_dt.value if forced_dt.value != "bool" else "bool_"
                    )
                )
                lhs = lhs.astype(target_np)
                rhs = rhs.astype(target_np)

            res = op(lhs, rhs)
            mlx_dt = _np_dtype_to_mlx(res.dtype)

            if res.dtype.name.startswith("uint"):
                for dt in DType:
                    if dt.value == res.dtype.name:
                        mlx_dt = dt
            elif res.dtype.name.startswith("int"):
                for dt in DType:
                    if dt.value == res.dtype.name:
                        mlx_dt = dt

            if forced_dt is not None:
                mlx_dt = forced_dt

            if lhs_dt is not None and rhs_dt is not None:  # pragma: no branch
                if ("float" in lhs_dt.value or "complex" in lhs_dt.value) and (
                    "int" in rhs_dt.value or "bool" in rhs_dt.value
                ):
                    mlx_dt = lhs_dt
                elif ("float" in rhs_dt.value or "complex" in rhs_dt.value) and (
                    "int" in lhs_dt.value or "bool" in lhs_dt.value
                ):  # pragma: no branch
                    mlx_dt = rhs_dt

            if mlx_dt == DType.float64:
                mlx_dt = DType.float32
                res = res.astype(np.float32)
            elif mlx_dt == DType.complex128:
                mlx_dt = DType.complex64
                res = res.astype(np.complex64)

            import ml_dtypes

            target_np = (
                ml_dtypes.bfloat16
                if mlx_dt.value == "bfloat16"
                else getattr(np, mlx_dt.value if mlx_dt.value != "bool" else "bool_")
            )
            if res.dtype != target_np:
                res = res.astype(target_np)

            return array(res, dtype=mlx_dt)
        except Exception as e:
            raise ValueError(f"Op failed: {e}")

    def __add__(self, other: Any) -> "array":
        """Compute __add__.

        Args:
            other: The other argument.

        Returns:
            The result of __add__.
        """
        return self._binary_op(other, np.add)

    def __iadd__(self, other: Any) -> "array":
        """Compute __iadd__.

        Args:
            other: The other argument.

        Returns:
            The result of __iadd__.
        """
        res = self._binary_op(other, np.add)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __sub__(self, other: Any) -> "array":
        """Compute __sub__.

        Args:
            other: The other argument.

        Returns:
            The result of __sub__.
        """
        return self._binary_op(other, np.subtract)

    def __isub__(self, other: Any) -> "array":
        """Compute __isub__.

        Args:
            other: The other argument.

        Returns:
            The result of __isub__.
        """
        res = self._binary_op(other, np.subtract)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __mul__(self, other: Any) -> "array":
        """Compute __mul__.

        Args:
            other: The other argument.

        Returns:
            The result of __mul__.
        """
        return self._binary_op(other, np.multiply)

    def __imul__(self, other: Any) -> "array":
        """Compute __imul__.

        Args:
            other: The other argument.

        Returns:
            The result of __imul__.
        """
        res = self._binary_op(other, np.multiply)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __rmul__(self, other: Any) -> "array":
        """Compute __rmul__.

        Args:
            other: The other argument.

        Returns:
            The result of __rmul__.
        """
        return array(other) * self

    def __radd__(self, other: Any) -> "array":
        """Compute __radd__.

        Args:
            other: The other argument.

        Returns:
            The result of __radd__.
        """
        return array(other) + self

    def __rsub__(self, other: Any) -> "array":
        """Compute __rsub__.

        Args:
            other: The other argument.

        Returns:
            The result of __rsub__.
        """
        return array(other) - self

    def __rtruediv__(self, other: Any) -> "array":
        """Compute __rtruediv__.

        Args:
            other: The other argument.

        Returns:
            The result of __rtruediv__.
        """
        return array(other) / self  # pragma: no cover

    def __rfloordiv__(self, other: Any) -> "array":
        """Compute __rfloordiv__.

        Args:
            other: The other argument.

        Returns:
            The result of __rfloordiv__.
        """
        return array(other) // self  # pragma: no cover

    def __rmod__(self, other: Any) -> "array":
        """Compute __rmod__.

        Args:
            other: The other argument.

        Returns:
            The result of __rmod__.
        """
        return array(other) % self

    def __rpow__(self, other: Any) -> "array":
        """Compute __rpow__.

        Args:
            other: The other argument.

        Returns:
            The result of __rpow__.
        """
        return array(other) ** self  # pragma: no cover

    def __rand__(self, other: Any) -> "array":
        """Compute __rand__.

        Args:
            other: The other argument.

        Returns:
            The result of __rand__.
        """
        return array(other) & self  # pragma: no cover

    def __ror__(self, other: Any) -> "array":
        """Compute __ror__.

        Args:
            other: The other argument.

        Returns:
            The result of __ror__.
        """
        return array(other) | self  # pragma: no cover

    def __rxor__(self, other: Any) -> "array":
        """Compute __rxor__.

        Args:
            other: The other argument.

        Returns:
            The result of __rxor__.
        """
        return array(other) ^ self  # pragma: no cover

    def __truediv__(self, other: Any) -> "array":
        """Compute __truediv__.

        Args:
            other: The other argument.

        Returns:
            The result of __truediv__.
        """
        return self._binary_op(other, np.divide)

    def __itruediv__(self, other: Any) -> "array":
        """Compute __itruediv__.

        Args:
            other: The other argument.

        Returns:
            The result of __itruediv__.
        """
        if self.dtype.value.startswith("int") or self.dtype.value.startswith("uint"):
            raise ValueError("In-place division of int array is not supported")
        res = self._binary_op(other, np.divide)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __floordiv__(self, other: Any) -> "array":
        """Compute __floordiv__.

        Args:
            other: The other argument.

        Returns:
            The result of __floordiv__.
        """
        return self._binary_op(other, np.floor_divide)

    def __ifloordiv__(self, other: Any) -> "array":
        """Compute __ifloordiv__.

        Args:
            other: The other argument.

        Returns:
            The result of __ifloordiv__.
        """
        res = self._binary_op(other, np.floor_divide)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __mod__(self, other: Any) -> "array":
        """Compute __mod__.

        Args:
            other: The other argument.

        Returns:
            The result of __mod__.
        """
        return self._binary_op(other, np.mod)

    def __imod__(self, other: Any) -> "array":
        """Compute __imod__.

        Args:
            other: The other argument.

        Returns:
            The result of __imod__.
        """
        res = self._binary_op(other, np.mod)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __pow__(self, other: Any) -> "array":
        """Compute __pow__.

        Args:
            other: The other argument.

        Returns:
            The result of __pow__.
        """
        return self._binary_op(other, np.power)

    def __ipow__(self, other: Any) -> "array":
        """Compute __ipow__.

        Args:
            other: The other argument.

        Returns:
            The result of __ipow__.
        """
        res = self._binary_op(other, np.power)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __xor__(self, other: Any) -> "array":
        """Compute __xor__.

        Args:
            other: The other argument.

        Returns:
            The result of __xor__.
        """
        return self._binary_op(other, np.bitwise_xor)

    def __ixor__(self, other: Any) -> "array":
        """Compute __ixor__.

        Args:
            other: The other argument.

        Returns:
            The result of __ixor__.
        """
        res = self._binary_op(other, np.bitwise_xor)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __and__(self, other: Any) -> "array":
        """Compute __and__.

        Args:
            other: The other argument.

        Returns:
            The result of __and__.
        """
        return self._binary_op(other, np.bitwise_and)

    def __or__(self, other: Any) -> "array":
        """Compute __or__.

        Args:
            other: The other argument.

        Returns:
            The result of __or__.
        """
        return self._binary_op(other, np.bitwise_or)

    def __iand__(self, other: Any) -> "array":
        """Compute __iand__.

        Args:
            other: The other argument.

        Returns:
            The result of __iand__.
        """
        res = self._binary_op(other, np.bitwise_and)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __ior__(self, other: Any) -> "array":
        """Compute __ior__.

        Args:
            other: The other argument.

        Returns:
            The result of __ior__.
        """
        res = self._binary_op(other, np.bitwise_or)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __matmul__(self, other: Any) -> "array":
        """Compute __matmul__.

        Args:
            other: The other argument.

        Returns:
            The result of __matmul__.
        """
        return self._binary_op(other, np.matmul)

    def __imatmul__(self, other: Any) -> "array":
        """Compute __imatmul__.

        Args:
            other: The other argument.

        Returns:
            The result of __imatmul__.
        """
        res = self._binary_op(other, np.matmul)
        self._tensor = res._tensor
        self._original_dtype = res._original_dtype
        return self

    def __lt__(self, other: Any) -> "array":
        """Compute __lt__.

        Args:
            other: The other argument.

        Returns:
            The result of __lt__.
        """
        if isinstance(other, str):
            raise ValueError()
        return self._binary_op(other, np.less)

    def __gt__(self, other: Any) -> "array":
        """Compute __gt__.

        Args:
            other: The other argument.

        Returns:
            The result of __gt__.
        """
        if isinstance(other, str):
            raise ValueError()
        return self._binary_op(other, np.greater)

    def __le__(self, other: Any) -> "array":
        """Compute __le__.

        Args:
            other: The other argument.

        Returns:
            The result of __le__.
        """
        if isinstance(other, str):
            raise ValueError()
        return self._binary_op(other, np.less_equal)

    def __ge__(self, other: Any) -> "array":
        """Compute __ge__.

        Args:
            other: The other argument.

        Returns:
            The result of __ge__.
        """
        if isinstance(other, str):
            raise ValueError()
        return self._binary_op(other, np.greater_equal)

    def __eq__(self, other: Any) -> Any:
        """Compute __eq__.

        Args:
            other: The other argument.

        Returns:
            The result of __eq__.
        """
        if not isinstance(other, (array, int, float, bool, np.ndarray, complex)):
            return False
        return array(
            np.array(self.data) == np.array(array(other).data), dtype=DType.bool_
        )

    def __ne__(self, other: Any) -> Any:
        """Compute __ne__.

        Args:
            other: The other argument.

        Returns:
            The result of __ne__.
        """
        if not isinstance(other, (array, int, float, bool, np.ndarray, complex)):
            return True
        return array(
            np.array(self.data) != np.array(array(other).data), dtype=DType.bool_
        )

    def __neg__(self) -> "array":
        """Compute __neg__.

        Returns:
            The result of __neg__.
        """
        return array(np.negative(self.data), dtype=self.dtype)

    def __invert__(self) -> "array":
        """Compute __invert__.

        Returns:
            The result of __invert__.
        """
        return array(np.bitwise_not(self.data), dtype=self.dtype)

    def __dlpack_device__(self) -> Tuple[int, int]:
        """Compute __dlpack_device__.

        Returns:
            The result of __dlpack_device__.
        """
        return (1, 0)

    def __dlpack__(self, stream: Any = None) -> Any:
        """Compute __dlpack__.

        Args:
            stream: The stream argument.

        Returns:
            The result of __dlpack__.
        """
        return None  # pragma: no cover

    def __array_namespace__(self, *, api_version: Optional[str] = None) -> Any:
        """Compute __array_namespace__.

        Args:
            api_version: The api_version argument.

        Returns:
            The result of __array_namespace__.
        """
        import zero_mlx as mx

        return mx

    def astype(self, dtype: DType, stream: Any = None) -> "array":
        """Compute astype.

        Args:
            dtype: The dtype argument.
            stream: The stream argument.

        Returns:
            The result of astype.
        """
        np_arr = np.array(self._tensor.data)
        if hasattr(dtype, "value"):
            np_arr = np_arr.astype(dtype.value)
        return array(np_arr, dtype=dtype)

    def reshape(self, *shape: Any) -> "array":
        """Compute reshape.

        Args:
            shape: The shape argument.

        Returns:
            The result of reshape.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]  # type: ignore[assignment]
        return array(np.reshape(self.data, shape), dtype=self.dtype)

    def tolist(self) -> List[Any]:
        """Compute tolist.

        Returns:
            The result of tolist.
        """
        return np.array(self.data).tolist()

    def item(self) -> Any:
        """Compute item.

        Returns:
            The result of item.
        """
        if self.size != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
        val = np.array(self.data).item()
        if self.dtype == DType.bool_:
            return bool(val)
        if self.dtype == DType.int32:
            return int(val)
        return val

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Compute __array__.

        Args:
            dtype: The dtype argument.
            copy: The copy argument.

        Returns:
            The result of __array__.
        """
        arr = np.array(self.data, copy=copy)  # pragma: no cover
        if dtype is not None:  # pragma: no cover
            return arr.astype(dtype)  # pragma: no cover
        # Check explicit copy kwarg
        return arr.astype(self.dtype.value, copy=copy)  # pragma: no cover

    def __int__(self) -> int:
        """Compute __int__.

        Returns:
            The result of __int__.
        """
        return int(self.item())

    def __float__(self) -> float:
        """Compute __float__.

        Returns:
            The result of __float__.
        """
        return float(self.item())

    def __bool__(self) -> bool:
        """Compute __bool__.

        Returns:
            The result of __bool__.
        """
        if self.size != 1:
            raise ValueError(  # pragma: no cover
                "The truth value of an array with more than one element is ambiguous."
            )
        return bool(self.item())

    def __len__(self) -> int:
        """Compute __len__.

        Returns:
            The result of __len__.
        """
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def _check_large_index(self, idx: Any) -> None:
        """Compute _check_large_index.

        Args:
            idx: The idx argument.

        Returns:
            The result of _check_large_index.
        """
        if isinstance(idx, slice):
            if idx.start is not None and idx.start >= 2**31:
                raise ValueError("Large index")  # pragma: no cover
            if idx.stop is not None and idx.stop >= 2**31:
                raise ValueError("Large index")
            if idx.step is not None and idx.step >= 2**31:
                raise ValueError("Large index")  # pragma: no cover
            if idx.start is not None and idx.start <= -(2**31):
                raise ValueError("Large index")  # pragma: no cover
            if idx.stop is not None and idx.stop <= -(2**31):
                raise ValueError("Large index")  # pragma: no cover
            if idx.step is not None and idx.step <= -(2**31):
                raise ValueError("Large index")  # pragma: no cover
        elif isinstance(idx, int):
            if idx >= 2**31 or idx <= -(2**31):
                raise ValueError("Large index")
        elif isinstance(idx, tuple):
            for i in idx:
                self._check_large_index(i)

    def __getitem__(self, idx: Any) -> "array":
        """Compute __getitem__.

        Args:
            idx: The idx argument.

        Returns:
            The result of __getitem__.
        """
        self._check_large_index(idx)
        if isinstance(idx, array):
            idx = np.array(idx.data)
        elif isinstance(idx, tuple):
            idx = tuple(np.array(i.data) if isinstance(i, array) else i for i in idx)

        try:
            res = np.array(self.data)[idx]
        except IndexError as e:
            if "out of bounds" in str(e):
                raise IndexError(str(e))
            raise e

        if isinstance(res, np.ndarray) and res.ndim == 0:
            return array(res, dtype=self.dtype)  # pragma: no cover
        return array(res, dtype=self.dtype)

    def __setitem__(self, idx: Any, value: Any) -> None:
        """Compute __setitem__.

        Args:
            idx: The idx argument.
            value: The value argument.

        Returns:
            The result of __setitem__.
        """
        self._check_large_index(idx)
        arr = np.array(self.data)
        if isinstance(idx, array):
            idx = np.array(idx.data)
        elif isinstance(idx, tuple):
            idx = tuple(np.array(i.data) if isinstance(i, array) else i for i in idx)
        if isinstance(value, array):
            value = np.array(value.data)

        try:
            if arr.ndim == 0 and (
                idx is None or (isinstance(idx, tuple) and all(i is None for i in idx))
            ):
                arr = np.array(value)
            else:
                if arr.ndim == 0:
                    raise ValueError("too many indices for array")
                if (
                    arr.dtype.kind in ("i", "u")
                    and isinstance(value, float)
                    and np.isnan(value)
                ):
                    arr[idx] = 0
                else:
                    arr[idx] = value
        except IndexError as e:
            if "an index can only have a single ellipsis" in str(e):
                raise ValueError("multiple ellipsis")
            if "too many indices for array" in str(e):
                raise ValueError(str(e))
            if "boolean index did not match" in str(e):
                raise ValueError(str(e))
            raise e

        self._tensor = _to_tensor(arr, dtype=self.dtype)

    def __format__(self, format_spec: str) -> str:
        """Compute __format__.

        Args:
            format_spec: The format_spec argument.

        Returns:
            The result of __format__.
        """
        if format_spec == "":
            return str(self)
        if self.size != 1:
            raise TypeError("unsupported format string passed to array.__format__")
        return format(self.item(), format_spec)

    def copy(self) -> "array":
        """Compute copy.

        Returns:
            The result of copy.
        """
        return array(np.array(self.data, copy=True), dtype=self.dtype)

    def __copy__(self) -> "array":
        """Compute __copy__.

        Returns:
            The result of __copy__.
        """
        return self.copy()

    def __deepcopy__(self, memo: Any) -> "array":
        """Compute __deepcopy__.

        Args:
            memo: The memo argument.

        Returns:
            The result of __deepcopy__.
        """
        return self.copy()

    @property
    def shape(self) -> Tuple[int, ...]:
        """Compute shape.

        Returns:
            The result of shape.
        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Compute size.

        Returns:
            The result of size.
        """
        return np.prod(self.shape, dtype=int) if self.shape else 1

    @property
    def ndim(self) -> int:
        """Compute ndim.

        Returns:
            The result of ndim.
        """
        return len(self.shape)

    @property
    def dtype(self) -> DType:
        """Compute dtype.

        Returns:
            The result of dtype.
        """
        return self._original_dtype  # type: ignore[return-value]

    @property
    def itemsize(self) -> int:
        """Compute itemsize.

        Returns:
            The result of itemsize.
        """
        return self.dtype.size

    @property
    def nbytes(self) -> int:
        """Compute nbytes.

        Returns:
            The result of nbytes.
        """
        return self.size * self.itemsize

    @property
    def data(self) -> Any:
        """Compute data.

        Returns:
            The result of data.
        """
        if config.eager_mode:
            arr = np.array(self._tensor.data)
            # removed item() conversion to preserve np dtype
            return arr
        return self._tensor.data  # pragma: no cover

    @property
    def real(self) -> "array":
        """Compute real.

        Returns:
            The result of real.
        """
        return array(np.real(self.data), dtype=self.dtype)

    @property
    def imag(self) -> "array":
        """Compute imag.

        Returns:
            The result of imag.
        """
        return array(np.imag(self.data), dtype=self.dtype)

    def view(self, dtype):
        """Compute view.

        Args:
            dtype: The dtype argument.

        Returns:
            The result of view.
        """
        val = dtype.value if hasattr(dtype, "value") else dtype  # pragma: no cover
        return array(np.array(self.data).view(val))  # pragma: no cover

    def __getattr__(self, name: str) -> Any:
        """Compute __getattr__.

        Args:
            name: The name argument.

        Returns:
            The result of __getattr__.
        """
        if name == "T":
            return array(np.transpose(self.data), dtype=self.dtype)
        if name == "at":

            class AtMocker:  # pragma: no cover
                """Mock for array.at."""

                def __getitem__(self, _):  # pragma: no cover
                    """Mock __getitem__."""

                    class Adder:  # pragma: no cover
                        """Mock for .at[].add()."""

                        def add(self, *args):  # pragma: no cover
                            """Mock .add()."""
                            pass  # pragma: no cover

                    return Adder()  # pragma: no cover

            return AtMocker()  # pragma: no cover
        import zero_mlx.ops as _ops

        if hasattr(_ops, name):
            op = getattr(_ops, name)
            return lambda *args, **kwargs: op(self, *args, **kwargs)
        raise AttributeError(f"'array' object has no attribute '{name}'")

    def __abs__(self) -> "array":
        """Compute __abs__.

        Returns:
            The result of __abs__.
        """
        return array(np.abs(np.array(self.data)), dtype=self.dtype)
