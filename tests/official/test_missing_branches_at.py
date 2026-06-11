from zero_mlx import array


def test_at_op_real_minimum():
    a = array([1, 2, 3])
    a.at[array([1])].minimum(2)
    a.at[1].minimum(2)
