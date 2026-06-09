from zero_mlx import array, core, stream


def test_array():
    a = array(1)
    b = a + 2
    assert b.data == 3


def test_core_eval():
    a = array(1)
    core.eval(a)


def test_stream():
    with stream():
        array(1)
