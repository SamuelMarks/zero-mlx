import pytest

from zero_mlx import optimizers


def test_optimizers():
    # Optimizer base
    opt = optimizers.Optimizer(schedulers="sched")
    assert opt.schedulers == "sched"

    # AdaDelta
    adadelta = optimizers.AdaDelta(0.1, rho=0.8, eps=1e-6)
    assert adadelta.learning_rate == 0.1
    assert adadelta.rho == 0.8
    assert adadelta.eps == 1e-6

    # Adafactor
    adafactor = optimizers.Adafactor(learning_rate=0.01, eps=(1e-30, 1e-4))
    assert adafactor.learning_rate == 0.01
    assert adafactor.eps == (1e-30, 1e-4)

    # Adagrad
    adagrad = optimizers.Adagrad(0.1, eps=1e-6)
    assert adagrad.learning_rate == 0.1
    assert adagrad.eps == 1e-6

    # Adam
    adam = optimizers.Adam(0.1, betas=(0.9, 0.99), eps=1e-6, bias_correction=True)
    assert adam.learning_rate == 0.1
    assert adam.betas == (0.9, 0.99)
    assert adam.eps == 1e-6
    assert adam.bias_correction is True

    # AdamW
    adamw = optimizers.AdamW(
        0.1, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.02, bias_correction=True
    )
    assert adamw.learning_rate == 0.1
    assert adamw.betas == (0.9, 0.99)
    assert adamw.eps == 1e-6
    assert adamw.weight_decay == 0.02
    assert adamw.bias_correction is True

    # Adamax
    adamax = optimizers.Adamax(0.1, betas=(0.9, 0.99), eps=1e-6)
    assert adamax.learning_rate == 0.1
    assert adamax.betas == (0.9, 0.99)
    assert adamax.eps == 1e-6

    # Lion
    lion = optimizers.Lion(0.1, betas=(0.9, 0.99), weight_decay=0.02)
    assert lion.learning_rate == 0.1
    assert lion.betas == (0.9, 0.99)
    assert lion.weight_decay == 0.02

    # Module
    mod = optimizers.Module()
    with pytest.raises(NotImplementedError):
        mod()

    # MultiOptimizer
    multi = optimizers.MultiOptimizer([adam, sgd := optimizers.SGD(0.1)])
    assert multi.optimizers == [adam, sgd]
    assert multi.filters == []

    # Muon
    muon = optimizers.Muon(
        0.1, momentum=0.9, weight_decay=0.02, nesterov=False, ns_steps=10
    )
    assert muon.learning_rate == 0.1
    assert muon.momentum == 0.9
    assert muon.weight_decay == 0.02
    assert muon.nesterov is False
    assert muon.ns_steps == 10

    # RMSprop
    rmsprop = optimizers.RMSprop(0.1, alpha=0.9, eps=1e-6)
    assert rmsprop.learning_rate == 0.1
    assert rmsprop.alpha == 0.9
    assert rmsprop.eps == 1e-6

    # SGD
    sgd = optimizers.SGD(
        0.1, momentum=0.9, weight_decay=0.02, dampening=0.1, nesterov=True
    )
    assert sgd.learning_rate == 0.1
    assert sgd.momentum == 0.9
    assert sgd.weight_decay == 0.02
    assert sgd.dampening == 0.1
    assert sgd.nesterov is True
