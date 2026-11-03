import numpy as np
import pytest

from intro_ai.mlp import (
    EXPERIMENTS,
    MLP,
    RMSprop,
    cross_entropy,
    load_digits_split,
    one_hot,
    run_experiments,
    softmax,
    train,
)


def test_softmax_rows_sum_to_one_and_are_stable():
    Z = np.array([[1000.0, 1000.0], [-1000.0, 0.0], [1.0, 2.0]])
    P = softmax(Z)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(np.isfinite(P))
    assert np.allclose(P[0], [0.5, 0.5])


def test_cross_entropy_of_perfect_and_uniform_predictions():
    Y = one_hot(np.array([0, 2]), 3)
    assert cross_entropy(Y, Y) == pytest.approx(0.0, abs=1e-9)
    assert cross_entropy(np.full((2, 3), 1 / 3), Y) == pytest.approx(np.log(3))


@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
def test_backprop_matches_finite_differences(activation):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(7, 5))
    Y = one_hot(rng.integers(0, 3, size=7), 3)
    model = MLP((5, 4, 6, 3), activation=activation, seed=2)
    for b in model.biases:  # non-zero biases so their gradients are exercised
        b += rng.normal(scale=0.1, size=b.shape)
    _, grads = model.loss_and_grads(X, Y)
    eps = 1e-6
    for p, g in zip(model.params(), grads, strict=True):
        num = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps
            up = model.loss_and_grads(X, Y)[0]
            p[idx] = old - eps
            down = model.loss_and_grads(X, Y)[0]
            p[idx] = old
            num[idx] = (up - down) / (2 * eps)
        assert np.allclose(g, num, rtol=1e-4, atol=1e-7)


def test_rmsprop_updates_parameters_in_place():
    model = MLP((3, 2), seed=0)
    before = [p.copy() for p in model.params()]
    _, grads = model.loss_and_grads(np.ones((2, 3)), one_hot(np.array([0, 1]), 2))
    RMSprop().step(model.params(), grads)
    assert any(not np.array_equal(a, b) for a, b in zip(before, model.params(), strict=True))


def test_digits_split_is_stratified_and_scaled():
    X_tr, X_te, y_tr, y_te = load_digits_split()
    assert X_tr.shape == (1347, 64)
    assert X_te.shape == (450, 64)
    assert X_tr.min() >= 0.0
    assert X_tr.max() <= 1.0
    share = np.bincount(y_te) / (np.bincount(y_tr) + np.bincount(y_te))
    assert np.allclose(share, 0.25, atol=0.01)


def test_training_reduces_loss_and_generalises():
    X_tr, X_te, y_tr, y_te = load_digits_split()
    model = MLP((64, 64, 10), activation="relu", seed=0)
    hist = train(model, X_tr, y_tr, X_te, y_te, epochs=15, lr=3e-3, seed=0)
    assert hist.train_loss[-1] < 0.5 * hist.train_loss[0]
    assert hist.test_acc[-1] > 0.9
    assert len(hist.test_acc) == 15


def test_training_is_deterministic_for_a_seed():
    X_tr, X_te, y_tr, y_te = load_digits_split()
    runs = []
    for _ in range(2):
        model = MLP((64, 16, 10), seed=3)
        runs.append(train(model, X_tr, y_tr, X_te, y_te, epochs=2, seed=3).test_loss)
    assert runs[0] == runs[1]


def test_run_experiments_trains_a_fresh_model_each_time():
    runs = run_experiments(EXPERIMENTS[:2], max_epochs=1)
    assert [r["name"] for r in runs] == ["baseline", "few-epochs"]
    # Same architecture, same seed, one epoch each: identical results prove that the
    # second experiment did not continue training the first model.
    assert runs[0]["test_loss"] == runs[1]["test_loss"]
    assert runs[0]["params"] == 64 * 512 + 512 + 512 * 10 + 10


def test_invalid_models():
    with pytest.raises(ValueError, match="activation"):
        MLP((4, 2), activation="tanh")
    with pytest.raises(ValueError, match="at least"):
        MLP((4,))
