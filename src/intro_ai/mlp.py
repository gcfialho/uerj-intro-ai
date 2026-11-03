"""A multilayer perceptron written with NumPy only: forward pass, backpropagation, RMSprop.

The course notebook trained a Keras MLP on MNIST. This module re-implements the same
kind of model from scratch and trains it on scikit-learn's bundled 8x8 digits dataset,
which ships with the library and needs no download.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

ACTIVATIONS = ("sigmoid", "relu")


def softmax(Z: Array) -> Array:
    """Row-wise softmax, shifted by the row maximum for numerical stability."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def one_hot(y: NDArray[np.integer], n_classes: int) -> Array:
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def cross_entropy(P: Array, Y: Array) -> float:
    """Mean categorical cross-entropy; probabilities are clipped away from 0."""
    return float(-np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1.0)), axis=1)))


def _act(Z: Array, kind: str) -> Array:
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-Z))
    return np.maximum(Z, 0.0)


def _act_grad(Z: Array, A: Array, kind: str) -> Array:
    if kind == "sigmoid":
        return A * (1.0 - A)
    return (Z > 0).astype(Z.dtype)


@dataclass
class MLP:
    """Fully connected network: hidden layers use ``activation``, the output layer softmax."""

    sizes: tuple[int, ...]
    activation: str = "sigmoid"
    seed: int = 0
    weights: list[Array] = field(init=False)
    biases: list[Array] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.sizes) < 2:
            raise ValueError("need at least an input and an output size")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        rng = np.random.default_rng(self.seed)
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:], strict=True):
            limit = np.sqrt(6.0 / (fan_in + fan_out))  # Glorot uniform, as in Keras
            self.weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))

    @property
    def n_params(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases, strict=True))

    def params(self) -> list[Array]:
        return [*self.weights, *self.biases]

    def _forward(self, X: Array) -> tuple[list[Array], list[Array]]:
        """Return pre-activations Z and activations A (A[0] = X, A[-1] = softmax output)."""
        A, Z = [X], []
        last = len(self.weights) - 1
        for i, (W, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = A[-1] @ W + b
            Z.append(z)
            A.append(softmax(z) if i == last else _act(z, self.activation))
        return Z, A

    def predict_proba(self, X: Array) -> Array:
        return self._forward(X)[1][-1]

    def predict(self, X: Array) -> NDArray[np.intp]:
        return np.argmax(self.predict_proba(X), axis=1)

    def loss_and_grads(self, X: Array, Y: Array) -> tuple[float, list[Array]]:
        """Cross-entropy loss and its gradient w.r.t. ``params()`` (weights, then biases)."""
        Z, A = self._forward(X)
        n = X.shape[0]
        loss = cross_entropy(A[-1], Y)
        dW: list[Array] = [np.empty(0)] * len(self.weights)
        db: list[Array] = [np.empty(0)] * len(self.biases)
        delta = (A[-1] - Y) / n  # d loss / d Z_last for softmax + cross-entropy
        for i in range(len(self.weights) - 1, -1, -1):
            dW[i] = A[i].T @ delta
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * _act_grad(Z[i - 1], A[i], self.activation)
        return loss, [*dW, *db]

    def evaluate(self, X: Array, y: NDArray[np.integer]) -> tuple[float, float]:
        """Return (loss, accuracy)."""
        P = self.predict_proba(X)
        loss = cross_entropy(P, one_hot(y, P.shape[1]))
        return loss, float(np.mean(np.argmax(P, axis=1) == y))


@dataclass
class RMSprop:
    """RMSprop with Keras' default hyper-parameters."""

    lr: float = 1e-3
    rho: float = 0.9
    eps: float = 1e-7
    _cache: list[Array] | None = None

    def step(self, params: list[Array], grads: list[Array]) -> None:
        if self._cache is None:
            self._cache = [np.zeros_like(p) for p in params]
        for p, g, c in zip(params, grads, self._cache, strict=True):
            c *= self.rho
            c += (1.0 - self.rho) * g * g
            p -= self.lr * g / (np.sqrt(c) + self.eps)


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_acc: list[float] = field(default_factory=list)


def train(
    model: MLP,
    X_train: Array,
    y_train: NDArray[np.integer],
    X_test: Array,
    y_test: NDArray[np.integer],
    *,
    epochs: int,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> History:
    """Mini-batch RMSprop. Train and test loss/accuracy are recorded after every epoch."""
    rng = np.random.default_rng(seed)
    opt = RMSprop(lr=lr)
    n_classes = model.sizes[-1]
    Y_train = one_hot(y_train, n_classes)
    hist = History()
    for _ in range(epochs):
        order = rng.permutation(X_train.shape[0])
        for start in range(0, order.size, batch_size):
            idx = order[start : start + batch_size]
            _, grads = model.loss_and_grads(X_train[idx], Y_train[idx])
            opt.step(model.params(), grads)
        for (loss, acc), (ls, accs) in (
            (model.evaluate(X_train, y_train), (hist.train_loss, hist.train_acc)),
            (model.evaluate(X_test, y_test), (hist.test_loss, hist.test_acc)),
        ):
            ls.append(loss)
            accs.append(acc)
    return hist


def load_digits_split(
    test_size: float = 0.25, seed: int = 0
) -> tuple[Array, Array, NDArray[np.intp], NDArray[np.intp]]:
    """scikit-learn's 8x8 digits (1,797 images), pixels scaled to [0, 1], stratified split."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_tr, X_te, y_tr, y_te


@dataclass(frozen=True)
class Experiment:
    name: str
    hidden: tuple[int, ...]
    activation: str
    epochs: int
    description: str


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("baseline", (512,), "sigmoid", 20, "notebook model: 512 sigmoid units, 20 epochs"),
    Experiment("few-epochs", (512,), "sigmoid", 2, "same model, 2 epochs"),
    Experiment("many-epochs", (512,), "sigmoid", 500, "same model, 500 epochs"),
    Experiment("deeper-sigmoid", (512, 300, 300, 300), "sigmoid", 20, "+3 x 300 sigmoid layers"),
    Experiment("deeper-relu", (512, 300, 300, 300), "relu", 20, "+3 x 300 ReLU layers"),
)


def run_experiments(
    experiments: tuple[Experiment, ...] = EXPERIMENTS,
    *,
    seed: int = 0,
    max_epochs: int | None = None,
) -> list[dict]:
    """Train a fresh model per experiment and return its per-epoch history.

    ``max_epochs`` caps every run (used for quick smoke tests).
    """
    X_tr, X_te, y_tr, y_te = load_digits_split(seed=seed)
    out = []
    for exp in experiments:
        epochs = exp.epochs if max_epochs is None else min(exp.epochs, max_epochs)
        model = MLP((X_tr.shape[1], *exp.hidden, 10), activation=exp.activation, seed=seed)
        hist = train(model, X_tr, y_tr, X_te, y_te, epochs=epochs, seed=seed)
        best = int(np.argmax(hist.test_acc))
        lowest = int(np.argmin(hist.test_loss))
        out.append(
            {
                "name": exp.name,
                "description": exp.description,
                "hidden": list(exp.hidden),
                "activation": exp.activation,
                "epochs": epochs,
                "params": model.n_params,
                "train_acc": hist.train_acc[-1],
                "test_acc": hist.test_acc[-1],
                "train_loss": hist.train_loss[-1],
                "test_loss": hist.test_loss[-1],
                "best_test_acc": hist.test_acc[best],
                "best_test_epoch": best + 1,
                "min_test_loss": hist.test_loss[lowest],
                "min_test_loss_epoch": lowest + 1,
                "history": {
                    "train_loss": hist.train_loss,
                    "train_acc": hist.train_acc,
                    "test_loss": hist.test_loss,
                    "test_acc": hist.test_acc,
                },
            }
        )
    return out
