"""End to end behaviour of the MLP on small synthetic data."""

import numpy as np

from src.scratch.network import MLP


def two_blobs(n=400, d=20, seed=0):
    """Two well separated Gaussian clusters. Any working model reaches 100 percent."""
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.standard_normal((n, d)) - 2.0, rng.standard_normal((n, d)) + 2.0])
    y = np.array([0] * n + [1] * n)
    return x.astype(np.float32), y


def test_forward_shape():
    model = MLP([20, 16, 2], seed=0)
    assert model.forward(np.zeros((7, 20), dtype=np.float32)).shape == (7, 2)


def test_learns_a_separable_problem():
    x, y = two_blobs()
    model = MLP([20, 16, 2], seed=0)
    model.fit(x, y, epochs=10, batch_size=32, learning_rate=0.1, verbose=False)
    assert model.evaluate(x, y)[1] > 0.99


def test_loss_decreases():
    x, y = two_blobs(seed=1)
    model = MLP([20, 16, 2], seed=1)
    history = model.fit(x, y, epochs=8, batch_size=32, learning_rate=0.1, verbose=False)
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_save_and_load_round_trip(tmp_path):
    x, y = two_blobs(seed=2)
    model = MLP([20, 16, 2], seed=2)
    model.fit(x, y, epochs=3, batch_size=32, verbose=False)
    before = model.predict(x)

    path = tmp_path / "model.npz"
    model.save(path)
    after = MLP.load(path).predict(x)
    assert np.array_equal(before, after)


def test_dropout_is_off_at_evaluation_time():
    x, _ = two_blobs(seed=3)
    model = MLP([20, 16, 2], dropout=0.5, seed=3)
    assert np.allclose(model.predict_logits(x[:16]), model.predict_logits(x[:16]))


def test_parameter_count():
    model = MLP([784, 256, 128, 10], seed=0)
    expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
    total = sum(p.size for layer in model.layers for p in layer.parameters().values())
    assert total == expected
