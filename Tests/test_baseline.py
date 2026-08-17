import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from FileHandler.windowing import sliding_windows
from Models.base import AnomalyDetector
from Models.pca_baseline import PCAReconstructionDetector


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        AnomalyDetector()


def test_score_before_fit_raises():
    windows = np.random.randn(10, 5, 1)
    with pytest.raises(RuntimeError):
        PCAReconstructionDetector(n_components=1).score(windows)


def test_invalid_n_components_raises():
    with pytest.raises(ValueError):
        PCAReconstructionDetector(n_components=0)


def test_n_components_exceeding_data_raises():
    windows = np.random.randn(3, 2, 1)  # only 3 windows, 2 dims each
    with pytest.raises(ValueError):
        PCAReconstructionDetector(n_components=5).fit(windows)


def test_matches_sklearn_pca_reference():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 6))
    windows = X.reshape(50, 6, 1)

    ours = PCAReconstructionDetector(n_components=2).fit(windows)
    our_scores = ours.score(windows)

    ref = SklearnPCA(n_components=2).fit(X)
    X_reconstructed = ref.inverse_transform(ref.transform(X))
    ref_scores = ((X - X_reconstructed) ** 2).mean(axis=1)

    np.testing.assert_allclose(our_scores, ref_scores, atol=1e-10)
    np.testing.assert_allclose(np.abs(ours.components_), np.abs(ref.components_), atol=1e-10)


def test_perfect_reconstruction_when_n_components_equals_rank():
    # if we keep as many components as there are dimensions, reconstruction
    # should be essentially exact -- nothing was thrown away
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 4))
    windows = X.reshape(20, 4, 1)

    model = PCAReconstructionDetector(n_components=4).fit(windows)
    scores = model.score(windows)
    np.testing.assert_allclose(scores, 0.0, atol=1e-10)


def test_flags_a_real_anomaly_over_normal_windows():
    rng = np.random.default_rng(3)
    n = 200
    signal = np.sin(np.arange(n) / 10) * 5 + 70 + rng.normal(0, 0.3, n)
    signal[100:110] = 95.0  # injected anomaly

    windows, ends = sliding_windows(signal, window_size=10, stride=1)
    model = PCAReconstructionDetector(n_components=2).fit(windows)
    scores = model.score(windows)

    anomaly_region = (ends >= 100) & (ends < 110)
    normal_region = (ends >= 20) & (ends < 40)

    assert scores[anomaly_region].mean() > scores[normal_region].mean() * 5


def test_explained_variance_ratio_is_between_zero_and_one():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 5))
    windows = X.reshape(30, 5, 1)
    model = PCAReconstructionDetector(n_components=2).fit(windows)
    ratio = model.explained_variance_ratio(windows)
    assert 0.0 <= ratio <= 1.0


def test_more_components_never_increases_error():
    # more kept components = strictly more information retained, so
    # reconstruction error on the SAME data used for fitting should only
    # go down (or stay the same), never up
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 8))
    windows = X.reshape(40, 8, 1)

    err_2 = PCAReconstructionDetector(n_components=2).fit(windows).score(windows).mean()
    err_5 = PCAReconstructionDetector(n_components=5).fit(windows).score(windows).mean()
    assert err_5 <= err_2 + 1e-10


def test_fit_returns_self_for_chaining():
    windows = np.random.randn(10, 4, 1)
    model = PCAReconstructionDetector(n_components=2)
    assert model.fit(windows) is model