import numpy as np
import pytest

import libpysal.kernels as kernels


@pytest.fixture
def distances():
    return np.array([0.0, 0.5, 1.0, 1.5])


@pytest.fixture
def bandwidth():
    return 1.0


# --- Individual kernel tests ---

def test_trim_clipping():
    d = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    bw = 1.0
    result = kernels._trim(d, bw)
    expected = np.clip(np.abs(d) / bw, 0, 1)
    np.testing.assert_array_almost_equal(result, expected)


def test_triangular(distances, bandwidth):
    expected = 1 - np.clip(np.abs(distances) / bandwidth, 0, 1)
    np.testing.assert_array_almost_equal(
        kernels._triangular(distances, bandwidth), expected
    )


def test_parabolic(distances, bandwidth):
    z = np.clip(np.abs(distances) / bandwidth, 0, 1)
    expected = 0.75 * (1 - z**2)
    np.testing.assert_array_almost_equal(
        kernels._parabolic(distances, bandwidth), expected
    )


def test_gaussian(distances, bandwidth):
    z = distances / bandwidth
    exponent = -0.5 * z**2
    c = 1 / np.sqrt(2 * np.pi)
    expected = c * np.exp(exponent)
    np.testing.assert_array_almost_equal(
        kernels._gaussian(distances, bandwidth), expected
    )


def test_bisquare(distances, bandwidth):
    z = np.clip(distances / bandwidth, 0, 1)
    expected = (15 / 16) * (1 - z**2) ** 2
    np.testing.assert_array_almost_equal(
        kernels._bisquare(distances, bandwidth), expected
    )


def test_cosine(distances, bandwidth):
    z = np.clip(distances / bandwidth, 0, 1)
    expected = (np.pi / 4) * np.cos(np.pi / 2 * z)
    np.testing.assert_array_almost_equal(
        kernels._cosine(distances, bandwidth), expected
    )


def test_exponential(distances, bandwidth):
    z = distances / bandwidth
    #expected = np.where(z <= 1, np.exp(-z), 0)
    expected = np.exp(-z)
    np.testing.assert_array_almost_equal(
        kernels._exponential(distances, bandwidth), expected
    )


def test_boxcar(distances, bandwidth):
    expected = (distances < bandwidth).astype(int)
    np.testing.assert_array_equal(
        kernels._boxcar(distances, bandwidth), expected
    )


def test_identity(distances, bandwidth):
    np.testing.assert_array_equal(
        kernels._identity(distances, bandwidth), distances
    )


# --- Dispatcher tests ---

@pytest.mark.parametrize("name", [
    "triangular", "parabolic", "gaussian", "bisquare",
    "cosine", "boxcar", "discrete", "exponential", "identity", None
])
def test_kernel_dispatcher_names(distances, bandwidth, name):
    # Should not raise or error
    result = kernels.kernel(distances, bandwidth, kernel=name)
    assert isinstance(result, np.ndarray)
    assert result.shape == distances.shape


def test_kernel_dispatcher_callable(distances, bandwidth):
    def linear_kernel(d, bw):
        return 1 - np.clip(d / bw, 0, 1)

    result = kernels.kernel(distances, bandwidth, kernel=linear_kernel)
    expected = linear_kernel(distances, bandwidth)
    np.testing.assert_array_almost_equal(result, expected)


def test_kernel_dispatcher_invalid_name(distances, bandwidth):
    with pytest.raises(KeyError):
        kernels.kernel(distances, bandwidth, kernel="not-a-kernel")

