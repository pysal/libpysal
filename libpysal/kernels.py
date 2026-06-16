"""kernels.py

This module defines a collection of common kernel functions used for
distance-based weighting in spatial analysis, nonparametric regression,
and density estimation.

Each kernel function takes as input an array of distances and a
bandwidth parameter and returns an array of values for the kernel
evaluated over the distances with the specified bandwidth.

A general ``kernel()`` dispatcher is provided to apply a named kernel or a
user-supplied callable.

Available kernels:
    - ``triangular``
    - ``parabolic`` (Epanechnikov)
    - ``gaussian``
    - ``bisquare`` (quartic)
    - ``tricube``
    - ``cosine``
    - ``exponential``
    - ``boxcar`` (uniform)
    - ``identity`` (raw distances)

Mathematical Formulation
------------------------

All kernels are evaluated as:

.. math::

    K(z), \\quad \\text{where} \\ z = \\frac{d}{h}

- :math:`d` is the distance.
- :math:`h` is the kernel bandwidth.
- For :math:`z > 1`, all kernels return :math:`K(z) = 0`.

"""

import numpy as np


def _trim(distances: np.ndarray, bandwidth) -> np.ndarray:
    """
    Normalize and clip distances to the range [0, 1].

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float or ndarray
        Bandwidth parameter. Can be a single float or an array of the same shape as distances.

    Returns
    -------
    ndarray
        Clipped and normalized distances.
    """
    if not isinstance(bandwidth, (int, float)):
        bandwidth = np.asarray(bandwidth)

    return np.clip(np.abs(distances) / bandwidth, 0, 1)


def _triangular(distances: np.ndarray, bandwidth) -> np.ndarray:
    """
    Triangular kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float or ndarray
        Bandwidth parameter. Can be a single float or an array of the same shape as distances.

    Returns
    -------
    ndarray
        Triangular kernel weights.
    """
    return 1 - _trim(distances, bandwidth)


def _parabolic(distances: np.ndarray, bandwidth) -> np.ndarray:
    """
    Parabolic (Epanechnikov) kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float or ndarray
        Bandwidth parameter. Can be a single float or an array of the same shape as distances.

    Returns
    -------
    ndarray
        Parabolic kernel weights.
    """
    z = _trim(distances, bandwidth)
    return 0.75 * (1 - z**2)


def _gaussian(distances: np.ndarray, bandwidth) -> np.ndarray:
    """
    Gaussian kernel function (truncated at z=1).

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float or ndarray
        Bandwidth parameter. Can be a single float or an array of the same shape as distances.

    Returns
    -------
    ndarray
        Gaussian kernel weights.
    """
    if not isinstance(bandwidth, (int, float)):
       bandwidth = np.asarray(bandwidth)
        
    z = distances / bandwidth
    exponent_term = -0.5 * (z**2)
    c = 1 / np.sqrt(2 * np.pi)
    k = c * np.exp(exponent_term)
    return k


def _bisquare(distances: np.ndarray, bandwidth) -> np.ndarray:
    """
    Bisquare (quartic) kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float or ndarray
        Bandwidth parameter. Can be a single float or an array of the same shape as distances.

    Returns
    -------
    ndarray
        Bisquare kernel weights.
    """
    z = _trim(distances, bandwidth)
    return (15 / 16) * (1 - z**2) ** 2


def _tricube(distances: np.ndarray, bandwidth) -> np.ndarray:
    """
    Tricube kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float or ndarray
        Bandwidth parameter. Can be a single float or an array of the same shape as distances.

    Returns
    -------
    ndarray
        Tricube kernel weights.
    """
    z = _trim(distances, bandwidth)
    return (70 / 81) * (1 - z**3) ** 3


def _cosine(distances: np.ndarray, bandwidth) -> np.ndarray:
    """
    Cosine kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float or ndarray
        Bandwidth parameter. Can be a single float or an array of the same shape as distances.

    Returns
    -------
    ndarray
        Cosine kernel weights.
    """
    z = _trim(distances, bandwidth)
    return (np.pi / 4) * np.cos(np.pi / 2 * z)


def _exponential(distances: np.ndarray, bandwidth) -> np.ndarray:
    """
    Exponential kernel function, truncated at z=1.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float or ndarray
        Bandwidth parameter. Can be a single float or an array of the same shape as distances.

    Returns
    -------
    ndarray
        Exponential kernel weights.
    """
    if not isinstance(bandwidth, (int, float)):
        bandwidth = np.asarray(bandwidth)
        
    z = distances / bandwidth
    return np.exp(-z)


def _boxcar(distances: np.ndarray, bandwidth) -> np.ndarray:
    """
    Boxcar (uniform) kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float or ndarray
        Bandwidth parameter. Can be a single float or an array of the same shape as distances.

    Returns
    -------
    ndarray
        Binary weights: 1 if distance < bandwidth, else 0.
    """
    distances = np.asarray(distances)
    if isinstance(bandwidth, (int, float)):
        return (distances < bandwidth).astype(float)
    else:
        bandwidth = np.asarray(bandwidth)
        return (distances < bandwidth).astype(float)


def _identity(distances: np.ndarray, _) -> np.ndarray:
    """
    Identity kernel (no weighting, returns raw distances).

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    _ : float
        Unused bandwidth parameter.

    Returns
    -------
    ndarray
        The raw input distances.
    """
    return distances


_kernel_functions = {
    "triangular": _triangular,
    "parabolic": _parabolic,
    "gaussian": _gaussian,
    "bisquare": _bisquare,
    "tricube": _tricube,
    "cosine": _cosine,
    "boxcar": _boxcar,
    "discrete": _boxcar,
    "exponential": _exponential,
    "identity": _identity,
    None: _identity,
}


def kernel(
    distances: np.ndarray, bandwidth, kernel="gaussian", taper=True, decay=False
) -> np.ndarray:
    """
    Evaluate a kernel function over a distance array.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float or ndarray
        Kernel bandwidth. Can be a single float or an array of the same shape as distances.
    kernel : str or callable, optional
        The kernel function to use. If a string, must be one of the predefined
        kernel names: 'triangular', 'parabolic', 'gaussian', 'bisquare',
        'tricube', 'cosine', 'boxcar', 'discrete', 'exponential', 'identity'.
        If callable, it should have the signature `(distances, bandwidth)`.
        If None, the 'identity' kernel is used.
    taper : bool (default: True)
        Set kernel = 0 for all distances exceeding the bandwith. To
        evaluate kernel beyond bandwith set taper=False.
    decay : bool (default: False)
        Whether to calculate the kernel using the decay formulation.
        In the decay form, a kernel measures the distance decay in
        similarity between observations. It varies from maximal
        similarity (1) at a distance of zero to minimal similarity (0
        or negative) at some very large (possibly infinite) distance.
        Otherwise, kernel functions are treated as proper
        volume-preserving probability distributions.

    Returns
    -------
    ndarray
        Kernel function evaluated at distance values.
    """
    if isinstance(kernel, str) and kernel not in _kernel_functions:
        raise ValueError(
            f"Invalid kernel '{kernel}'. "
            f"Supported kernels are: {list(_kernel_functions.keys())}, "
            "None, or a callable."
        )

    func = _kernel_functions.get(kernel, kernel)

    if not callable(func):
        raise ValueError("kernel must be either a valid string, None, or a callable.")

    k = func(distances, bandwidth)

    if taper is True:
        if isinstance(bandwidth, (int, float)):
            k[distances > bandwidth] = 0.0
        else:
            bandwidth = np.asarray(bandwidth)
            k[distances > bandwidth] = 0.0

    elif isinstance(taper, (float, int)) and not isinstance(taper, bool):
        k[distances > taper] = 0.0

    if decay:
        if isinstance(bandwidth, (int, float)):
            k /= func(0.0, bandwidth)
        else:
            k /= func(0.0, np.mean(bandwidth))

    return k
