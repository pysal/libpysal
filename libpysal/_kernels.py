"""
Kernel weight functions for spatial or statistical analysis.

This module provides a collection of kernel functions used to compute
weights based on distances and a given bandwidth. These functions are 
commonly used in kernel density estimation, geographically weighted 
regression, and other localized modeling techniques.

Available kernel types:
- 'triangular'
- 'parabolic'
- 'gaussian'
- 'bisquare'
- 'cosine'
- 'boxcar' / 'discrete'
- 'exponential'
- 'identity' (returns raw distances)


Notes
-----
Unless otherwise stated, kernel functions as defined in Anselin, L. (2024) An Introduction to Spatial 
Data Science with GeoDa: Volume 1 Exploring Spatial Data. CRC Press. p. 230.
"""

import numpy as np

def _triangular(distances, bandwidth):
    """
    Triangular kernel.

    Parameters
    ----------
    distances : array-like
        Input distances.
    bandwidth : float
        Bandwidth parameter.

    Returns
    -------
    weights : ndarray
        Weights computed using the triangular kernel.
    """
    u = np.clip(distances / bandwidth, 0, 1)
    return 1 - u


def _parabolic(distances, bandwidth):
    """
    Parabolic (Epanechnikov) kernel.

    Parameters
    ----------
    distances : array-like
        Input distances.
    bandwidth : float
        Bandwidth parameter.

    Returns
    -------
    weights : ndarray
        Weights computed using the parabolic kernel.
    """
    u = np.clip(distances / bandwidth, 0, 1)
    return 0.75 * (1 - u**2)


def _gaussian(distances, bandwidth):
    """
    Gaussian kernel.

    Parameters
    ----------
    distances : array-like
        Input distances.
    bandwidth : float
        Bandwidth parameter.

    Returns
    -------
    weights : ndarray
        Weights computed using the Gaussian kernel.
    """
    u = distances / bandwidth
    exponent_term = -0.5 * (u ** 2)
    c = 1 / np.sqrt(2 * np.pi)
    return c * np.exp(exponent_term)


def _bisquare(distances, bandwidth):
    """
    Bisquare (or biweight) kernel.

    Parameters
    ----------
    distances : array-like
        Input distances.
    bandwidth : float
        Bandwidth parameter.

    Returns
    -------
    weights : ndarray
        Weights computed using the bisquare kernel.
    """
    u = np.clip(distances / bandwidth, 0, 1)
    return (15 / 16) * (1 - u**2) ** 2


def _cosine(distances, bandwidth):
    """
    Cosine kernel.

    Parameters
    ----------
    distances : array-like
        Input distances.
    bandwidth : float
        Bandwidth parameter.

    Returns
    -------
    weights : ndarray
        Weights computed using the cosine kernel.

    Notes
    ----- 
    Source: Silverman, B.W. (1986). Density Estimation for Statistics and 
    Data Analysis.

    """
    u = np.clip(distances / bandwidth, 0, 1)
    return (np.pi / 4) * np.cos(np.pi / 2 * u)


def _exponential(distances, bandwidth):
    """
    Exponential kernel.

    Parameters
    ----------
    distances : array-like
        Input distances.
    bandwidth : float
        Bandwidth parameter.

    Returns
    -------
    weights : ndarray
        Weights computed using the exponential kernel.

    Notes
    -----
    TODO: source
    """
    u = distances / bandwidth
    return np.exp(-u)


def _boxcar(distances, bandwidth):
    """
    Boxcar (uniform) kernel.

    Parameters
    ----------
    distances : array-like
        Input distances.
    bandwidth : float
        Bandwidth parameter.

    Returns
    -------
    weights : ndarray
        Weights of 1 for distances < bandwidth, 0 otherwise.

    Notes
    -----
    TODO: source
    """
    r = (distances < bandwidth).astype(int)
    return r


def _identity(distances, _):
    """
    Identity function (returns the input distances).

    Parameters
    ----------
    distances : array-like
        Input distances.
    _ : any
        Ignored.

    Returns
    -------
    distances : ndarray
        Unchanged input distances.
    """
    return distances


# dispatcher

_kernel_functions = {
    "triangular": _triangular,
    "parabolic": _parabolic,
    "gaussian": _gaussian,
    "bisquare": _bisquare,
    "cosine": _cosine,
    "boxcar": _boxcar,
    "discrete": _boxcar,
    "exponential": _exponential,
    "identity": _identity,
    None: _identity,
}


def _kernel(distances, bandwidth, kernel='gaussian'):
    """
    Compute kernel weights given distances and bandwidth.

    Parameters
    ----------
    distances : array-like
        Input distances.
    bandwidth : float
        Bandwidth parameter controlling the kernel shape.
    kernel : str or callable, optional
        The kernel to use. Can be one of the predefined kernel names
        or a custom function of the form `f(distances, bandwidth)`.

    Returns
    -------
    weights : ndarray
        Computed kernel weights.
    
    Raises
    ------
    ValueError
        If a string kernel name is provided and is not recognized.
    """
    if callable(kernel):
        k = kernel(distances, bandwidth)
    else:
        kernel = kernel.lower()
        if kernel not in _kernel_functions:
            raise ValueError(f"Unknown kernel: {kernel!r}. Choose from {list(_kernel_functions)}.")
        k = _kernel_functions[kernel](distances, bandwidth)

    return k
