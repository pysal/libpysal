"""
kernels.py

This module defines a collection of common kernel functions used for
distance-based weighting in spatial analysis, nonparametric regression,
and density estimation.

Each kernel function takes as input an array of distances and a bandwidth
parameter and returns an array of weights according to the shape of the kernel.

A general ``kernel()`` dispatcher is provided to apply a named kernel or a
user-supplied callable.

Available kernels:
    - ``triangular``
    - ``parabolic`` (Epanechnikov)
    - ``gaussian``
    - ``bisquare`` (quartic)
    - ``cosine``
    - ``exponential``
    - ``boxcar`` (uniform)
    - ``identity`` (raw distances)

Mathematical Formulation
------------------------

All kernels are evaluated as:

.. math::

    K(z), \\quad \\text{where} \\ z = \\frac{d}{h}

- :math:`d` is the distance between points.
- :math:`h` is the kernel bandwidth.
- For :math:`z > 1`, all kernels return :math:`K(z) = 0`.

"""

import numpy


def _trim(d, bandwidth):
    """
    Normalize and clip distances to the range [0, 1].

    Parameters
    ----------
    d : ndarray
        Array of distances.
    bandwidth : float
        Bandwidth parameter.

    Returns
    -------
    ndarray
        Clipped and normalized distances.
    """
    return numpy.clip(numpy.abs(d) / bandwidth, 0, 1)


def _triangular(distances, bandwidth):
    """
    Triangular kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float
        Kernel bandwidth.

    Returns
    -------
    ndarray
        Triangular kernel weights.
    """
    return 1 - _trim(distances, bandwidth)


def _parabolic(distances, bandwidth):
    """
    Parabolic (Epanechnikov) kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float
        Kernel bandwidth.

    Returns
    -------
    ndarray
        Parabolic kernel weights.
    """
    z = _trim(distances, bandwidth)
    return 0.75 * (1 - z**2)


def _gaussian(distances, bandwidth):
    """
    Gaussian kernel function (truncated at z=1).

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float
        Kernel bandwidth.

    Returns
    -------
    ndarray
        Gaussian kernel weights.
    """
    z = distances / bandwidth
    exponent_term = -0.5 * (z**2)
    c = 1 / numpy.sqrt(2 * numpy.pi)
    k = c * numpy.exp(exponent_term)
    return numpy.where(z <= 1, k, 0)


def _bisquare(distances, bandwidth):
    """
    Bisquare (quartic) kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float
        Kernel bandwidth.

    Returns
    -------
    ndarray
        Bisquare kernel weights.
    """
    z = numpy.clip(distances / bandwidth, 0, 1)
    return (15 / 16) * (1 - z**2) ** 2


def _cosine(distances, bandwidth):
    """
    Cosine kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float
        Kernel bandwidth.

    Returns
    -------
    ndarray
        Cosine kernel weights.
    """
    z = numpy.clip(distances / bandwidth, 0, 1)
    return (numpy.pi / 4) * numpy.cos(numpy.pi / 2 * z)


def _exponential(distances, bandwidth):
    """
    Exponential kernel function, truncated at z=1.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float
        Kernel bandwidth.

    Returns
    -------
    ndarray
        Exponential kernel weights.
    """
    z = distances / bandwidth
    return numpy.exp(-z)


def _boxcar(distances, bandwidth):
    """
    Boxcar (uniform) kernel function.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float
        Kernel bandwidth.

    Returns
    -------
    ndarray
        Binary weights: 1 if distance < bandwidth, else 0.
    """
    return (distances < bandwidth).astype(int)


def _identity(distances, _):
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
    "cosine": _cosine,
    "boxcar": _boxcar,
    "discrete": _boxcar,
    "exponential": _exponential,
    "identity": _identity,
    None: _identity,
}


def kernel(distances, bandwidth, kernel="gaussian", taper=None, decay=False):
    """
    Evaluate a kernel function over a distance array.

    Parameters
    ----------
    distances : ndarray
        Array of distances.
    bandwidth : float
        Kernel bandwidth.
    kernel : str or callable, optional
        The kernel function to use. If a string, must be one of the predefined
        kernel names: 'triangular', 'parabolic', 'gaussian', 'bisquare',
        'cosine', 'boxcar', 'discrete', 'exponential', 'identity'.
        If callable, it should have the signature `(distances, bandwidth)`.
        If None, the 'identity' kernel is used.
    taper : bool (default: None)
        remove edges in the graph depending on their distance. If True,
        edges are removed if points are separated more than the bandwidth.
        If False, no edges are pruned. If a float is provided, all edges with
        points separated by more than "taper" are removed.
    as_decay: bool (default: False)
        whether to calculate the kernel using the decay formulation. In the
        decay form, a kernel measures the distance decay in similarity between
        observations. It varies from from maximal similarity (1) at a distance
        of zero to minimal similarity (0 or negative) at some very large
        (possibly infinite) distance. Otherwise, kernel functions are treated as
        proper volume-preserving probability distributions.

    Returns
    -------
    ndarray
        Kernel function evaluated at distance values.

    Notes:
    ------
    Some kernels ("gaussian","exponential") are defined as un-tapered kernels
    by default. This is because they are supported over all real values. In
    contrast, many other kernels (e.g. "triangular", "parabolic"), are tapered
    when distances are greater than the bandwidth.
    """
    if taper is None:
        # only taper if we're not using built-in gaussian/exponential
        taper = kernel not in ("gaussian", "exponential")
    if callable(kernel):
        func = kernel
    elif kernel is None:
        func = _kernel_functions[None]
    else:
        func = _kernel_functions[kernel.lower()]

    k = func(distances, bandwidth)

    if decay:
        k /= func(0, bandwidth)

    if taper:
        k[distances > bandwidth] = 0

    return k
