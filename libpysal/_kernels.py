import numpy


def _triangular(distances, bandwidth):
    u = distances/bandwidth
    u = 1-numpy.clip(u, 0,1)
    return u/bandwidth

def _parabolic(distances, bandwidth):
    u = numpy.clip(distances / bandwidth, 0, 1)
    return 0.75 * (1 - u**2)


def _gaussian(distances, bandwidth):
    u = distances / bandwidth
    exponent_term = -0.5 * (u**2)
    c = 1 / (bandwidth * numpy.sqrt(2 * numpy.pi))
    return c * numpy.exp(exponent_term)


def _bisquare(distances, bandwidth):
    u = numpy.clip(distances / bandwidth, 0, 1)
    return (15 / 16) * (1 - u**2) ** 2


def _cosine(distances, bandwidth):
    u = numpy.clip(distances / bandwidth, 0, 1)
    return (numpy.pi / 4) * numpy.cos(numpy.pi / 2 * u)


def _exponential(distances, bandwidth):
    u = distances / bandwidth
    return numpy.exp(-u)


def _boxcar(distances, bandwidth):
    r = (distances < bandwidth).astype(int)
    return r


def _identity(distances, _):
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


def _kernel(
    distances,
    bandwidth,
    kernel="gaussian",
):
    """
    distances: array-like
    bandwidth  float
    kernel: string or callable

    """

    if callable(kernel):
        k = kernel(distances, bandwidth)
    else:
        kernel = kernel.lower()
        k = _kernel_functions[kernel](distances, bandwidth)

    return k
