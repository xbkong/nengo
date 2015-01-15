"""
Extra functions to extend the capabilities of Numpy.
"""
from __future__ import absolute_import

import collections

import numpy as np

from nengo.utils.magic import memoize

maxint = np.iinfo(np.int32).max


def broadcast_shape(shape, length):
    """Pad a shape with ones following standard Numpy broadcasting."""
    n = len(shape)
    if n < length:
        return tuple([1] * (length - n) + list(shape))
    else:
        return shape


def array(x, dims=None, min_dims=0, **kwargs):
    y = np.array(x, **kwargs)
    dims = max(min_dims, y.ndim) if dims is None else dims

    if y.ndim < dims:
        shape = np.ones(dims, dtype='int')
        shape[:y.ndim] = y.shape
        y.shape = shape
    elif y.ndim > dims:
        raise ValueError(
            "Input cannot be cast to array with %d dimensions" % dims)

    return y


def filt(x, tau, axis=0, x0=None, copy=True):
    """First-order causal lowpass filter.

    This performs standard first-order lowpass filtering with transfer function
                         1
        T(s) = ----------------------
               tau_in_seconds * s + 1
    discretized using the zero-order hold method.

    Parameters
    ----------
    x : array_like
        The signal to filter.
    tau : float
        The dimensionless filter time constant (tau = tau_in_seconds / dt).
    axis : integer
        The axis along which to filter.
    copy : boolean
        Whether to copy the input data, or simply work in-place.
    """
    x = np.array(x, copy=copy)
    y = np.rollaxis(x, axis=axis)  # y is rolled view on x

    # --- buffer method
    if x0 is not None:
        if x0.shape != y[0].shape:
            raise ValueError("'x0' %s must have same shape as y[0] %s" %
                             x0.shape, y[0].shape)
        yy = np.array(x0)
    else:
        # yy is our buffer for the current filter state
        yy = np.zeros_like(y[0])

    d = -np.expm1(-1. / tau)  # zero-order hold filtering
    for i, yi in enumerate(y):
        yy += d * (yi - yy)
        y[i] = yy

    return x


def filtfilt(x, tau, axis=0, copy=True):
    """Zero-phase second-order non-causal lowpass filter, implemented by
    filtering the input in forward and reverse directions.

    This function is equivalent to scipy's or Matlab's filtfilt function
    with the first-order lowpass filter
                         1
        T(s) = ----------------------
               tau_in_seconds * s + 1
    as the filter. The resulting equivalent filter has zero phase distortion
    and a transfer function magnitude equal to the square of T(s),
    discretized using the zero-order hold method.

    Parameters
    ----------
    x : array_like
        The signal to filter.
    tau : float
        The dimensionless filter time constant (tau = tau_in_seconds / dt).
    axis : integer
        The axis along which to filter.
    copy : boolean
        Whether to copy the input data, or simply work in-place.
    """
    x = np.array(x, copy=copy)
    y = np.rollaxis(x, axis=axis)  # y is rolled view on x

    # --- buffer method
    d = -np.expm1(-1. / tau)

    # filter forwards
    yy = np.zeros_like(y[0])  # yy is our buffer for the current filter state
    for i, yi in enumerate(y):
        yy += d * (yi - yy)
        y[i] = yy

    # filter backwards
    z = y[::-1]  # z is a flipped view on y
    for i, zi in enumerate(z):
        yy += d * (zi - yy)
        z[i] = yy

    return x


def lowpass_transfer_fn(tau, dt):
    """Returns the transfer function description of a low pass synapse."""
    if tau > 0.03 * dt:
        d = -np.expm1(-dt / tau)
        num, den = [d], [d - 1]
    else:
        num, den = [1.], []  # just copy the input
    return num, den


def alpha_transfer_fn(tau, dt):
    """Returns the transfer function description of an alpha synapse."""
    if tau > 0.03 * dt:
        a = dt / tau
        ea = np.exp(-a)
        num, den = [-a*ea + (1 - ea), ea*(a + ea - 1)], [-2 * ea, ea**2]
    else:
        num, den = [1.], []  # just copy the input
    return num, den


def discrete_delay(tf, steps):
    num, den = tf
    return [0]*steps + num, den


def lti(signals, transfer_fn, axis=0, normalize=True):
    """Linear time-invariant (LTI) system simulation.

    Uses the transfer function description of an LTI system
    to filter an input signal.

    Parameters
    ----------
    signals : array_like
        An array of signals to apply the LTI system to.
    transfer_fn : (num, den)
        A tuple of the transfer function numerator and denominator,
        both of which should be array_like.
    axis : int
        The axis along which to filter.
    """
    outputs = np.zeros_like(signals)

    signalsr = np.rollaxis(signals, axis)
    outputsr = np.rollaxis(outputs, axis)

    a, b = transfer_fn
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    if normalize:
        if b[0] != 1.:
            a = a / b[0]
            b = b / b[0]
        b = b[1:]  # drop first element (equal to 1)

    x = collections.deque(maxlen=len(a))
    y = collections.deque(maxlen=len(b))
    for i, si in enumerate(signalsr):
        x.appendleft(si)
        for k, xk in enumerate(x):
            outputsr[i] += a[k] * xk
        for k, yk in enumerate(y):
            outputsr[i] -= b[k] * yk
        y.appendleft(outputsr[i])

    return outputs


def norm(x, axis=None, keepdims=False):
    """Euclidean norm

    Parameters
    ----------
    x : array_like
        Array to compute the norm over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. `None` sums all axes. See `np.sum`.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See `np.sum` in
        newer versions of Numpy (>= 1.7).
    """
    y = np.sqrt(np.sum(x**2, axis=axis))
    return np.expand_dims(y, axis=axis) if keepdims else y


def meshgrid_nd(*args):
    args = [np.asarray(a) for a in args]
    s = len(args) * (1,)
    return np.broadcast_arrays(*(
        a.reshape(s[:i] + (-1,) + s[i + 1:]) for i, a in enumerate(args)))


def rms(x, axis=None, keepdims=False):
    """Root-mean-square amplitude

    Parameters
    ----------
    x : array_like
        Array to compute RMS amplitude over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. `None` sums all axes. See `np.sum`.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See `np.sum` in
        newer versions of Numpy (>= 1.7).
    """
    y = np.sqrt(np.mean(x**2, axis=axis))
    return np.expand_dims(y, axis=axis) if keepdims else y


def rmse(x, y, axis=None, keepdims=False):
    """Root-mean-square error amplitude

    Equivalent to rms(x - y, axis=axis, keepdims=keepdims).

    Parameters
    ----------
    x, y : array_like
        Arrays to compute RMS amplitude over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. `None` sums all axes. See `np.sum`.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See `np.sum` in
        newer versions of Numpy (>= 1.7).
    """
    return rms(x - y, axis=axis, keepdims=keepdims)


def generate_signal(n, upper, lower=0, norm=0.5, dt=0.001):
    """Returns a power-normalized signal between given frequencies

    Parameters
    ----------
    n : scalar
        The number of frequencies to generate. These will range from `0` to
        `(n-1)/(2*dt*n)` in increments of `dt`. Thus, the returned signal will
        have `2*n - 1` points in increments of `dt`.
    norm : float, optional
        The root-mean squared power (i.e. length) of the resulting signal.
        Defaults to 1.0.
    upper : float
        The largest frequency (Hz) to be contained in the signal. Any amount
        greater than or equal to `1/(2*dt)` will result in the highest possible
        frequency.
    lower : float, optional
        The smallest frequency (Hz) to be contained in the signal.
        Defaults to 0.
    dt : float, optional
        The time interval between points in the signal. Defaults to 1ms.
    """
    hertz = np.arange(n)/(2*dt*n)
    freqs = np.random.normal(size=n) + 1j*np.random.normal(size=n)
    freqs[(hertz < lower) | (hertz > upper)] = 0

    coeffs = np.append(freqs, np.conjugate(freqs[:0:-1]))
    x = np.fft.ifft(coeffs).real
    return norm * (x / rms(x))


@memoize
def dft_transform(n, mode='half'):
    """Returns the complex linear transform for an n-dimensional vector

    Parameters
    ----------
    n : int
        Dimensionality of input vector (number of columns in transform)
    mode : str, optional
        If 'half' (default), then the output dimension will be `n // 2 + 1`
        since the other half is the complex conjugate for real inputs.
        If 'full', then the output dimension will match the input.
    """
    if mode == 'full':
        m = n
    elif mode == 'half':
        m = (n // 2 + 1)
    else:
        raise ValueError("'mode' must be one of 'full' or 'half', got %s" %
                           mode)
    x = np.arange(n)
    w = np.arange(m)
    return np.exp((-2.j * np.pi / n) * (w[:, None] * x[None, :]))
