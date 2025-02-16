import numpy as np
import sympy as sp


def gauss_heat(t, t0, FWHM):
    """Gauss pulse

    :param t: times
    :type t: array-like
    :param t0: shift, s
    :type t0: float
    :param FWHM: FWHM, s
    :type FWHM: float
    :return: impulse form
    :rtype: array-like
    """

    sigma = FWHM / 2 / np.sqrt(2 * np.log(2))  # standard deviation, s
    return 1 / np.sqrt(2 * np.pi) / sigma * sp.exp(-((t - t0) ** 2) / 2 / sigma**2)


def trapezoid_heat(t, t1, t2, delta, tau_le, tau_te):
    """Trapezoid pulse

    :param t: times
    :type t: array-like
    :param t1: rising time, s
    :type t1: float
    :param t2: plateau time, s
    :type t2: float
    :param delta: slope
    :type delta: float
    :param tau_le: leading edge, s
    :type tau_le: float
    :param tau_te: trailing edge, s
    :type tau_te: float
    :return: impulse form
    :rtype: array-like
    """

    f1 = lambda t: 1 - sp.exp(-t / tau_le)
    f2 = lambda t: f1(t1) * (1 - delta * (t - t1) / (t2 - t1))
    f3 = lambda t: f2(t2) * (sp.exp(-(t - t2) / tau_te))
    shape = lambda t: sp.Piecewise(
        (f1(t), t <= t1),
        (f2(t), (t > t1) & (t <= t2)),
        (f3(t), True),
    )

    norm = (
        t1
        + tau_le * (np.exp(-t1 / tau_le) - 1)
        + f1(t1) * (t2 - t1) * (1 - delta / 2)
        + f2(t2) * tau_te
    )
    return shape(t) / norm
