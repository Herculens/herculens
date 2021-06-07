import jax.numpy as np


# def phi_q2_ellipticity(phi, q):
#     """
#     transforms orientation angle and axis ratio into complex ellipticity moduli e1, e2
#
#     :param phi: angle of orientation (in radian)
#     :param q: axis ratio minor axis / major axis
#     :return: eccentricities e1 and e2 in complex ellipticity moduli
#     """
#     e1 = (1. - q) / (1. + q) * np.cos(2 * phi)
#     e2 = (1. - q) / (1. + q) * np.sin(2 * phi)
#     return e1, e2


def ellipticity2phi_q(e1, e2):
    """Transform complex ellipticity components to position angle and axis ratio.

    Parameters
    ----------
    e1, e2 : float or array_like
        Ellipticity components.

    Returns
    -------
    phi, q : same type as e1, e2
        Position angle (rad) and axis ratio (semi-minor / semi-major axis)

    """
    phi = np.arctan2(e2, e1) / 2
    c = np.sqrt(e1**2 + e2**2)
    c = np.minimum(c, 0.9999)
    q = (1. - c) / (1. + c)
    return phi, q
