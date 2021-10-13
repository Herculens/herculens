import jax.numpy as jnp
from jax import lax


def phi_q2_ellipticity(phi, q):
    """
    transforms orientation angle and axis ratio into complex ellipticity moduli e1, e2

    :param phi: angle of orientation (in radian)
    :param q: axis ratio minor axis / major axis
    :return: eccentricities e1 and e2 in complex ellipticity moduli
    """
    e1 = (1. - q) / (1. + q) * jnp.cos(2 * phi)
    e2 = (1. - q) / (1. + q) * jnp.sin(2 * phi)
    return e1, e2

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
    # replace value by low float instead to avoid NaNs
    e1 = lax.cond(e1 == 0.0, lambda _: 1e-4, lambda _: e1, operand=None)
    e2 = lax.cond(e2 == 0.0, lambda _: 1e-4, lambda _: e2, operand=None)
    phi = jnp.arctan2(e2, e1) / 2
    c = jnp.sqrt(e1**2 + e2**2)
    c = jnp.minimum(c, 0.9999)
    q = (1. - c) / (1. + c)
    return phi, q

def shear_polar2cartesian(phi, gamma):
    """

    :param phi: shear angle (radian)
    :param gamma: shear strength
    :return: shear components gamma1, gamma2
    """
    gamma1 = gamma*jnp.cos(2*phi)
    gamma2 = gamma*jnp.sin(2*phi)
    return gamma1, gamma2

def shear_cartesian2polar(gamma1, gamma2):
    """
    :param gamma1: cartesian shear component
    :param gamma2: cartesian shear component
    :return: shear angle, shear strength
    """
    phi = jnp.arctan2(gamma2, gamma1) / 2
    gamma = jnp.sqrt(gamma1 ** 2 + gamma2 ** 2)
    return phi, gamma
    