# Utility functions
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the Util module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp
from jax import lax


def phi_q2_ellipticity(phi, q):
    """
    transforms orientation angle and axis ratio into complex ellipticity moduli e1, e2

    :param phi: angle of orientation (in radian)
    :param q: axis ratio minor axis / major axis
    :return: eccentricities e1 and e2 in complex ellipticity moduli
    """
    # e1 = jnp.where(q == 1., 1e-8, (1. - q) / (1. + q) * jnp.cos(2. * phi))
    # e2 = jnp.where(q == 1., 1e-8, (1. - q) / (1. + q) * jnp.sin(2. * phi))
    e1 = (1. - q) / (1. + q) * jnp.cos(2. * phi)
    e2 = (1. - q) / (1. + q) * jnp.sin(2. * phi)
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
    # e1 = lax.cond(e1 == 0.0, lambda _: 1e-4, lambda _: e1, operand=None)  # does not work with TFP!
    # e2 = lax.cond(e2 == 0.0, lambda _: 1e-4, lambda _: e2, operand=None)  # does not work with TFP!
    # e1 = jnp.where(e1 == 0., 1e-8, e1)
    # e2 = jnp.where(e2 == 0., 1e-8, e2)
    phi = jnp.arctan2(e2, e1) / 2.
    c = jnp.sqrt(e1**2 + e2**2)
    c = jnp.minimum(c, 0.9999)
    q = (1. - c) / (1. + c)
    return phi, q

def ellipticity2phi_q_numpy(e1, e2):
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
    e1_ = np.copy(e1)
    e1_[e1 == 0.] == 1e-4
    e2_ = np.copy(e2)
    e2_[e2 == 0.] == 1e-4
    phi = np.arctan2(e2_, e1_) / 2
    c = np.sqrt(e1_**2 + e2_**2)
    c = np.minimum(c, 0.9999)
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

def shear_cartesian2polar_numpy(gamma1, gamma2):
    """
    :param gamma1: cartesian shear component
    :param gamma2: cartesian shear component
    :return: shear angle, shear strength
    """
    gamma1 = np.array(gamma1)
    gamma2 = np.array(gamma2)
    phi = np.arctan2(gamma2, gamma1) / 2
    gamma = np.sqrt(gamma1 ** 2 + gamma2 ** 2)
    return phi, gamma
    
def cart2polar(x, y, center_x=0, center_y=0):
    """
    transforms cartesian coords [x,y] into polar coords [r,phi] in the frame of the lens center

    :param x: set of x-coordinates
    :type x: array of size (n)
    :param y: set of x-coordinates
    :type y: array of size (n)
    :param center_x: rotation point
    :type center_x: float
    :param center_y: rotation point
    :type center_y: float
    :returns:  array of same size with coords [r,phi]
    """
    coord_shift_x = x - center_x
    coord_shift_y = y - center_y
    r = jnp.sqrt(coord_shift_x**2+coord_shift_y**2)
    phi = jnp.arctan2(coord_shift_y, coord_shift_x)
    return r, phi

def polar2cart(r, phi, center):
    """
    transforms polar coords [r,phi] into cartesian coords [x,y] in the frame of the lense center

    :param coord: set of coordinates
    :type coord: array of size (n,2)
    :param center: rotation point
    :type center: array of size (2)
    :returns:  array of same size with coords [x,y]
    :raises: AttributeError, KeyError
    """
    x = r*jnp.cos(phi)
    y = r*jnp.sin(phi)
    return x - center[0], y - center[1]

def transform_e1e2_product_average(x, y, e1, e2, center_x, center_y):
    """
    maps the coordinates x, y with eccentricities e1 e2 into a new elliptical coordinate system
    such that R = sqrt(R_major * R_minor)

    :param x: x-coordinate
    :param y: y-coordinate
    :param e1: eccentricity
    :param e2: eccentricity
    :param center_x: center of distortion
    :param center_y: center of distortion
    :return: distorted coordinates x', y'
    """
    phi_G, q = ellipticity2phi_q(e1, e2)
    x_shift = x - center_x
    y_shift = y - center_y

    cos_phi = jnp.cos(phi_G)
    sin_phi = jnp.sin(phi_G)

    xt1 = cos_phi * x_shift + sin_phi * y_shift
    xt2 = -sin_phi * x_shift + cos_phi * y_shift
    return xt1 * jnp.sqrt(q), xt2 / jnp.sqrt(q)

def transform_e1e2_square_average(x, y, e1, e2, center_x, center_y):
    """
    maps the coordinates x, y with eccentricities e1 e2 into a new elliptical coordinate system
    such that R = sqrt(R_major**2 + R_minor**2)

    :param x: x-coordinate
    :param y: y-coordinate
    :param e1: eccentricity
    :param e2: eccentricity
    :param center_x: center of distortion
    :param center_y: center of distortion
    :return: distorted coordinates x', y'
    """
    phi_G, q = ellipticity2phi_q(e1, e2)
    x_shift = x - center_x
    y_shift = y - center_y
    cos_phi = jnp.cos(phi_G)
    sin_phi = jnp.sin(phi_G)
    e = q2e(q)
    x_ = (cos_phi * x_shift + sin_phi * y_shift) * jnp.sqrt(1 - e)
    y_ = (-sin_phi * x_shift + cos_phi * y_shift) * jnp.sqrt(1 + e)
    return x_, y_


def q2e(q):
    e = jnp.abs(1 - q**2) / (1 + q**2)
    return e


def statistics_from_samples(samples, losses):
    min_loss_idx = np.argmin(losses)
    map_values = samples[min_loss_idx, :]
    mean_values = np.mean(samples, axis=0)
    perc_16, perc_50, perc_84 = np.percentile(samples, q=[16, 50, 84], axis=0)
    median_values = perc_50
    return map_values, mean_values, median_values, perc_16, perc_84
