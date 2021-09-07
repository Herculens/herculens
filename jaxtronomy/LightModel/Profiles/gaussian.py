import numpy as np
import jax.numpy as jnp


class Gaussian(object):
    """
    class for Gaussian light profile
    The two-dimensional Gaussian profile amplitude is defined such that the 2D integral leads to the 'amp' value.

    profile name in LightModel module: 'GAUSSIAN'
    """
    def __init__(self):
        self.param_names = ['amp', 'sigma', 'center_x', 'center_y']
        self.lower_limit_default = {'amp': 0, 'sigma': 0, 'center_x': -100, 'center_y': -100}
        self.upper_limit_default = {'amp': 1000, 'sigma': 100, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        """
        surface brightness per angular unit

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        c = amp / (2 * np.pi * sigma**2)
        R2 = (x - center_x) ** 2 / sigma**2 + (y - center_y) ** 2 / sigma**2
        return c * jnp.exp(-R2 / 2.)

    def total_flux(self, amp, sigma, center_x=0, center_y=0):
        """
        integrated flux of the profile

        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        return amp

    def light_3d(self, r, amp, sigma):
        """
        3D brightness per angular volume element

        :param r: 3d distance from center of profile
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :return: 3D brightness per angular volume element
        """
        amp3d = amp / np.sqrt(2 * sigma**2) / np.sqrt(np.pi)
        sigma3d = sigma
        return self.function(r, 0, amp3d, sigma3d)
