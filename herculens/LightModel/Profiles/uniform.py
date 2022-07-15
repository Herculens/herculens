# Defines a uniform profile
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'austinpeel', 'aymgal'


import jax.numpy as jnp


__all__ = ['Uniform']


class Uniform(object):
    """
    class for uniform light profile
    """
    param_names = ['amp']
    lower_limit_default = {'amp': -100}
    upper_limit_default = {'amp': 100}
    fixed_default = {key: False for key in param_names}

    def __init__(self):
        pass

    def function(self, x, y, amp):
        """

        :param x:
        :param y:
        :param sigma0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        return jnp.ones_like(x) * amp
