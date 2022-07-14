# Defines a multipole in the potential
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel.Profiles module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'lynevdv', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp

import herculens.Util.param_util as param_util


__all__ = ['Multipole']


class Multipole(object):
    """
    This class contains a multipole contribution (for 1 component with m>=2)
    This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf
    m : int, multipole order, m>=2
    a_m : float, multipole strength
    phi_m : float, multipole orientation in radian
    """
    param_names = ['m', 'a_m', 'phi_m', 'center_x', 'center_y']
    lower_limit_default = {'m': 2,'a_m':0, 'phi_m':-np.pi, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'m': 100,'a_m':100, 'phi_m':np.pi, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def function(self, x, y, m, a_m, phi_m, center_x=0, center_y=0):
        """
        Lensing potential of multipole contribution (for 1 component with m>=2)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param m: int, multipole order, m>=2
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: x-position
        :return: lensing potential
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        f_ = r*a_m /(1-m**2) * jnp.cos(m*(phi-phi_m))
        return f_

    def derivatives(self,x,y, m, a_m, phi_m, center_x=0, center_y=0):
        """
        Deflection of a multipole contribution (for 1 component with m>=2)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param m: int, multipole order, m>=2
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: x-position
        :return: deflection angles alpha_x, alpha_y
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        f_x = jnp.cos(phi)*a_m/(1-m**2) * jnp.cos(m*(phi-phi_m)) + jnp.sin(phi)*m*a_m/(1-m**2)*jnp.sin(m*(phi-phi_m))
        f_y = jnp.sin(phi)*a_m/(1-m**2) * jnp.cos(m*(phi-phi_m)) - jnp.cos(phi)*m*a_m/(1-m**2)*jnp.sin(m*(phi-phi_m))
        return f_x, f_y

    def hessian(self, x, y, m, a_m, phi_m, center_x=0, center_y=0):
        """
        Hessian of a multipole contribution (for 1 component with m>=2)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param m: int, multipole order, m>=2
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: x-position
        :return: f_xx, f_xy, f_yx, f_yy
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = jnp.maximum(r, 0.000001)
        f_xx = 1./r * jnp.sin(phi)**2 * a_m * jnp.cos(m*(phi-phi_m))
        f_yy = 1./r * jnp.cos(phi)**2 * a_m * jnp.cos(m*(phi-phi_m))
        f_xy = -1./r * a_m * jnp.cos(phi) * jnp.sin(phi) * jnp.cos(m*(phi-phi_m))
        return f_xx, f_xy, f_xy, f_yy
