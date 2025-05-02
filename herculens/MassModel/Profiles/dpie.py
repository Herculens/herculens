# Defines en dual pseudo isothermal elliptical (dPIE) mass profile
#
# This is based on the GLEE definition of the profile (Suyu & Halkola 2010, Suyu et al. 2012).
# 
# Copyright (c) 2025: herculens developers and contributors


__author__ = 'aymgal'


import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit
from jax.tree_util import register_pytree_node

from herculens.MassModel.Profiles.piemd import PIEMD


__all__ = [
    'DPIE', 
]


class DPIE(object):
    """
    Dual pseudo isothermal elliptical (dPIE) mass profile, based on the
    different of two PIEMD profiles as implemented in GLEE.

    The convergence is
    kappa(x,y) = (Elimit / 2) * (s^2/(s^2-w^2)) * (1/sqrt(w^2 + rem^2) - 1/sqrt(s^2 + rem^2)
    
    with parameters
    Elimit = strength, (= Einstein radius of SIS in limiting case of s->inf, w->0)
    w = core radius
    s = truncation/scale radius (that must be >w)
    rem = elliptical mass radius = sqrt(x^2/(1+e)^2 + y^2/(1-e)^2)
    e = ellipticity = (1-q)/(1+q)
    q = axis ratio (that is <=1)

    Parameters
    ----------
    r_soft : _type_, optional
        Softening radius, by default 1e-8
    scale_flag : bool, optional
        'scale' flag usually given in GLEE config files, by default True, 
        which leads to theta_E being simply the lensing strength.
    """
    param_names = ['theta_E', 'r_core', 'r_trunc', 'q', 'phi', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'r_core': 0, 'r_trunc': 0, 'q': 0.2, 'phi': -np.pi/2., 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 10, 'r_core': 1e10, 'r_trunc': 1e10, 'q': 1., 'phi': +np.pi/2., 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}
    
    def __init__(self, r_soft=1e-8, scale_flag=True):
        self._r_soft = r_soft
        self._scale_flag = scale_flag # if True, theta_E corresponds to the Einstein radius of the profile
        self._piemd = PIEMD(r_soft=self._r_soft, scale_flag=False)  # the PIEMD scale flag must be False here.
    
    @partial(jit, static_argnums=(0,))
    def function(self, x, y, theta_E, r_core, r_trunc, q, phi, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param r_core: core radius
        :param r_trunc: truncation radius
        :param q: axis ratio
        :param phi: position angle
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        theta_E_piemd, w, s = self._param_conv(theta_E, r_core, r_trunc, self._scale_flag)
        f_w = self._piemd.function(x, y, theta_E_piemd, w, q, phi, center_x=center_x, center_y=center_y)
        f_s = self._piemd.function(x, y, theta_E_piemd, s, q, phi, center_x=center_x, center_y=center_y)
        f = f_w - f_s
        return f

    @partial(jit, static_argnums=(0,))
    def derivatives(self, x, y, theta_E, r_core, r_trunc, q, phi, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param r_core: core radius
        :param r_trunc: truncation radius
        :param q: axis ratio
        :param phi: position angle
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        theta_E_piemd, w, s = self._param_conv(theta_E, r_core, r_trunc, self._scale_flag)
        f_x_w, f_y_w = self._piemd.derivatives(x, y, theta_E_piemd, w, q, phi, center_x=center_x, center_y=center_y)
        f_x_s, f_y_s = self._piemd.derivatives(x, y, theta_E_piemd, s, q, phi, center_x=center_x, center_y=center_y)
        f_x = f_x_w - f_x_s
        f_y = f_y_w - f_y_s
        return f_x, f_y
    
    @partial(jit, static_argnums=(0,))
    def hessian(self, x, y, theta_E, r_core, r_trunc, q, phi, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param r_core: core radius
        :param r_trunc: truncation radius
        :param q: axis ratio
        :param phi: position angle
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        theta_E_piemd, w, s = self._param_conv(theta_E, r_core, r_trunc, self._scale_flag)
        f_xx_w, f_yy_w, f_xy_w = self._piemd.hessian(x, y, theta_E_piemd, w, q, phi, center_x=center_x, center_y=center_y)
        f_xx_s, f_yy_s, f_xy_s = self._piemd.hessian(x, y, theta_E_piemd, s, q, phi, center_x=center_x, center_y=center_y)
        f_xx = f_xx_w - f_xx_s
        f_yy = f_yy_w - f_yy_s
        f_xy = f_xy_w - f_xy_s
        return f_xx, f_yy, f_xy
    
    def _param_conv(self, theta_E, r_core, r_trunc, scale_flag):
        """This functions follows the GLEE documentation"""
        w, s = r_core, r_trunc
        w, s = self._check_radii(w, s)
        if scale_flag is True:
            E0 = theta_E
            prefac = E0**2 / ( (jnp.sqrt(w**2 + E0**2) - w) - (jnp.sqrt(s**2 + E0**2) - s) )
        else:
            Elim = theta_E
            prefac = Elim * s**2 / (s**2 - w**2)
        return prefac, w, s
    
    def _check_radii(self, w, s):
        # make sure the core radius parameters do not go below some small value for numerical stability
        w = jnp.where(w < self._r_soft, self._r_soft, w)
        # NOTE: the following swap of values *may* cause issues with JAX autodiff
        w_ = jnp.where(s < w, s, w)
        s_ = jnp.where(s < w, w, s)
        return w_, s_

def flatten_func(self):
    children = (self._piemd,)
    aux_data = {
        'r_soft': self._r_soft,
        'scale_flag': self._scale_flag,
    }
    return (children, aux_data)

def unflatten_func(aux_data, children):
    # Here we avoid `__init__` because it has extra logic we don't require:
    obj = object.__new__(DPIE)
    obj._piemd = children[0]
    obj._r_soft = aux_data['r_soft']
    obj._flag_dpie = aux_data['scale_flag']
    return obj

register_pytree_node(DPIE, flatten_func, unflatten_func)
