# Defines en dual pseudo isothermal elliptical (dPIE) mass profile
# 
# Copyright (c) 2024, herculens developers and contributors

__author__ = 'aymgal'


import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit
from jax.tree_util import register_pytree_node

from herculens.Util import util, param_util


__all__ = [
    'DPIE_GLEE', 
    'DPIE_GLEE_STATIC',
]

try:
    from herculens.MassModel.Profiles.glee.piemd_jax import Piemd_GPU
except ImportError:
    print("WARNING: dPIE profile class cannot be loaded. "
          "If you wish to use this profile, "
          "please contact the author to use this dPIE profile "
          "as it depends on non-public libraries.")
    class DPIE_GLEE(object):
        pass
    class DPIE_GLEE_STATIC(object):
        pass


class DPIE_GLEE(object):
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

    """
    param_names = ['theta_E', 'r_core', 'r_trunc', 'q', 'phi', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'r_core': 0, 'r_trunc': 0, 'q': 0.2, 'phi': -np.pi/2., 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 10, 'r_core': 1e10, 'r_trunc': 1e10, 'q': 1., 'phi': +np.pi/2., 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}
    
    def __init__(self, r_soft=1e-8, scale_flag=True):
        self._r_soft = r_soft
        self._dpie_flag = scale_flag # if True, theta_E corresponds to the Einstein radius of the profile
        self._piemd = None
        self._piemd_flag = False
    
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
        piemd = self.piemd(x, y)
        theta_E_scl, w, s = self._param_conv(theta_E, r_core, r_trunc, self._dpie_flag)
        f_w = piemd._potential(center_x, center_y, q, phi, theta_E_scl, w, self._piemd_flag)
        f_s = piemd._potential(center_x, center_y, q, phi, theta_E_scl, s, self._piemd_flag)
        f = f_w - f_s
        return f.reshape(*x.shape)

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
        piemd = self.piemd(x, y)
        theta_E_scl, w, s = self._param_conv(theta_E, r_core, r_trunc, self._dpie_flag)
        f_x_w, f_y_w = piemd._deflection_angle(center_x, center_y, q, phi, theta_E_scl, w, self._piemd_flag)
        f_x_s, f_y_s = piemd._deflection_angle(center_x, center_y, q, phi, theta_E_scl, s, self._piemd_flag)
        f_x = f_x_w - f_x_s
        f_y = f_y_w - f_y_s
        return f_x.reshape(*x.shape), f_y.reshape(*y.shape)
    
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
        piemd = self.piemd(x, y)
        theta_E_scl, w, s = self._param_conv(theta_E, r_core, r_trunc, self._dpie_flag)
        f_xx_w, f_yy_w, f_xy_w = piemd._hessian(center_x, center_y, q, phi, theta_E_scl, w, self._piemd_flag)
        f_xx_s, f_yy_s, f_xy_s = piemd._hessian(center_x, center_y, q, phi, theta_E_scl, s, self._piemd_flag)
        f_xx = f_xx_w - f_xx_s
        f_yy = f_yy_w - f_yy_s
        f_xy = f_xy_w - f_xy_s
        return f_xx.reshape(*x.shape), f_yy.reshape(*y.shape), f_xy.reshape(*y.shape)
    
    @partial(jit, static_argnums=(0,))
    def _param_conv(self, theta_E, r_core, r_trunc, scale_flag):
        w, s = r_core, r_trunc
        # w, s = self._check_radii(r_core, r_trunc)
        w2 = w**2
        s2 = s**2
        if scale_flag is True:
            theta_E2 = theta_E**2
            theta_E_scaled = theta_E2 / ( (jnp.sqrt(w2 + theta_E2) - w) - (jnp.sqrt(s2 + theta_E2) - s) )
        else:
            theta_E_scaled = theta_E * s2 / (s2 - w2)
        return theta_E_scaled, w, s
    
    # @jit
    # def _check_radii(self, w, s):
    #     # make sure the core radius parameters do not go below some small value for numerical stability
    #     w = jnp.where(w < self._r_soft, self._r_soft, w)
    #     # NOTE: the following swap of values *may* cause issues with JAX autodiff
    #     w_ = jnp.where(s < w, s, w)
    #     s_ = jnp.where(s < w, w, s)
    #     return w_, s_
    
    @partial(jit, static_argnums=(0,))
    def piemd(self, x, y):
        if self._piemd is None:
            # NOTE: first 4 arguments of Piemd_GPU do not matter for our use, so we give zeros
            self._piemd = Piemd_GPU(0., 0., 0., 0., xx=x, yy=y)
        return self._piemd

def flatten_func(self):
    children = (self._piemd,)
    aux_data = {
        'r_soft': self._r_soft,
        'scale_flag': self._dpie_flag,
    }
    return (children, aux_data)

def unflatten_func(aux_data, children):
    # Here we avoid `__init__` because it has extra logic we don't require:
    obj = object.__new__(DPIE_GLEE)
    obj._piemd = children[0]
    obj._r_soft = aux_data['r_soft']
    obj._dpie_flag = aux_data['scale_flag']
    return obj

register_pytree_node(DPIE_GLEE, flatten_func, unflatten_func)



class DPIE_GLEE_STATIC(object):
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

    """
    param_names = ['theta_E', 'r_core', 'r_trunc', 'q', 'phi', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'r_core': 0, 'r_trunc': 0, 'q': 0.2, 'phi': -np.pi/2., 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 10, 'r_core': 1e10, 'r_trunc': 1e10, 'q': 1., 'phi': +np.pi/2., 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}
    
    def __init__(self, scale_flag=True):
        self._r_soft = 1e-8
        self._dpie_flag = scale_flag # if True, theta_E corresponds to the Einstein radius of the profile
        self._piemd = None

    def set_eval_coord_grid(self, x, y):
        self._piemd = Piemd_GPU(0., 0., 0., 0., xx=x, yy=y)

    @jit
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
        theta_E_scl, w, s = self._param_conv(theta_E, r_core, r_trunc, self._dpie_flag)
        f_w = self._piemd._potential(center_x, center_y, q, phi, theta_E_scl, w, flags=False)
        f_s = self._piemd._potential(center_x, center_y, q, phi, theta_E_scl, s, flags=False)
        f = f_w - f_s
        return f.reshape(*x.shape)

    # @partial(jit, static_argnums=(0, 1, 2))
    @jit
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
        theta_E_scl, w, s = self._param_conv(theta_E, r_core, r_trunc, self._dpie_flag)
        f_x_w, f_y_w = self._piemd._deflection_angle(center_x, center_y, q, phi, theta_E_scl, w, flags=False)
        f_x_s, f_y_s = self._piemd._deflection_angle(center_x, center_y, q, phi, theta_E_scl, s, flags=False)
        f_x = f_x_w - f_x_s
        f_y = f_y_w - f_y_s
        return f_x.reshape(*x.shape), f_y.reshape(*y.shape)
    
    @jit
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
        theta_E_scl, w, s = self._param_conv(theta_E, r_core, r_trunc, self._dpie_flag)
        f_xx_w, f_yy_w, f_xy_w = self._piemd._hessian(center_x, center_y, q, phi, theta_E_scl, w, flags=False)
        f_xx_s, f_yy_s, f_xy_s = self._piemd._hessian(center_x, center_y, q, phi, theta_E_scl, s, flags=False)
        f_xx = f_xx_w - f_xx_s
        f_yy = f_yy_w - f_yy_s
        f_xy = f_xy_w - f_xy_s
        return f_xx.reshape(*x.shape), f_yy.reshape(*y.shape), f_xy.reshape(*y.shape)
    
    def _param_conv(self, theta_E, r_core, r_trunc, scale_flag):
        w, s = r_core, r_trunc
        # w, s = self._check_radii(w, s)
        w2 = w**2
        s2 = s**2
        if scale_flag is True:
            theta_E2 = theta_E**2
            theta_E_scaled = theta_E2 / ( (jnp.sqrt(w2 + theta_E2) - w) - (jnp.sqrt(s2 + theta_E2) - s) )
        else:
            theta_E_scaled = theta_E * s2 / (s2 - w2)
        return theta_E_scaled, w, s
    
    # TODO: it seems that it leads to larger compilation time, but this needs to be more tested
    # def _check_radii(self, w, s):
    #     # make sure the core radius parameters do not go below some small value for numerical stability
    #     w = jnp.where(w < self._r_soft, self._r_soft, w)
    #     # NOTE: the following swap of values *may* cause issues with JAX autodiff
    #     w_ = jnp.where(s < w, s, w)
    #     s_ = jnp.where(s < w, w, s)
    #     return w_, s_

def flatten_func(self):
    children = (self._piemd,)
    aux_data = {
        'r_soft': self._r_soft,
        'scale_flag': self._dpie_flag,
    }
    return (children, aux_data)

def unflatten_func(aux_data, children):
    # Here we avoid `__init__` because it has extra logic we don't require:
    obj = object.__new__(DPIE_GLEE_STATIC)
    obj._piemd = children[0]
    obj._r_soft = aux_data['r_soft']
    obj._dpie_flag = aux_data['scale_flag']
    return obj

register_pytree_node(DPIE_GLEE_STATIC, flatten_func, unflatten_func)
