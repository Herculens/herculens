# Defines en dual pseudo isothermal elliptical (dPIE) mass profile
# 
# Copyright (c) 2024, herculens developers and contributors

__author__ = 'aymgal'


import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit
from jax.tree_util import register_pytree_node, register_pytree_node_class

from herculens.Util import util, param_util

try:
    from herculens.MassModel.Profiles.glee.piemd_jax import Piemd_GPU
except ImportError:
    raise ImportError("Please contact the author to use this dPIE profile "
                        "as it depends on non-public libraries.")


__all__ = [
    'DPIE_GLEE', 
    'DPIE_GLEE_STATIC',
    'DPIE_PJAFFE',  # NOTE: the DPIE_PJAFFE is not well tested
]


@register_pytree_node_class
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
        self._piemd_flag = False
        self._piemd_cls = None

    def tree_flatten(self):
        children = (self._piemd_cls,)
        aux_data = {
            'r_soft': self._r_soft,
            'scale_flag': self._dpie_flag,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # return cls(*children, **aux_data)
        return cls(**aux_data)
    
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
        piemd = self._get_piemd(x, y)
        theta_E_scl, w, s = self._param_conv(theta_E, r_core, r_trunc, self._dpie_flag)
        f_w = piemd._potential(center_x, center_y, q, phi, theta_E_scl, w, False)
        f_s = piemd._potential(center_x, center_y, q, phi, theta_E_scl, s, False)
        f = f_w - f_s
        return f.reshape(*x.shape)

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
        piemd = self._get_piemd(x, y)
        theta_E_scl, w, s = self._param_conv(theta_E, r_core, r_trunc, self._dpie_flag)
        f_x_w, f_y_w = piemd._deflection_angle(center_x, center_y, q, phi, theta_E_scl, w, False)
        f_x_s, f_y_s = piemd._deflection_angle(center_x, center_y, q, phi, theta_E_scl, s, False)
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
        piemd = self._get_piemd(x, y)
        theta_E_scl, w, s = self._param_conv(theta_E, r_core, r_trunc, self._dpie_flag)
        f_xx_w, f_yy_w, f_xy_w = piemd._hessian(center_x, center_y, q, phi, theta_E_scl, w, False)
        f_xx_s, f_yy_s, f_xy_s = piemd._hessian(center_x, center_y, q, phi, theta_E_scl, s, False)
        f_xx = f_xx_w - f_xx_s
        f_yy = f_yy_w - f_yy_s
        f_xy = f_xy_w - f_xy_s
        return f_xx.reshape(*x.shape), f_yy.reshape(*y.shape), f_xy.reshape(*y.shape)
    
    @jit
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
    
    @jit
    def _get_piemd(self, x, y):
        # NOTE: first 4 arguments of Piemd_GPU do not matter for our use, so we give zeros
        return self._piemd_cls(0., 0., 0., 0., xx=x, yy=y)

    @staticmethod
    def _get_piemd(x, y):
        return Piemd_GPU(0., 0., 0., 0., xx=x, yy=y)


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


class DPIE_PJAFFE(object):
    """
    TODO: finish implementation.

    Dual pseudo isothermal elliptical (dPIE) mass profile.

    The implementation tries to follow the GLEE definitions.

    The convergence is
    kappa(x,y) = (Elimit / 2) * (s^2/(s^2-w^2)) * (1/sqrt(w^2 + rem^2) - 1/sqrt(s^2 + rem^2)
    
    with parameters
    Elimit = strength, (= Einstein radius of SIS in limiting case of s->inf, w->0)
    w = core radius
    s = truncation/scale radius (that must be >w)
    rem = elliptical mass radius = sqrt(x^2/(1+e)^2 + y^2/(1-e)^2)
    e = ellipticity = (1-q)/(1+q)
    q = axis ratio (that is <=1)

    The 3D density is
    rho(x,y,z) = rho0 / ((1 + r3D^2/w^2)*(1 + r3D^2/s^2)),    s>w

    It uses as backend the PJaffe from JAXtronomy, originally parametrized differently.
    """
    param_names = ['theta_E', 'r_core', 'r_trunc', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'r_core': 0, 'r_trunc': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 10, 'r_core': 1e10, 'r_trunc': 1e10, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}
    
    def __init__(self):
        self._r_min = self._backend._s
        self._r_max = 1e10
        try:
            from jaxtronomy.LensModel.Profiles.p_jaffe import PJaffe
        except ImportError:
            raise ImportError("JAXtronomy needs to be installed to use the DPIE profile.")
        else:
            self._backend = PJaffe()

    def function(self, x, y, theta_E, r_core, r_trunc, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param r_core: core radius
        :param r_trunc: truncation radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        x_ = x - center_x
        y_ = y - center_y
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        x_, y_ = util.rotate(x_, y_, phi)
        y_ /= q
        sigma0, Ra, Rs = self._conv_param(theta_E, r_core, r_trunc, q)
        f = self._backend.function(
            x_, y_,
            sigma0, Ra, Rs,
            center_x=center_x, center_y=center_y,
        )
        return f

    def derivatives(self, x, y, theta_E, r_core, r_trunc, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param r_core: core radius
        :param r_trunc: truncation radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        x_ = x - center_x
        y_ = y - center_y
        phi, q = param_util.ellipticity2phi_q(e1, e2)

        # x_, y_ = util.rotate(x_, y_, phi)
        # e = 1 - q
        # y_ /= q
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, 0, 0)

        sigma0, Ra, Rs = self._conv_param(theta_E, r_core, r_trunc, q)
        f_x_, f_y_ =  self._backend.derivatives(
            x_, y_,
            sigma0, Ra, Rs,
            center_x=center_x, center_y=center_y,
        )

        # f_y_ /= q
        e = param_util.q2e(q)
        f_x_ *= np.sqrt(1 - e)
        f_y_ *= np.sqrt(1 + e)
        f_x_, f_y_ = util.rotate(f_x_, f_y_, - phi)

        f_x, f_y = f_x_, f_y_
        return f_x, f_y
    
    def hessian(self, x, y, theta_E, r_core, r_trunc, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param r_core: core radius
        :param r_trunc: truncation radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        return 0., 0., 0.

    # def hessian(self, x, y, theta_E, e1, e2, gamma, center_x=0, center_y=0):
    #     """

    #     :param x: x-coordinate in image plane
    #     :param y: y-coordinate in image plane
    #     :param theta_E: Einstein radius
    #     :param e1: eccentricity component
    #     :param e2: eccentricity component
    #     :param t: power law slope
    #     :param center_x: profile center
    #     :param center_y: profile center
    #     :return: f_xx, f_yy, f_xy
    #     """

    #     b, t, q, phi_G = self.param_conv(theta_E, e1, e2, gamma)
    #     # shift
    #     x_ = x - center_x
    #     y_ = y - center_y
    #     # rotate
    #     x__, y__ = util.rotate(x_, y_, phi_G)
    #     # evaluate
    #     f__xx, f__yy, f__xy = self.epl_major_axis.hessian(x__, y__, b, t, q)
    #     # rotate back
    #     kappa = 1./2 * (f__xx + f__yy)
    #     gamma1__ = 1./2 * (f__xx - f__yy)
    #     gamma2__ = f__xy
    #     gamma1 = jnp.cos(2 * phi_G) * gamma1__ - jnp.sin(2 * phi_G) * gamma2__
    #     gamma2 = +jnp.sin(2 * phi_G) * gamma1__ + jnp.cos(2 * phi_G) * gamma2__
    #     f_xx = kappa + gamma1
    #     f_yy = kappa - gamma1
    #     f_xy = gamma2
    #     return f_xx, f_yy, f_xy

    def _conv_param(self, theta_E, r_core, r_trunc, q):
        """
        Converts parameters from the parametrization used in Herculens
        to the one used in JAXtronomy.
        The chosen parametrization is such that it is consistent with GLEE.
        """
        r_core, r_trunc = self._check_radii(r_core, r_trunc)
        Ra = r_core # * 2. / (1 + q)
        Rs = r_trunc # * 2. / (1 + q)
        prefac_g = 0.5 * r_trunc**2 / (r_trunc**2 - r_core**2)  # GLEE convergence prefactor
        prefac_l = Ra * Rs / (Ra + Rs)  # JAXtronomy convergence prefactor
        sigma0 = (theta_E * prefac_g / prefac_l) # / (1 + q)
        return sigma0, Ra, Rs
    
    def _check_radii(self, r_core, r_trunc):
        r_core = jnp.where(r_core < self._r_min, self._r_min, r_core)
        r_trunc = jnp.where(r_core > self._r_max, self._r_max, r_trunc)
        return r_core, r_trunc
    
    # def _theta_E_q_convert(self, theta_E, q):
    #     """
    #     converts a spherical averaged Einstein radius to an elliptical (major axis) Einstein radius.
    #     This then follows the convention of the SPEMD profile in lenstronomy.
    #     (theta_E / theta_E_gravlens) = sqrt[ (1+q^2) / (2 q) ]

    #     :param theta_E: Einstein radius in lenstronomy conventions
    #     :param q: axis ratio minor/major
    #     :return: theta_E in convention of kappa=  b *(q2(s2 + x2) + y2􏰉)−1/2
    #     """
    #     theta_E_new = theta_E / (jnp.sqrt((1. + q**2) / (2. * q)))
    #     return theta_E_new