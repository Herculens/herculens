# JAX implementation of the PIEMD profile for the lens modeling code GLEE.
#
# This is based on the GLEE implementation see Suyu & Halkola (2010) 
# and Suyu et al. (2012), which was ported to JAX in Wang et al. (2025). 
#
# This module is largely based on https://github.com/HanWang2021/Lens-Mass-Profile-GLaD-.
# The main change from the original code above is that lensing methods (e.g. get_deflection_angle)
# take as arguments the coordinates, rather than fixing those in the constructor.
# All methods specific to the use of the profile in the GLaD code have been removed as well.
#
# Copyright (c) 2025: Han Wang (MPA), Aymeric Galan (MPA)
# Copyright (c) 2023: herculens developers and contributors
# Copyright (c) 2010: Sherry Suyu, Aleksi Halkola (MPA)


__author__ = 'HanWang2021', 'aymgal'


from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node



def rotation(xx, yy, phi):
    cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
    xx_rot = cos_phi*xx + sin_phi*yy
    yy_rot = -sin_phi*xx + cos_phi*yy
    return xx_rot, yy_rot


class PIEMD(object):
    """
    Pseudo isothermal elliptical mass distribution (PIEMD), based on the
    implementation in the GLEE modelling code.
    
    The original GLEE parametrization is

    kappa(x,y) = E0 / (2*Sqrt(w^2 + rem^2)),
    where rem^2 = x^2/(1+e)^2 + y^2/(1-e)^2,

    with parameters:
    E0 is the lens strength,
    w is the core radius, and
    e is the ellipticity, e=(1.-q)/(1.+q)   [inverting gives q = (1-e)/(1+e)]

    In the limiting case where w=e=0, E0 is the Einstein radius of the SIS.

    If `scale_flag=False`, then E0 = thetaE

    If `scale_flag=True`, then E0 = thetaE, E0 = E0*E0 / (sqrt(E0 * E0 + w * w) - w)

    If (w < r_soft) then w will be set to `r_soft`.

    In the Herculens-adapted code below, we use theta_E parameter for E0 (or Elim), phi for the position angle, and q for the axis ratio, 
    and r_core for the core radius w.
    The 'scale' flag and softening radius are set at construction time.

    Parameters
    ----------
    r_soft : _type_, optional
        Softening radius, by default 1e-8
    scale_flag : bool, optional
        'scale' flag usually given in GLEE config files, by default True, 
        which leads to theta_E being simply the lensing strength.
    """

    param_names = ['theta_E', 'r_core', 'q', 'phi', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'r_core': 0, 'q': 0.2, 'phi': -np.pi/2., 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 10, 'r_core': 1e10, 'q': 1., 'phi': +np.pi/2., 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def __init__(self, r_soft=1e-8, scale_flag=True):
        self._r_soft = r_soft
        self._scale_flag = scale_flag
    
    @partial(jax.jit, static_argnums=(0,))
    def function(self, x, y, theta_E, r_core, q, phi, center_x=0, center_y=0):  # flags should be also an array
        # return lens potential
        e = (1.-q)/(1.+q)
        w = r_core
        theta_E = jnp.where(self._scale_flag, theta_E**2/(jnp.sqrt(theta_E*theta_E + w * w) - w), theta_E)
        return jax.lax.cond(
            e != 0, 
            self._potential_elliptical, 
            self._potential_spherical, 
            x, y, center_x, center_y, e, phi, theta_E, w
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def derivatives(self, x, y, theta_E, r_core, q, phi, center_x=0, center_y=0):
        # return deflection angle
        e = (1.-q)/(1.+q)
        w = r_core
        theta_E = jnp.where(self._scale_flag, theta_E**2/(jnp.sqrt(theta_E*theta_E + w * w) - w), theta_E)
        return jax.lax.cond(
            e!=0, 
            self._deflection_elliptical, 
            self._deflection_spherical, 
            x, y, center_x, center_y, e, phi, theta_E, w
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def hessian(self, x, y, theta_E, r_core, q, phi, center_x=0, center_y=0):
        e = (1.-q)/(1.+q)
        w = r_core
        theta_E = jnp.where(self._scale_flag, theta_E**2/(jnp.sqrt(theta_E*theta_E + w * w) - w), theta_E)
        return jax.lax.cond(
            e!=0, 
            self._hessian_elliptical, 
            self._hessian_spherical,
            x, y, center_x, center_y, e, phi, theta_E, w
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def kappa(self, x, y, theta_E, r_core, q, phi, center_x=0, center_y=0):
        xx = x - center_x
        yy = y - center_y
        xx, yy = rotation(xx, yy, phi)
        r2 = xx**2 + yy**2/(q*q)
        w = r_core
        theta_E = jnp.where(self._scale_flag, theta_E**2/(jnp.sqrt(theta_E*theta_E + w * w) - w), theta_E)
        kap = (theta_E/(1+q))*jnp.sqrt((1/(4*w**2/(1+q)**2 + r2)))
        return kap
    
    @partial(jax.jit, static_argnums=(0,))
    def _potential_spherical(self, x, y, x_centre, y_centre, _, pa, theta_E, w):
        # potential, given q = 1
        xx = x - x_centre
        yy = y - y_centre
        xx, yy = rotation(xx, yy, pa)
        rm = jnp.sqrt((xx)**2 + (yy)**2)
        phi = theta_E *(jnp.sqrt(rm*rm+w*w) - w*jnp.log(w+jnp.sqrt(rm*rm+w*w)) - w + w*jnp.log(2.*w) )
        return phi
    
    @partial(jax.jit, static_argnums=(0,))
    def _potential_elliptical(self, x, y, x_centre, y_centre, e, pa, theta_E, w):
        # potential, given q != 1
        xx = x - x_centre
        yy = y - y_centre
        xx, yy = rotation(xx, yy, pa)
        #jax.debug.print("theta_E: {}", theta_E)
        rem = jnp.sqrt(xx*xx/((1+e)*(1+e))+yy*yy/((1-e)*(1-e)))
        rm = jnp.sqrt(xx*xx + yy*yy)
        sang = yy/rm
        cang = xx/rm
        eta = -0.5*jnp.arcsinh((2.*jnp.sqrt(e)/(1.-e)) * sang) + 0.5j * jnp.arcsin((2.*jnp.sqrt(e)/(1.+e)) * cang)
        zeta = 0.5*jnp.log((rem + jnp.sqrt(rem*rem + w*w))/w) + 0.0j
        cosheta = jnp.cosh(eta)
        coshplus = jnp.cosh(eta+zeta)
        coshminus = jnp.cosh(eta-zeta)
        Kstar = jnp.sinh(2.*eta) * jnp.log(cosheta*cosheta/(coshplus*coshminus)) + jnp.sinh(2.*zeta) * jnp.log(coshplus/coshminus)
        phi = (theta_E*w*(1.-e*e)/(2.*rem*jnp.sqrt(e))) * jnp.imag((xx - 1j*yy) * Kstar)
        return phi

    def _deflection_spherical(self, x, y, x_centre, y_centre, e, pa, theta_E, w):
        # return deflection_angle, q = 1, e is not used here, passing it as input is only because of the syntax of jax.lax.cond()
        tol = 1.e-10
        xx =  x - x_centre
        yy =  y - y_centre
        xx, yy = rotation(xx, yy, pa)
        commen_factor = jnp.sqrt(w*w + xx*xx + yy*yy)/(xx*xx+yy*yy)
        condition  = (xx*xx+yy*yy)/(w*w)
        alp_x = jnp.where(condition>=tol, theta_E*xx*((-w/(xx*xx+ yy*yy) + commen_factor)), 0.5*theta_E*(xx)/w)
        alp_y = jnp.where(condition>=tol, theta_E*yy*(-(w/(xx*xx + yy*yy))+ commen_factor),  0.5*theta_E*(yy)/w)
        alp_x, alp_y = rotation(alp_x, alp_y, -pa)
        return alp_x, alp_y

    def _deflection_elliptical(self, x, y, x_centre, y_centre, e, pa, E0, w):
        # return deflection_angle, given q!=1
        dx =  x - x_centre
        dy =  y - y_centre
        dx, dy = rotation(dx, dy, pa)
        Istar = ((-0.5j) * (1 - (e * e)) * E0 * jnp.log((((1 - e) * dx) / (1 + e) - (1j * (1 + e) * dy) / (1 - e) + (2j) * jnp.sqrt(e) * jnp.sqrt(w * w + (dx * dx) / ((1 + e) * (1 + e)) + (dy * dy) / ((1 - e) * (1 - e)))) / ((2j) * jnp.sqrt(e) * w + dx - (1j) * dy))) / jnp.sqrt(e)
        alp_x, alp_y = jnp.real(Istar), jnp.imag(Istar)
        alp_x, alp_y = rotation(alp_x, alp_y, -pa)
        return alp_x, alp_y
    
    def _hessian_spherical(self, x, y, x_centre, y_centre, e, pa ,theta_E, w):
        xx =  x - x_centre
        yy =  y - y_centre
        dx, dy = rotation(xx, yy, pa)
        psi11 = (theta_E*(w*w*(-(dx)*(dx) + (dy)*(dy)) +(dy)*(dy)*((dx)*(dx) + (dy)*(dy)) +
		          w*((dx) - (dy))*((dx) + (dy))*jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy))))/(jnp.power((dx)*(dx) + (dy)*(dy),2)*jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy)))

        psi22 = (theta_E*(w*w*((dx) - (dy))*((dx) + (dy)) + (dx)*(dx)*((dx)*(dx) + (dy)*(dy)) + w*(-(dx)*(dx) + (dy)*(dy))* jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy))))/(jnp.power((dx)*(dx) + (dy)*(dy),2)*jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy)))
        psi12 = -((theta_E*(dx)*(dy)*((dx)*(dx) + (dy)*(dy) + 2*w*(w - jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy)))))/(jnp.power((dx)*(dx) + (dy)*(dy),2)*jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy))))
        return psi11, psi22, psi12

    def _hessian_elliptical(self, x, y, x_centre, y_centre, e, pa ,theta_E, w):
        xx =  x - x_centre
        yy =  y - y_centre
        dx, dy = rotation(xx, yy, pa)

        psi11= jnp.imag(((1 - (e*e))*theta_E*((2.j)*jnp.sqrt(e)*w + (dx) - (1.j)*(dy))*
		            (((1 - e)/(1 + e) + ((2.j)*jnp.sqrt(e)*(dx))/
		             (((1+e)*(1+e))*jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                    (dy)*(dy)/((1-e)*(1-e)))))/
		            ((2.j)*jnp.sqrt(e)*w + (dx) - (1.j)*(dy)) -
		             (((1 - e)*(dx))/(1 + e) - ((1.j)*(1 + e)*(dy))/(1 - e) +
		             (2.j)*jnp.sqrt(e)*
	               jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                  (dy)*(dy)/((1-e)*(1-e))))/
	              jnp.power((2.j)*jnp.sqrt(e)*w + (dx) - (1.j)*(dy),2)))/
		          (jnp.sqrt(e)*(((1 - e)*(dx))/(1 + e) -
		                    ((1.j)*(1 + e)*(dy))/(1 - e) +
		                    (2.j)*jnp.sqrt(e)*
		                  jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
	                          (dy)*(dy)/((1-e)*(1-e))))))/2.

        psi22=-jnp.real(((1 - (e*e))*theta_E*(2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy))*
		              ((((-1.j)*(1 + e))/(1 - e) +
		                (2.j*jnp.sqrt(e)*(dy))/
		               (((1-e)*(1-e))*
		                jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                      (dy)*(dy)/((1-e)*(1-e)))))/
		            (2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy)) +
	                 (1.j*(((1 - e)*(dx))/(1 + e) -
	                  (1.j*(1 + e)*(dy))/(1 - e) + 2.j*jnp.sqrt(e)*
		                   jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
	                        (dy)*(dy)/((1-e)*(1-e)))))/
	              jnp.power(2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy),2)))/
		          (jnp.sqrt(e)*(((1 - e)*(dx))/(1 + e) -
		                      (1.j*(1 + e)*(dy))/(1 - e) +
	                      2.j*jnp.sqrt(e)*jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                           (dy)*(dy)/((1-e)*(1-e))))))/2.

        psi12=-jnp.real(((1 - (e*e))*theta_E*(2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy))*
		             (((1 - e)/(1 + e) + (2.j*jnp.sqrt(e)*(dx))/
		                (((1+e)*(1+e))*jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                     (dy)*(dy)/((1-e)*(1-e)))))/
		              (2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy)) -
		              (((1 - e)*(dx))/(1 + e) - (1.j*(1 + e)*(dy))/(1 - e) +
	               2.j*jnp.sqrt(e)*jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                    (dy)*(dy)/((1-e)*(1-e))))/
		             jnp.power(2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy),2)))/
		 	   (jnp.sqrt(e)*(((1 - e)*(dx))/(1 + e) -
	                       (1.j*(1 + e)*(dy))/(1 - e) +
	                       2.j*jnp.sqrt(e)*jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
	                        (dy)*(dy)/((1-e)*(1-e))))))/2.
        return psi11, psi22, psi12

def flatten_func(self):
    children = ()
    aux_data = {
        'r_soft': self._r_soft,
        'scale_flag': self._scale_flag,
    }
    return (children, aux_data)

def unflatten_func(aux_data, children):
    # Here we avoid `__init__` because it has extra logic we don't require:
    obj = object.__new__(PIEMD)
    obj._r_soft = aux_data['r_soft']
    obj._flag_dpie = aux_data['scale_flag']
    return obj

register_pytree_node(PIEMD, flatten_func, unflatten_func)
