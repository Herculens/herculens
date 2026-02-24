# Defines Sersic profiles
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LightModel.Profiles module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'jiwoncpark', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp
from jax.scipy import special
from jax import jit, grad

import herculens.Util.param_util as param_util


__all__ = ['Sersic', 'CoreSersic']


class SersicBase(object):

    def __init__(
            self, 
            smoothing=0.0001, 
            exponent=2, 
            radius_major_axis=True
        ):
        """Base class for Sersic profiles.

        Parameters
        ----------
        smoothing : float, optional
            Smoothing scale of the innermost part of the profile (for numerical reasons).
            Default is 0.0001.
        exponent : float, optional
            Exponent parameter. Default is 2.
        radius_major_axis : bool, optional
            If True, defines the half-light radius of the Sersic light profile along
            the semi-major axis. If False, uses the product average
            of semi-major and semi-minor axis.
            Default is False.
        """
        self._radius_major_axis = radius_major_axis
        self._super = False if exponent == 2 else True
        self._e = float(exponent)
        self._s = float(smoothing)

    @staticmethod
    def b_n(n):
        """
        Compute the b_n parameter for Sérsic profiles.
        
        This function computes B(n), which is an approximation of the exact solution
        to the relation: 2*incomplete_gamma_function(2n; b_n) = Gamma_function(2*n).
        
        Parameters
        ----------
        n : float or array-like
            The Sérsic index.
        
        Returns
        -------
        float or ndarray
            The b_n parameter value(s), with a minimum floor of 1e-5 to ensure
            numerical stability.
        
        Notes
        -----
        The approximation used is: b_n = 1.9992 * n - 0.3271
        
        References
        ----------
        The b_n parameter is a key component in defining the normalized Sérsic
        light profile and relates to the concentration of light in the profile.
        """
        bn = 1.9992 * n - 0.3271
        return jnp.maximum(bn, 1e-5)

    def radial_distance(self, x, y, e1, e2, center_x, center_y):
        """
        Calculate the radial distance from the center of a Sersic profile.

        Accounts for orientation and axis ratio of the elliptical profile. Supports
        both standard elliptical and superelliptical distance calculations.

        Parameters
        ----------
        x : float or ndarray
            x-coordinate(s)
        y : float or ndarray
            y-coordinate(s)
        e1 : float
            First eccentricity component
        e2 : float
            Second eccentricity component
        center_x : float
            x-coordinate of the Sersic profile center
        center_y : float
            y-coordinate of the Sersic profile center

        Returns
        -------
        float or ndarray
            Radial distance from the center, computed according to the major axis
            convention and superelliptical parameters if enabled.

        Notes
        -----
        The calculation depends on internal flags:
        - `_radius_major_axis`: Determines which transformation method to use
        - `_super`: If True, uses superelliptical distance metric; otherwise uses
          standard elliptical distance
        - `_e`: Exponent parameter for superelliptical distance calculation
        """
        if self._radius_major_axis:
            phi_G, q = param_util.ellipticity2phi_q(e1, e2)
            x_shift = x - center_x
            y_shift = y - center_y
            cos_phi = jnp.cos(phi_G)
            sin_phi = jnp.sin(phi_G)
            xt1 = cos_phi * x_shift + sin_phi * y_shift
            xt2 = -sin_phi * x_shift + cos_phi * y_shift
            if not self._super:
                xt2difq2 = xt2 / (q * q)
                r = jnp.sqrt(xt1 * xt1 + xt2 * xt2difq2)
            else:
                r = jnp.power(jnp.power(jnp.abs(xt1), self._e) + jnp.power(jnp.abs(xt2/q), self._e), 1/self._e)
        else:
            x_, y_ = param_util.transform_e1e2_product_average(
                x, y, e1, e2, center_x, center_y
            )
            if not self._super:
                r = jnp.sqrt(x_**2 + y_**2)
            else:
                r = jnp.power(jnp.power(jnp.abs(x_), self._e) + jnp.power(jnp.abs(y_), self._e), 1/self._e)
        return r

    def _total_flux(self, I_eff, r_eff, n_sersic):
        """Compute the total flux of a round Sersic profile.

        Parameters
        ----------
        I_eff : float
            Surface brightness at the effective radius (in same units as r_eff)
        r_eff : float
            Projected half-light radius
        n_sersic : float
            Sersic index

        Returns
        -------
        float
            Integrated flux to infinity
        """
        bn = self.b_n(n_sersic)
        return (
            I_eff
            * r_eff**2
            * 2
            * jnp.pi
            * n_sersic
            * jnp.exp(bn)
            / bn ** (2 * n_sersic)
            * special.gamma(2 * n_sersic)
        )

    def total_flux(self, amp, R_sersic, n_sersic, e1, e2, **kwargs):
        """Compute the analytical integral of the total flux for a Sersic profile.

        Parameters
        ----------
        amp : float
            Amplitude parameter in Sersic function (surface brightness at R_sersic).
        R_sersic : float
            Half-light radius in semi-major axis.
        n_sersic : float
            Sersic index.
        e1 : float
            First eccentricity component.
        e2 : float
            Second eccentricity component.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            Analytic integral of the total flux of the Sersic profile.

        Notes
        -----
        If `_radius_major_axis` is True, the semi-major axis R_eff is converted to a
        product-averaged definition by accounting for the axis ratio q derived from
        eccentricity parameters e1 and e2. Otherwise, R_eff is set equal to R_sersic.
        """
        # compute product average half-light radius
        if self._radius_major_axis:
            _, q = param_util.ellipticity2phi_q(e1, e2)
            # translate semi-major axis R_eff into product averaged definition for circularization
            r_eff = R_sersic * jnp.sqrt(q)
        else:
            r_eff = R_sersic
        return self._total_flux(amp, r_eff, n_sersic)

    def _R_stable(self, R):
        """Floor R_ at self._s for numerical stability."""
        return jnp.maximum(self._s, R)

    def _r_sersic(
            self, R, R_sersic, n_sersic, max_R_frac=1000.0,
        ):
        """Evaluate the Sersic profile at radius R."""
        R_ = self._R_stable(R)
        R_sersic_ = self._R_stable(R_sersic)
        bn = self.b_n(n_sersic)
        R_frac = R_ / R_sersic_
        good_inds = (jnp.asarray(R_frac) <= max_R_frac).astype(int)
        result = good_inds * jnp.exp(-bn * (R_frac**(1. / n_sersic) - 1.))
        return jnp.nan_to_num(result)

    @property
    def num_amplitudes(self):
        return 1


class Sersic(SersicBase):
    """Class to evaluate an (elliptical) Sersic profile.

    .. math::

        I(R) = I_{\\rm e} \\exp \\left( -b_n \\left[(R/R_{\\rm Sersic})^{\\frac{1}{n}}-1\\right]\\right)

    with :math:`I_0 = amp`,
    :math:`R = \\sqrt{q \\theta^2_x + \\theta^2_y/q}`
    and
    with :math:`b_{n}\\approx 1.999n-0.327`

    Parameters
    ----------
    smoothing : float, optional
        Smoothing length to avoid numerical divergence at r==, by default 0.00001
    exponent : _type_, optional
        If different from 2, corresponds to the 'super-ellipse' profile, by default 2.
    radius_major_axis : bool, optional
        If True, all radius parameters are defined along the major axis
        otherwise they are defined along the intermediate axis (product average 
        of semi-major and semi-minor axis), by default True
    """
    param_names = ['amp', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'R_sersic': 0, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5,'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'R_sersic': 100, 'n_sersic': 8, 'e1': 0.5, 'e2': 0.5,'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def __init__(
            self, 
            smoothing=0.00001, 
            exponent=2., 
            radius_major_axis=True,
        ):
        super().__init__(smoothing=smoothing, exponent=exponent, radius_major_axis=radius_major_axis)

    def function(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0, max_R_frac=1000.0):
        """
        Compute the Sersic profile at given coordinates.

        Parameters
        ----------
        x : float or array_like
            X-coordinate(s).
        y : float or array_like
            Y-coordinate(s).
        amp : float
            Surface brightness/amplitude value at the half light radius.
        R_sersic : float
            Semi-major axis half light radius.
        n_sersic : float
            Sersic index.
        e1 : float
            Eccentricity parameter (first component).
        e2 : float
            Eccentricity parameter (second component).
        center_x : float, optional
            Center x-coordinate. Default is 0.
        center_y : float, optional
            Center y-coordinate. Default is 0.
        max_R_frac : float, optional
            Maximum window outside of which the profile is zeroed, in units of R_sersic.
            Default is 1000.0.

        Returns
        -------
        float or array_like
            Sersic profile value(s) at the given coordinate(s) (x, y).
        """
        R_sersic = jnp.maximum(0, R_sersic)
        R = self.radial_distance(x, y, e1, e2, center_x, center_y)
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result

    def derivatives(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x, center_y, max_R_frac=1000.0):
        """
        Compute partial derivatives of the Sersic profile.
        This method calculates the partial derivatives of the Sersic light profile
        with respect to spatial coordinates x and y using automatic differentiation.
        Parameters
        ----------
        x : float or ndarray
            x-coordinate(s).
        y : float or ndarray
            y-coordinate(s).
        amp : float
            Surface brightness/amplitude value at the half-light radius.
        R_sersic : float
            Semi-major axis half-light radius.
        n_sersic : float
            Sersic index.
        e1 : float
            Eccentricity parameter (first component).
        e2 : float
            Eccentricity parameter (second component).
        center_x : float
            Center x-coordinate.
        center_y : float
            Center y-coordinate.
        max_R_frac : float, optional
            Maximum window outside of which the profile is zeroed,
            in units of R_sersic. Default is 1000.0.
        Returns
        -------
        f_x : float or ndarray
            Partial derivative of the Sersic profile with respect to x.
        f_y : float or ndarray
            Partial derivative of the Sersic profile with respect to y.
        """
        def _function(p):
            return self.function(p[0], p[1], 
                                 R_sersic, n_sersic, e1, e2,
                                 center_x, center_y, amp,
                                 max_R_frac=max_R_frac)
        
        grad_function = grad(_function)

        @jit
        def _grad_function(x, y):
            return grad_function([x, y])[0], grad_function([x, y])[1]
        
        f_x, f_y = jnp.vectorize(_grad_function)(x, y)
        return f_x, f_y


class CoreSersic(SersicBase):
    """This class contains the Core-Sersic function introduced by e.g. Trujillo et al.
    2004.

    .. math::

        I(R) = I' \\left[1 + (R_break/R)^{\\alpha} \\right]^{\\gamma_in / \\alpha}
        \\exp \\left{ -b_n \\left[(R^{\\alpha} + R_break^{\\alpha})/R_e^{\\alpha}  \\right]^{1 / (n\\alpha)}  \\right}

    with

    .. math::
        I' = I_b 2^{-\\gamma_in/ \\alpha} \\exp \\left[b_n 2^{1 / (n\\alpha)} (R_break/R_e)^{1/n}  \\right]

    where :math:`I_b` is the intensity at the break radius and :math:`R = \\sqrt{q \\theta^2_x + \\theta^2_y/q}`.
    """

    param_names = ['amp', 'R_sersic', 'R_break', 'n_sersic', 'gamma_in', 'alpha', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'R_sersic': 0, 'R_break': 0, 'n_sersic': 0.5, 'gamma_in': 0, 'alpha': 1e-5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'R_sersic': 100, 'R_break': 90, 'n_sersic': 8, 'gamma_in': 10, 'alpha': 100, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def function(
            self,
            x,
            y,
            amp,
            R_sersic,
            n_sersic,
            e1,
            e2,
            R_break,
            gamma_in,
            alpha=3.,
            center_x=0,
            center_y=0,
            max_R_frac=1000.0,
        ):
        """Evaluates the 'core-Sersic' profile defined in Trujillo et al. 2004.

        Note that the approximation from b to b_n is used, see Trujillo et al. 2004 for more details.

        Parameters
        ----------
        x : float or ndarray
            x-coordinate
        y : float or ndarray
            y-coordinate
        amp : float
            Surface brightness/amplitude value at the half light radius
        R_sersic : float
            Half light radius (either semi-major axis or product average of semi-major and semi-minor axis)
        R_break : float
            Core radius
        n_sersic : float
            Sersic index
        gamma_in : float
            Inner power-law exponent
        e1 : float
            Eccentricity parameter e1
        e2 : float
            Eccentricity parameter e2
        center_x : float
            Center in x-coordinate
        center_y : float
            Center in y-coordinate
        alpha : float, optional
            Sharpness of the transition between the cusp and the outer Sersic profile.
            Default is 3.0.
        max_R_frac : float, optional
            Maximum window outside which the mass is zeroed, in units of R_sersic.
            Default is 1000.0.

        Returns
        -------
        float or jax.Array
            Cored Sersic profile value at (x, y)
        """
        # NOTE: max_R_frac not implemented
        R_ = self.radial_distance(x, y, e1, e2, center_x, center_y)
        R = self._R_stable(R_)
        bn = self.b_n(n_sersic)
        result = (
            amp
            * (1 + (R_break / R) ** alpha) ** (gamma_in / alpha)
            * jnp.exp(
                -bn
                * (
                    ((R**alpha + R_break**alpha) / R_sersic**alpha)
                    ** (1.0 / (alpha * n_sersic))
                    - 1.0
                )
            )
        )
        return result
    
    def total_flux(self, *args, **kwargs):
        print("Warning: Note that the CoreSersic.total_flux() simply calls the simpler "
              "case of a non-cored elliptical Sersic profile, namely it ignores "
              "the break radius, inner slope and transition strength parameters. "
              "This is probably a good approximation when the core radius is small compared to the half-light radius, but may not be accurate otherwise.")
        return super().total_flux(*args, **kwargs)
    