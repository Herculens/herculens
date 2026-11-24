# Defines an elliptical lens mass profile whose projected convergence
# is that of a 3D broken power-law density (Du et al. 2020).
#
# Copyright (c) 2026, herculens developers and contributors

__author__ = 'aymgal' # partly using Claude Opus 5.5


import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from herculens.Util import util, param_util
from herculens.MassModel.Profiles.epl import EPL


__all__ = ['BPL3D']


class BPL3D(object):
    """Elliptical lens model built from a broken power-law 3D density.

    The 3D density follows Du et al. (2020, ApJ 892, 62; arXiv:1911.11761):

        rho(r) = rho_c (r / r_c)^(-gamma_in)   for r <= r_c
        rho(r) = rho_c (r / r_c)^(-gamma)      for r >= r_c

    with 0 <= gamma_in < 3 and 1 < gamma < 3. The profile is projected along
    the line of sight, and the ellipticity is then imposed on the projected
    convergence. The elliptical radius is R_el = sqrt(q x^2 + y^2 / q), which
    is the same convention as the EPL profile.

    The convergence is kappa = kappa_1 + kappa_2, where:

    - kappa_1 = (3 - gamma)/2 * (theta_E / R_el)^(gamma - 1) is exactly the
      EPL convergence. Its deflection is computed by the herculens EPL class.
    - kappa_2 is a compact (R_el < r_c) correction, given by the projection
      of rho_c [(r/r_c)^(-gamma_in) - (r/r_c)^(-gamma)] over r < r_c.

    With gamma_in = 0 the profile has a constant-density 3D core. With
    gamma_in = gamma it reduces exactly to the EPL.

    The deflection of kappa_2 is not computed from the series of Du et al.
    (their eq. 48 inside the core). Instead, the order of integration in the
    Bourassa & Kantowski (1975) integral (Du et al. eq. 44) is swapped. The
    inner integral over the projected radius is then elementary, and the
    correction reduces to a single 1D integral over the 3D radius
    r in [0, r_c]:

        alpha_2^*(z) = (4 rho_c / Sigma_crit) / z
            * [ int_0^{R'} delta(r) r^2 g(w r^2 / R_el^2) dr
              + int_{R'}^{r_c} delta(r) R_el^2 / 2 * J(R_el^2 / r^2, w) dr ]

    where R' = min(R_el, r_c), w = (1/q - q) R_el^2 / z^2,
    delta(r) = (r/r_c)^(-gamma_in) - (r/r_c)^(-gamma),
    g(w) = artanh(sqrt(w)) / sqrt(w), and
    J(p, w) = int_0^1 dt / sqrt((1 - p t)(1 - w t)), which has an elementary
    closed form.

    This expression is valid inside and outside the break radius, for any
    axis ratio. In particular it has no restriction to q >= sqrt(2)/2. The
    two integrals are evaluated with fixed nodes: tanh-sinh for the first one,
    which has algebraic endpoint singularities at r = 0, and Gauss-Legendre
    for the second one, after a variable change. The profile therefore stays
    differentiable everywhere. With the default node counts, the deflection
    angles agree with a brute-force scipy integration to better than 1e-10
    arcsec for q >= 0.55, and to about 1e-7 arcsec for q = 0.45 with
    gamma = 2.6.

    The normalisation theta_E is the "b" parameter of Du et al. It is the
    Einstein radius of the outer power law, i.e. of the EPL with the same
    theta_E and gamma. The actual Einstein radius differs when r_break is not
    small compared to theta_E; use the einstein_radius() method to get it.

    Parameters
    ----------
    num_quad_inner : int, optional
        Number of tanh-sinh nodes for the integral over [0, min(R_el, r_break)].
        By default 41.
    num_quad_outer : int, optional
        Number of Gauss-Legendre nodes for the integral over [R_el, r_break].
        This integral is non-zero only inside the break radius. By default 16.
    """
    param_names = ['theta_E', 'gamma', 'gamma_in', 'r_break', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'gamma': 1, 'gamma_in': 0, 'r_break': 0,
                           'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 10, 'gamma': 3, 'gamma_in': 3, 'r_break': 10,
                           'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def __init__(self, num_quad_inner=41, num_quad_outer=16):
        self._epl = EPL()
        # tanh-sinh (double-exponential) rule on (0, 1) for integral 1
        self._ts_x, self._ts_w, self._ts_tmin = _tanh_sinh_nodes(num_quad_inner)
        # Gauss-Legendre rule on (0, 1) for integral 2
        nodes, weights = np.polynomial.legendre.leggauss(num_quad_outer)
        self._gl_x = jnp.asarray(0.5 * (nodes + 1.))
        self._gl_w = jnp.asarray(0.5 * weights)

    def function(self, x, y, theta_E, gamma, gamma_in, r_break, e1, e2, center_x=0, center_y=0):
        """Lensing potential (not implemented)."""
        raise NotImplementedError("The lensing potential of the BPL3D profile is not implemented.")

    def derivatives(self, x, y, theta_E, gamma, gamma_in, r_break, e1, e2, center_x=0, center_y=0):
        """Deflection angles.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius of the outer power law ("b" in Du et al. 2020)
        :param gamma: outer 3D logarithmic density slope (1 < gamma < 3)
        :param gamma_in: inner 3D logarithmic density slope (0 <= gamma_in < 3)
        :param r_break: break radius of the 3D density
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        x, y = self._shift_center(x, y, center_x, center_y, r_break)
        f_x_1, f_y_1 = self._epl.derivatives(x, y, theta_E, e1, e2, gamma,
                                             center_x=center_x, center_y=center_y)
        f_x_2, f_y_2 = self._break_derivatives(x, y, theta_E, gamma, gamma_in, r_break,
                                               e1, e2, center_x, center_y)
        return f_x_1 + f_x_2, f_y_1 + f_y_2

    def hessian(self, x, y, theta_E, gamma, gamma_in, r_break, e1, e2, center_x=0, center_y=0):
        """Hessian of the lensing potential.

        The EPL part is analytic. The break correction is differentiated with
        jax.jacfwd.

        :return: f_xx, f_yy, f_xy
        """
        x, y = self._shift_center(x, y, center_x, center_y, r_break)
        f_xx_1, f_yy_1, f_xy_1 = self._epl.hessian(x, y, theta_E, e1, e2, gamma,
                                                   center_x=center_x, center_y=center_y)

        def alpha2(x_, y_):
            return jnp.stack(self._break_derivatives(x_, y_, theta_E, gamma, gamma_in, r_break,
                                                     e1, e2, center_x, center_y))

        jac = jnp.vectorize(jax.jacfwd(alpha2, argnums=(0, 1)),
                            signature='(),()->(2),(2)')
        d_dx, d_dy = jac(jnp.asarray(x, dtype=float), jnp.asarray(y, dtype=float))
        f_xx_2 = d_dx[..., 0]
        f_yy_2 = d_dy[..., 1]
        f_xy_2 = 0.5 * (d_dy[..., 0] + d_dx[..., 1])
        return f_xx_1 + f_xx_2, f_yy_1 + f_yy_2, f_xy_1 + f_xy_2

    def convergence(self, x, y, theta_E, gamma, gamma_in, r_break, e1, e2, center_x=0, center_y=0):
        """Analytic convergence (Du et al. 2020, eqs. 36-37), for testing and plotting.

        This uses scipy's hypergeometric function, so it is neither
        jit-compatible nor differentiable.
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_, y_ = util.rotate(np.asarray(x) - center_x, np.asarray(y) - center_y, phi_G)
        R = np.sqrt(q * x_**2 + y_**2 / q)
        return self.convergence_radial(R, theta_E, gamma, gamma_in, r_break)

    @staticmethod
    def convergence_radial(R, theta_E, gamma, gamma_in, r_break):
        """Analytic convergence as a function of the elliptical radius
        (Du et al. 2020, eqs. 36-37). Not jit-compatible (uses scipy)."""
        from scipy.special import hyp2f1, gamma as gamma_fn
        R = np.asarray(R, dtype=float)
        B = np.sqrt(np.pi) * gamma_fn((gamma - 1) / 2.) / gamma_fn(gamma / 2.)
        kappa_1 = (3. - gamma) / 2. * (theta_E / R)**(gamma - 1)
        zt2 = np.clip(1. - R**2 / r_break**2, 0., None)
        zt = np.sqrt(zt2)
        kappa_2 = - (3. - gamma) / B * (theta_E / r_break)**(gamma - 1) * zt * (
            hyp2f1(gamma / 2., 1., 1.5, zt2) - hyp2f1(gamma_in / 2., 1., 1.5, zt2))
        return kappa_1 + np.where(R < r_break, kappa_2, 0.)

    def mean_convergence(self, R, theta_E, gamma, gamma_in, r_break):
        """Mean convergence within the elliptical radius R.

        It does not depend on the axis ratio, because the convergence is
        constant on similar ellipses.
        """
        R = jnp.asarray(R, dtype=float)
        # circular case: alpha(R) = R * mean_kappa(R), evaluated along the x-axis
        a_1 = (theta_E / R)**(gamma - 1) * R
        a_2, _ = self._break_major_axis(R, jnp.zeros_like(R), theta_E, gamma, gamma_in, r_break, 1.)
        return (a_1 + a_2) / R

    def einstein_radius(self, theta_E, gamma, gamma_in, r_break, num_iter=60):
        """Effective Einstein radius, defined by mean_convergence(theta_E_eff) = 1.

        It is found by bisection in log-radius. It is jit-compatible, but it
        is not meant to be differentiated. NaN is returned if the lens is
        sub-critical, i.e. if the mean convergence is below 1 at all radii,
        which can happen when r_break is large compared to theta_E.
        """
        lo = jnp.log(1e-4 * theta_E)
        hi = jnp.log(1e2 * theta_E)

        def body(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            above = self.mean_convergence(jnp.exp(mid), theta_E, gamma, gamma_in, r_break) > 1.
            return jnp.where(above, mid, lo), jnp.where(above, hi, mid)

        lo, hi = jax.lax.fori_loop(0, num_iter, body, (lo, hi))
        theta_eff = jnp.exp(0.5 * (lo + hi))
        converged = jnp.abs(self.mean_convergence(theta_eff, theta_E, gamma, gamma_in, r_break) - 1.) < 1e-8
        return jnp.where(converged, theta_eff, jnp.nan)

    @staticmethod
    def _shift_center(x, y, center_x, center_y, r_break):
        """Move points that lie exactly at the centre by a tiny offset.

        The EPL and the break correction are each singular there, but their
        sum is not (for gamma_in < 2).
        """
        x = jnp.asarray(x, dtype=float)
        y = jnp.asarray(y, dtype=float)
        eps = 1e-6 * r_break
        at_center = ((x - center_x)**2 + (y - center_y)**2) < eps**2
        return jnp.where(at_center, center_x + eps, x), y

    def _break_derivatives(self, x, y, theta_E, gamma, gamma_in, r_break, e1, e2, center_x, center_y):
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi_G)
        f__x, f__y = self._break_major_axis(x_, y_, theta_E, gamma, gamma_in, r_break, q)
        return util.rotate(f__x, f__y, -phi_G)

    def _break_major_axis(self, x, y, b, gamma, gamma_in, r_c, q):
        """Deflection of the compact break correction kappa_2, in the frame
        where the major axis is along x."""
        x = jnp.asarray(x, dtype=float)
        y = jnp.asarray(y, dtype=float)
        # double-where trick to avoid NaNs (and NaN gradients) at the centre,
        # where the deflection of kappa_2 vanishes by symmetry
        eps = 1e-9 * r_c
        at_center = (x**2 + y**2) < eps**2
        x_s = jnp.where(at_center, eps, x)
        y_s = jnp.where(at_center, 0., y)

        z = x_s + 1j * y_s
        R_el2 = q * x_s**2 + y_s**2 / q
        R_el = jnp.sqrt(R_el2)
        w = (1. / q - q) * R_el2 / z**2 + 0j
        R_p = jnp.minimum(R_el, r_c)

        inside = R_el < r_c   # integral 2 is non-zero only inside the break radius
        # square roots needed below, computed once per pixel. The functions
        # used are even in sqrt(w), so the choice of branch does not matter.
        sw = jnp.sqrt(w)
        s1w = jnp.sqrt(1. - w)

        # add a trailing axis for the quadrature nodes
        R_el2_, sw_, s1w_, R_p_ = R_el2[..., None], sw[..., None], s1w[..., None], R_p[..., None]

        # integral 1, over [0, R'], with r = R' t and a tanh-sinh rule in t.
        # The integrand behaves as t^(2 - gamma) near t = 0, which the
        # double-exponential rule integrates with exponential convergence.
        # Note that sqrt(w r^2 / R_el^2) = sqrt(w) R' t / R_el.
        t, wts = self._ts_x, self._ts_w
        delta_1 = (R_p_ / r_c)**(-gamma_in) * t**(2. - gamma_in) - (R_p_ / r_c)**(-gamma) * t**(2. - gamma)
        integrand_1 = delta_1 * _artanh_ratio(sw_ * (R_p / R_el)[..., None] * t)
        I_1 = jnp.sum(wts * integrand_1, axis=-1)
        # analytic contribution of [0, t_min], not covered by the nodes, where
        # the artanh ratio is 1 to O(t_min^2). It matters for gamma -> 3.
        t_min = self._ts_tmin
        I_1 = I_1 + ((R_p / r_c)**(-gamma_in) * t_min**(3. - gamma_in) / (3. - gamma_in)
                     - (R_p / r_c)**(-gamma) * t_min**(3. - gamma) / (3. - gamma))
        I_1 = R_p**3 * I_1

        # integral 2, over [R', r_c]. Use r = R' (r_c / R')^(v^2), so that the
        # sqrt behaviour at r = R_el and the steep power laws near the centre
        # become smooth in v. For points outside the break radius, safe values
        # are used and the result is masked, to avoid NaN gradients.
        v, wts = self._gl_x, self._gl_w
        R_p_2 = jnp.where(inside, R_p, 0.5 * r_c)[..., None]
        R_el2_2 = jnp.where(inside, R_el2, 0.25 * r_c**2)[..., None]
        log_ratio = jnp.log(r_c / R_p_2)   # L >= 0
        # with R' = R_el inside the break radius: r / r_c = exp(-L (1 - v^2))
        # and p = R_el^2 / r^2 = exp(-2 L v^2). 1 - p is computed with expm1,
        # so p never exceeds 1 because of rounding.
        log_r_rc = - log_ratio * (1. - v**2)
        r2 = r_c * jnp.exp(log_r_rc)
        delta_2 = jnp.exp(-gamma_in * log_r_rc) - jnp.exp(-gamma * log_r_rc)
        p = jnp.exp(-2. * log_ratio * v**2)
        one_minus_p = -jnp.expm1(-2. * log_ratio * v**2)
        integrand_2 = delta_2 * R_el2_2 * _J(p, one_minus_p, sw_, s1w_) * r2 * v * log_ratio
        I_2 = jnp.where(inside, jnp.sum(wts * integrand_2, axis=-1), 0.)

        # 4 rho_c / Sigma_crit, from Du et al. (2020) eq. 20
        log_B = 0.5 * jnp.log(jnp.pi) + gammaln((gamma - 1.) / 2.) - gammaln(gamma / 2.)
        prefac = 2. * (3. - gamma) * b**(gamma - 1.) * r_c**(-gamma) * jnp.exp(-log_B)
        alpha_conj = prefac * (I_1 + I_2) / z   # = alpha_x - i alpha_y

        alpha_x = jnp.where(at_center, 0., alpha_conj.real)
        alpha_y = jnp.where(at_center, 0., -alpha_conj.imag)
        return alpha_x, alpha_y


def _artanh_ratio(s):
    """artanh(s) / s for complex s, which is g(w) = artanh(sqrt(w)) / sqrt(w)
    evaluated with s = sqrt(w). It is even in s, and valid for s^2 not in
    [1, inf). It uses artanh(s) = Log((1 + s) / (1 - s)) / 2, which is the
    principal branch on the whole cut plane, so only one complex log is needed."""
    small = jnp.abs(s) < 1e-4
    s_safe = jnp.where(small, 0.5, s)
    s2 = s * s
    return jnp.where(small, 1. + s2 / 3. + s2 * s2 / 5.,
                     0.5 * jnp.log((1. + s_safe) / (1. - s_safe)) / s_safe)


def _J(p, one_minus_p, sw, s1w):
    """J(p, w) = int_0^1 dt / sqrt((1 - p t)(1 - w t)), for real 0 < p <= 1
    and complex w not in [1, inf). The arguments are p, 1 - p,
    sw = sqrt(w) and s1w = sqrt(1 - w).

    A closed form is J = 2 / sqrt(p w) * ln[(sqrt(w) + sqrt(p)) /
    (sqrt(w (1 - p)) + sqrt(p (1 - w)))]. It is rewritten here as
    J = 4 S / AB * artanh(X) / X, with X = sqrt(p w) S / AB, which is
    numerically stable for small p and w. Re(AB) > 0 for principal square
    roots, so there is no division by zero.
    """
    sp = jnp.sqrt(p)
    s1p = jnp.sqrt(one_minus_p)
    S = sp / (1. + s1p) + sw / (1. + s1w)
    AB = sw * (1. + s1p) + sp * (1. + s1w)
    S_AB = S / AB
    return 4. * S_AB * _artanh_ratio(sp * sw * S_AB)


def _tanh_sinh_nodes(num, t_max=3.2):
    """Tanh-sinh (double-exponential) nodes and weights on (0, 1).

    The nodes x = 1 / (1 + exp(-pi sinh(t))) accumulate double-exponentially
    towards both endpoints, which handles algebraic endpoint singularities.
    Also returns the lower end of the range covered by the nodes, x_min,
    where the half-node-spacing beyond the first node is included.
    """
    k = np.arange(num) - (num - 1) / 2.
    h = 2. * t_max / (num - 1)
    t = k * h
    u = np.pi * np.sinh(t)
    x = 1. / (1. + np.exp(-u))
    w = h * np.pi * np.cosh(t) * x * (1. / (1. + np.exp(u)))
    x_min = 1. / (1. + np.exp(np.pi * np.sinh(t_max + h / 2.)))
    return jnp.asarray(x), jnp.asarray(w), float(x_min)
