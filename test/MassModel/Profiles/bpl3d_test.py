# This file tests the BPL3D mass profile (3D broken power law of Du et al. 2020)
# against the EPL, a brute-force (scipy) reference, and its own analytic convergence.

import pytest
import numpy as np
import numpy.testing as npt
from scipy import integrate, special

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from herculens.MassModel.Profiles.bpl3d import BPL3D
from herculens.MassModel.Profiles.epl import EPL
from herculens.Util import param_util


def _reference_deflection(x, y, theta_E, gamma, gamma_in, r_break, q, phi):
    """Bourassa & Kantowski (1975) integral of the convergence obtained by direct
    Abel projection of the 3D density, with scipy.quad."""
    B = np.sqrt(np.pi) * special.gamma((gamma - 1) / 2) / special.gamma(gamma / 2)
    rho_c = theta_E**(gamma - 1) * (3 - gamma) / (2 * B * r_break**gamma)

    def kappa(R):
        k1 = (3 - gamma) / 2 * (theta_E / R)**(gamma - 1)
        if R >= r_break:
            return k1
        d = lambda l: (np.hypot(R, l) / r_break)**(-gamma_in) - (np.hypot(R, l) / r_break)**(-gamma)
        return k1 + 2 * rho_c * integrate.quad(d, 0, np.sqrt(r_break**2 - R**2), epsabs=1e-14, epsrel=1e-13)[0]

    c, s = np.cos(phi), np.sin(phi)
    xr, yr = c * x + s * y, -s * x + c * y
    z = xr + 1j * yr
    R_el = np.sqrt(q * xr**2 + yr**2 / q)
    zeta2 = (1. / q - q) / z**2
    f = lambda R: kappa(R) * R / np.sqrt(1. - zeta2 * R**2)
    edges = [0., min(R_el, r_break)] + ([R_el] if R_el > r_break else [])
    I = 0j
    for lo, hi in zip(edges[:-1], edges[1:]):
        I += integrate.quad(lambda R: f(R).real, lo, hi, limit=300, epsabs=1e-13, epsrel=1e-12)[0]
        I += 1j * integrate.quad(lambda R: f(R).imag, lo, hi, limit=300, epsabs=1e-13, epsrel=1e-12)[0]
    a = np.conj(2. / z * I)
    return c * a.real - s * a.imag, s * a.real + c * a.imag


class TestBPL3D(object):

    def setup_method(self):
        self.profile = BPL3D()
        rng = np.random.default_rng(0)
        pts = np.concatenate([rng.uniform(-3., 3., (6, 2)), rng.uniform(-0.4, 0.4, (6, 2))])
        self.x, self.y = pts[:, 0], pts[:, 1]

    def test_epl_limit(self):
        e1, e2 = param_util.phi_q2_ellipticity(0.4, 0.6)
        for gamma in [1.6, 2.0, 2.4]:
            a = self.profile.derivatives(self.x, self.y, theta_E=1.5, gamma=gamma, gamma_in=gamma,
                                         r_break=0.4, e1=e1, e2=e2)
            b = EPL().derivatives(self.x, self.y, theta_E=1.5, gamma=gamma, e1=e1, e2=e2)
            npt.assert_allclose(a, b, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("gamma, gamma_in, r_break, q, phi", [
        (2.0, 0.0, 0.2, 0.6, 0.3),
        (2.4, 1.5, 0.05, 0.95, 0.0),
        (1.8, 2.4, 0.3, 0.7, 0.7),
        (2.6, 0.0, 0.4, 0.45, -0.5),
    ])
    def test_deflection_vs_reference(self, gamma, gamma_in, r_break, q, phi):
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        ax, ay = self.profile.derivatives(self.x, self.y, theta_E=1.5, gamma=gamma, gamma_in=gamma_in,
                                          r_break=r_break, e1=e1, e2=e2)
        ref = np.array([_reference_deflection(x, y, 1.5, gamma, gamma_in, r_break, q, phi)
                        for x, y in zip(self.x, self.y)])
        npt.assert_allclose(ax, ref[:, 0], rtol=0, atol=1e-6)
        npt.assert_allclose(ay, ref[:, 1], rtol=0, atol=1e-6)

    def test_hessian_trace(self):
        kw = dict(theta_E=1.5, gamma=2.2, gamma_in=0., r_break=0.3)
        e1, e2 = param_util.phi_q2_ellipticity(0.3, 0.7)
        f_xx, f_yy, _ = self.profile.hessian(self.x, self.y, e1=e1, e2=e2, **kw)
        kappa = self.profile.convergence(self.x, self.y, e1=e1, e2=e2, **kw)
        npt.assert_allclose(0.5 * (np.asarray(f_xx) + np.asarray(f_yy)), kappa, rtol=1e-9)

    def test_centre_and_gradients(self):
        names = ['theta_E', 'gamma', 'gamma_in', 'r_break', 'e1', 'e2', 'center_x', 'center_y']
        x = jnp.array([0., 1e-12, 0.3 / np.sqrt(0.7), 1.2])
        y = jnp.zeros_like(x)
        p0 = jnp.array([1.5, 2.2, 0., 0.3, 0.1, 0.05, 0., 0.])
        defl = self.profile.derivatives(x, y, **dict(zip(names, p0)))
        assert np.all(np.isfinite(np.asarray(defl)))
        loss = lambda p: jnp.sum(jnp.stack(self.profile.derivatives(x, y, **dict(zip(names, p))))**2)
        assert np.all(np.isfinite(np.asarray(jax.grad(loss)(p0))))

    def test_einstein_radius(self):
        th = self.profile.einstein_radius(1.5, 2.2, 0., 0.3)
        npt.assert_allclose(self.profile.mean_convergence(th, 1.5, 2.2, 0., 0.3), 1., atol=1e-8)
        assert np.isnan(self.profile.einstein_radius(1.5, 2.2, 0., 1.2))  # sub-critical lens
