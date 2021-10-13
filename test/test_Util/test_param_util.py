import numpy as np
import pytest
import numpy.testing as npt

from herculens.Util import util
import herculens.Util.param_util as param_util


def test_phi_q2_ellipticity():
    phi, q = 0, 1
    e1,e2 = param_util.phi_q2_ellipticity(phi, q)
    assert e1 == 0
    assert e2 == 0

    phi, q = 1, 1
    e1,e2 = param_util.phi_q2_ellipticity(phi, q)
    assert e1 == 0
    assert e2 == 0

    phi, q = 2.,0.95
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    assert e1 == -0.016760092842656733
    assert e2 == -0.019405192187382792

    phi, q = 0, 0.9
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    assert e1 == 0.05263157894736841
    assert e2 == 0

def test_ellipticity2phi_q():
    e1, e2 = 0.3,0
    phi,q = param_util.ellipticity2phi_q(e1, e2)
    assert phi == 0
    assert q == 0.53846153846153844

    # Works on np arrays as well
    e1 = np.array([0.3, 0.9])
    e2 = np.array([0.0, 0.9 ])
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    assert np.allclose(phi, [0.0, 0.39269908], atol=1.e-08)
    assert np.allclose(q, [0.53846153, 5.00025001e-05], atol=1.e-08)

def test_ellipticity2phi_q_symmetry():
    phi,q = 1.5, 0.8
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    phi_new, q_new = param_util.ellipticity2phi_q(e1, e2)
    npt.assert_almost_equal(phi, phi_new, decimal=10)
    npt.assert_almost_equal(q, q_new, decimal=7)

    phi,q = -1.5, 0.8
    e1,e2 = param_util.phi_q2_ellipticity(phi, q)
    phi_new,q_new = param_util.ellipticity2phi_q(e1, e2)
    npt.assert_almost_equal(phi, phi_new, decimal=10)
    npt.assert_almost_equal(q, q_new, decimal=7)

    e1, e2 = 0.1, -0.1
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    e1_new, e2_new = param_util.phi_q2_ellipticity(phi, q)
    npt.assert_almost_equal(e1, e1_new, decimal=7)
    npt.assert_almost_equal(e2, e2_new, decimal=7)

    e1, e2 = 0.99, 0.0
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    e1_new, e2_new = param_util.phi_q2_ellipticity(phi, q)
    phi_new, q_new = param_util.ellipticity2phi_q(e1_new, e2_new)
    npt.assert_almost_equal(phi, phi_new, decimal=10)
    npt.assert_almost_equal(q, q_new, decimal=7)
    npt.assert_almost_equal(e1, e1_new, decimal=7)
    npt.assert_almost_equal(e2, e2_new, decimal=7)

def test_phi_gamma_ellipticity():
    phi = -1.
    gamma = 0.1
    e1, e2 = param_util.shear_polar2cartesian(phi, gamma)
    print(e1, e2, 'e1, e2')
    phi_out, gamma_out = param_util.shear_cartesian2polar(e1, e2)
    assert phi == phi_out
    assert gamma == gamma_out

def test_phi_gamma_ellipticity_2():
    e1, e2 = -0.04, -0.01
    phi, gamma = param_util.shear_cartesian2polar(e1, e2)

    e1_out, e2_out = param_util.shear_polar2cartesian(phi, gamma)
    npt.assert_almost_equal(e1, e1_out, decimal=7)
    npt.assert_almost_equal(e2, e2_out, decimal=7)


if __name__ == '__main__':
    pytest.main()
