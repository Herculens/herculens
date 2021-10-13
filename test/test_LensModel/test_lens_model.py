import numpy as np
import numpy.testing as npt
import pytest
from herculens.LensModel.lens_model import LensModel
import unittest


class TestLensModel(object):
    """
    tests the source model routines
    """
    def test_init(self):
        lens_model_list = ['NIE', 'SIE', 'SHEAR', 'SHEAR_GAMMA_PSI', 'PIXELATED']
        lensModel = LensModel(lens_model_list)
        assert len(lensModel.lens_model_list) == len(lens_model_list)
        
    def test_gamma(self):
        lensModel = LensModel(lens_model_list=['SHEAR'])
        gamma1, gamm2  = 0.1, -0.1
        kwargs = [{'gamma1': gamma1, 'gamma2': gamm2}]
        e1_out, e2_out = lensModel.gamma(x=1., y=1., kwargs=kwargs)
        assert e1_out == gamma1
        assert e2_out == gamm2


if __name__ == '__main__':
    pytest.main()
