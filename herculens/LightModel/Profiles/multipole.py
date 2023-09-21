# Defines a multipole in the potential
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel.Profiles module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'lynevdv', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp

from herculens.MassModel.Profiles.multipole import Multipole as MassMultipole
import herculens.Util.param_util as param_util


__all__ = ['Multipole']


class Multipole(object):
    """
    This class contains a multipole contribution (for 1 component with m>=2)
    This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf
    m : int, multipole order, m>=2
    amp_m : float, multipole strength
    phi_m : float, multipole orientation in radian
    """
    param_names = ['m', 'amp_m', 'phi_m', 'center_x', 'center_y']
    lower_limit_default = {'m': 2,'amp_m':0, 'phi_m': -np.pi, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'m': 100,'amp_m': 100000, 'phi_m': np.pi, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def __init__(self):
        self._backend = MassMultipole()

    def function(self, x, y, m, amp_m, phi_m, center_x=0, center_y=0):
        """
        Lensing potential of multipole contribution (for 1 component with m>=2)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param m: int, multipole order, m>=2
        :param amp_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: x-position
        :return: lensing potential
        """
        return self._backend.function(x, y, m, amp_m, phi_m, center_x=center_x, center_y=center_y)

    def derivatives(self,x,y, m, amp_m, phi_m, center_x=0, center_y=0):
        """
        Deflection of a multipole contribution (for 1 component with m>=2)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param m: int, multipole order, m>=2
        :param amp_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: x-position
        :return: deflection angles alpha_x, alpha_y
        """
        return self._backend.derivatives(x, y, m, amp_m, phi_m, center_x=center_x, center_y=center_y)
