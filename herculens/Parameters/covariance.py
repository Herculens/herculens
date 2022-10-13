# Estimation of the parameter covariance matrix via the Fisher information
# 
# Copyright (c) 2022, herculens developers and contributors

__author__ = 'aymgal'


import numpy as np
from herculens.Util import model_util


__all__ = ['FisherCovariance']


class FisherCovariance(object):

    def __init__(self, parameter_class, differentiable_class):
        self._param = parameter_class  # TODO: this should hold the indices for splitting matrices
        self._diff = differentiable_class

    @property
    def fisher_matrix(self):
        if not hasattr(self, '_fim'):
            raise ValueError("Call first compute_fisher_information().")
        return self._fim

    @property
    def covariance_matrix(self):
        if not hasattr(self, '_cov'):
            self._cov = self.fisher2covar(self.fisher_matrix, inversion='full')
        return self._cov

    def model_covariance(self, lens_image, num_mc_samples=10000, seed=None,
                         return_cross_covariance=False):
        samples = self.draw_samples(num_samples=num_mc_samples, seed=seed)
        return model_util.estimate_model_covariance(lens_image, self._param, samples,
                                                    return_cross_covariance=return_cross_covariance)

    def draw_samples(self, num_samples=10000, seed=None):
        return model_util.draw_samples_from_covariance(self._param.best_fit_values(),
                                                       self.covariance_matrix,
                                                       num_samples=num_samples, seed=seed)

    def get_kwargs_sigma(self):
        sigma_values = np.sqrt(np.abs(np.diag(self.covariance_matrix)))
        return self._param.args2kwargs(sigma_values)

    def compute_fisher_information(self, recompute=False):
        if hasattr(self, '_fim') and not recompute:
            return  # nothing to do
        best_fit_values = self._param.best_fit_values()
        self._fim = self._diff.hessian(best_fit_values).block_until_ready()
        self._fim = np.array(self._fim)
        if hasattr(self, '_cov'):
            delattr(self, '_cov')

    def fisher2covar(self, fisher_matrix, inversion='full'):
        if inversion == 'full':
            return np.linalg.inv(fisher_matrix)
        elif inversion == 'diag':
            return 1. / np.diag(fisher_matrix)
        else:
            raise ValueError("Only 'full' and 'diag' options are supported for inverting the FIM.")

    @staticmethod
    def split_matrix(matrix, num_before, num_after):
        interior = matrix[num_before:-num_after, num_before:-num_after]

        block_UL = matrix[:num_before, :num_before]
        block_UR = matrix[:num_before, -num_after:]
        block_LR = matrix[-num_after:, -num_after:]
        block_LL = matrix[-num_after:, :num_before]
        exterior = np.block([[block_UL, block_UR], 
                            [block_LL, block_LR]])

        return interior, exterior
