__author__ = 'aymgal'

import numpy as np


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

    def draw_samples(self, num_samples=10000, seed=None):
        if seed is not None:
            np.random.seed(seed)
        mean = self._param.current_values()
        cov  = self.covariance_matrix
        samples = np.random.multivariate_normal(mean, cov, size=num_samples)
        return samples

    def get_kwargs_sigma(self):
        sigma_values = np.sqrt(np.abs(np.diag(self.covariance_matrix)))
        return self._param.args2kwargs(sigma_values)

    def compute_fisher_information(self, recompute=False):
        if hasattr(self, '_fim') and not recompute:
            return  # nothing to do
        best_fit_values = self._param.current_values()
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

    def split_fisher_matrix(self, num_before, num_after):
        fim_interior = self.fisher_matrix[num_before:-num_after, num_before:-num_after]

        fim_block_UL = self.fisher_matrix[:num_before, :num_before]
        fim_block_UR = self.fisher_matrix[:num_before, -num_after:]
        fim_block_LR = self.fisher_matrix[-num_after:, -num_after:]
        fim_block_LL = self.fisher_matrix[-num_after:, :num_before]
        fim_exterior = np.block([[fim_block_UL, fim_block_UR], 
                                 [fim_block_LL, fim_block_LR]])

        return fim_interior, fim_exterior
