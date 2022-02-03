__author__ = 'aymgal'

import numpy as np


class FisherCovariance(object):

    def __init__(self, parameter_class, inference_class):
        self.param = parameter_class  # TODO: this should hold the indices for splitting matrices
        self.inference = inference_class

    @property
    def full_covar_matrix(self):
        return np.linalg.inv(self._fim)

    def fisher2covar(self, fisher_matrix, inversion='full'):
        if inversion == 'full':
            return np.linalg.inv(fisher_matrix)
        elif inversion == 'diag':
            return 1. / np.diag(fisher_matrix)
        else:
            raise ValueError("Only 'full' and 'diag' options are supported for inverting the FIM.")

    @property
    def fisher_matrix(self):
        if not hasattr(self, '_fim'):
            raise ValueError("The FIM has not been computed.")
        return self._fim

    def compute_fisher_information(self, best_fit_values, recompute=False):
        if not recompute and hasattr(self, '_fim'):
            pass
        self._fim = self.inference.hessian(best_fit_values).block_until_ready()
        self._fim = np.array(self._fim)

    def split_fisher_matrix(self, num_before, num_after):
        fim_interior = self.fisher_matrix[num_before:-num_after, num_before:-num_after]

        fim_block_UL = self.fisher_matrix[:num_before, :num_before]
        fim_block_UR = self.fisher_matrix[:num_before, -num_after:]
        fim_block_LR = self.fisher_matrix[-num_after:, -num_after:]
        fim_block_LL = self.fisher_matrix[-num_after:, :num_before]
        fim_exterior = np.block([[fim_block_UL, fim_block_UR], 
                                 [fim_block_LL, fim_block_LR]])

        return fim_interior, fim_exterior
