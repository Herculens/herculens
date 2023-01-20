# Defines the model of a strong lens
# 
# Copyright (c) 2022, herculens developers and contributors
# based on the ImSim module from lenstronomy (version 1.9.3)

__author__ = 'aymgal'


__all__ = ['ProbabilisticModel']


class BaseProbModel(object):
    """Base class for probabilistic model"""

    @property
    def num_parameters(self):
        sample = self.draw_samples(1, seed=0)
        return len(sample)
    
    def params2kwargs(self, params):
        raise NotImplementedError("`params2kwargs` method must be implemented.")

    def log_prob(self, params):
        raise NotImplementedError("`log_prob` method must be implemented.")
    
    def log_likelihood(self, params):
        raise NotImplementedError("`log_likelihood` method must be implemented.")

    def draw_samples(self, num_samples, seed=0):
        raise NotImplementedError("`draw_samples` method must be implemented.")
