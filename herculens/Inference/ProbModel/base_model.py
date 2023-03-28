# Defines the model of a strong lens
# 
# Copyright (c) 2022, herculens developers and contributors
# based on the ImSim module from lenstronomy (version 1.9.3)

__author__ = 'aymgal'


__all__ = ['ProbabilisticModel']


class BaseProbModel(object):
    """Base class for probabilistic model"""

    def model(self):
        raise NotImplementedError("Must be implemented by user class")
        
    def params2kwargs(self, params):
        raise NotImplementedError("`params2kwargs` method must be implemented.")

    def log_prob(self, params):
        raise NotImplementedError("`log_prob` method must be implemented.")
    
    def sample_prior(self, num_samples, seed=0):
        raise NotImplementedError("`draw_samples` method must be implemented.")
