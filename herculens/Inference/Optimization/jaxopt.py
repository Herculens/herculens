# Handles different method to optimize a loss function
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal', 'austinpeel'


import time
import warnings
import numpy as np
from copy import deepcopy
import jax
import jaxopt

from herculens.Inference.Optimization.base_optim import BaseOptimizer


__all__ = ['JaxoptOptimizer']


class JaxoptOptimizer(BaseOptimizer):
    """Class that handles optimization tasks, i.e. finding best-fit point estimates of parameters
    It currently handles:
    - a subset of scipy.optimize.minimize routines, using first and second order derivatives when required
    - a particle swarm optimizer (PSO), implemented in lenstronomy
    """

    def __init__(self, jaxopt_method, *args, **kwargs):
        mod = __import__('jaxopt', fromlist=[jaxopt_method])
        self._solver_class = getattr(mod, jaxopt_method)
        self._jaxopt_method = jaxopt_method
        super().__init__(*args, **kwargs)

    def run(self, init_params, multi_start_from_prior=False, num_multi_start=1,
            progress_bar=True, return_param_history=False, **solver_kwargs):
        # TODO: should we call once / a few times all jitted functions before optimization, to potentially speed things up?
        metrics = MinimizeMetrics(self.func, with_param_history=return_param_history)
        if self._jaxopt_method == 'ScipyMinimize':
            solver = self._solver_class(fun=self.func_optim, jit=True, 
                                        callback=metrics, **solver_kwargs)
        else:
            solver = self._solver_class(self.func_optim, jit='auto', 
                                        callback=metrics, **solver_kwargs)

        if num_multi_start > 1: 
            raise NotImplementedError("Multi-start optimization to be implemented.")

        # @jax.jit
        def _run(init_params):
            metrics.reset()
            res = solver.run(init_params)
            return (res, self.func(res.params), 
                    metrics.get_loss_history())

        start = time.time()
        best_fit_list = []
        logL_best_fit_list = []
        loss_history_list = []
        # param_history_list = []
        extra_fields_list = []
        for n in self._for_loop(range(num_multi_start), progress_bar, 
                                total=num_multi_start, 
                                desc=f"jaxopt.{self._jaxopt_method}"):
            #init_params_n = init_samples[n, :]
            init_params_n = init_params
            res, loss, loss_hist = _run(init_params_n)
            if loss_hist == []:
                warnings.warn("The loss history does not contain any value")
            best_fit_list.append(res.params)
            logL_best_fit_list.append(-loss)
            loss_history_list.append(loss_hist)
            # param_history_list.append(param_hist)

        # select the best fit among the multi start runs
        if num_multi_start > 1:
            index = np.argmax(logL_best_fit_list)
        else:
            index = 0
        best_fit = best_fit_list[index]
        logL_best_fit = logL_best_fit_list[index]

        runtime = time.time() - start

        extra_fields = {}
        extra_fields['best_fit_index'] = index
        extra_fields['loss_history'] = loss_history_list[index]
        extra_fields['loss_history_list'] = loss_history_list
        if return_param_history is True:
            extra_fields['param_history'] = param_history_list[index]
            extra_fields['param_history_list'] = param_history_list  # maybe too memory consuming?

        return best_fit, logL_best_fit, extra_fields, runtime


class MinimizeMetrics(object):
    """simple callable class used as callback in scipy.optimize.minimize method"""
    
    def __init__(self, func, with_param_history=False):
        self._func = func
        self._call = self._call_1arg
        self._with_param_history = with_param_history
        self.reset()

    def reset(self):
        self.loss_history = []
        self.param_history = []

    def get_loss_history(self, copy=True):
        return deepcopy(self.loss_history) if copy else self.loss_history

    def get_param_history(self, copy=True):
        return deepcopy(self.param_history) if copy else self.param_history

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)
        
    def _call_1arg(self, x):
        self.loss_history.append(float(self._func(x)))
        if self._with_param_history:
            self.param_history.append(x)

    def _call_2args(self, x, state):
        # Input state parameter is necessary for 'trust-constr' method
        # You can use it to stop execution early by returning True
        self.loss_history.append(float(self._func(x)))
        if self._with_param_history:
            self.param_history.append(x)
