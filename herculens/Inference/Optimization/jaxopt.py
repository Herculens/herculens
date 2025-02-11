# Handles different method to optimize a loss function
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal', 'austinpeel'


import time
from copy import deepcopy
import jax
import jaxopt

from herculens.Inference.Optimization.base_optim import BaseOptimizer


__all__ = ['JaxoptOptimizer']


class JaxoptOptimizer(BaseOptimizer):
    """Wrapper to jaxopt's unconstrained optimizers"""

    def run_scipy(self, init_params, progress_bar=True, 
                  return_param_history=False, **solver_kwargs):
        # TODO: should we call once / a few times all jitted functions before optimization, to potentially speed things up?
        metrics = MinimizeMetrics(self.loss.function, with_param_history=return_param_history)
        solver = jaxopt.ScipyMinimize(fun=self.function_optim, jit=True, 
                                      callback=metrics, **solver_kwargs)

        init_params_ = deepcopy(init_params)
        start = time.time()
        # runs the optimizer
        res = solver.run(init_params_)
        # retrieve optimized parameters and loss value
        best_fit = res.params
        logL_best_fit = - self.loss.function(best_fit)
        runtime = time.time() - start

        extra_fields = {}
        extra_fields['loss_history'] = metrics.get_loss_history()
        if return_param_history is True:
            extra_fields['param_history'] = metrics.get_param_history()
        return best_fit, logL_best_fit, extra_fields, runtime

    def run(self, init_params, method='BFGS', progress_bar=False, **solver_kwargs):
        if method == 'BFGS':
            solver = jaxopt.BFGS(self.function_optim, value_and_grad=False, 
                                 **solver_kwargs)
        elif method == 'LBFGS':
            solver = jaxopt.LBFGS(self.function_optim, value_and_grad=False, 
                                  **solver_kwargs)
        elif method == 'LM':
            solver = jaxopt.LevenbergMarquardt(self.function_optim_LM_scalar, 
                                               **solver_kwargs)
        else:
            raise NotImplementedError

        # Defines and jits a single solver update
        @jax.jit
        def step(params_state, _):
            params, state = params_state
            params, state = solver.update(params, state)
            loss_val = self.loss.function(params)
            return (params, state), loss_val

        # Initialise optimizer state
        params = deepcopy(init_params)
        state = solver.init_state(params)

        # Gradient descent loop
        maxiter = solver_kwargs.pop('maxiter')

        start_time = time.time()
        if progress_bar:
            # param_history = []
            loss_history = []
            for i in self._for_loop(range(maxiter), progress_bar, 
                                    total=maxiter, 
                                    desc=f"jaxopt.{method}"):
                (params, state), loss_val = step((params, state), None)
                loss_history.append(loss_val)
                # if return_param_history is True:
                #     param_history.append(params)
        else:
            (params, state), loss_history = jax.lax.scan(step, (params, state), None, length=maxiter)
        runtime = time.time() - start_time

        # start_time = time.time()
        # init_params_ = deepcopy(init_params)
        # params, state = solver.run(init_params_)
        # runtime = time.time() - start_time

        best_fit = params
        logL_best_fit = - self.loss.function(best_fit)
        extra_fields = {'loss_history': loss_history}
        # if return_param_history is True:
        #     extra_fields['param_history'] = param_history
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
