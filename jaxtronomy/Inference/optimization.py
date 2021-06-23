import time
from scipy import optimize
from jax import jit
import numpy as np

from jaxtronomy.Inference.inference_base import InferenceBase

__all__ = ['Optimizer']


class Optimizer(InferenceBase):
    """Class that handles optimization tasks, i.e. finding best-fit point estimates of parameters
    It currently handles:
    - a subset of scipy.optimize.minimize routines, using first and second order derivatives when required
    - a particle swarm optimizer (PSO), implemented in lenstronomy
    """

    def __init__(self, loss_fn, param_class):
        super().__init__(loss_fn, param_class)

    @property
    def loss_history(self):
        if not hasattr(self, '_metrics'):
            raise ValueError("You must run the optimizer at least once to access the history")
        return self._metrics.loss_history

    @property
    def param_history(self):
        if not hasattr(self, '_metrics'):
            raise ValueError("You must run the optimizer at least once to access the history")
        return self._metrics.param_history

    def minimize(self, method='BFGS', restart_from_init=False, always_use_hessian=False):
        # TODO: should we call once / a few times all jitted functions, to potentially speed things up?
        init_params = self._param_class.initial_values(as_kwargs=False, original=restart_from_init)
        self._metrics = Metrics(self.loss)
        start = time.time()
        best_fit, extra_fields = self._run_scipy_minimizer(init_params, method, self._metrics,
                                                           always_use_hessian)
        runtime = time.time() - start
        logL = - float(self._metrics.loss_history[-1])
        self._param_class.set_best_fit(best_fit)
        return best_fit, logL, extra_fields, runtime

    def _run_scipy_minimizer(self, x0, method, callback, always_use_hessian):
        if method in ['Nelder-Mead']:  # TODO: add methods
            extra_kwargs = {}
        elif method in ['BFGS']:
            extra_kwargs = {'jac': self.jacobian}
        elif method in ['Newton-CG', 'trust-krylov', 'trust-exact']:
            if method in ['trust-exact'] or always_use_hessian:
                extra_kwargs = {'jac': self.jacobian, 'hess': self.hessian}
            else:
                extra_kwargs = {'jac': self.jacobian, 'hessp': self.hessian_vec_prod}
        opt = optimize.minimize(self.loss, x0, method=method, 
                                callback=callback, **extra_kwargs)
        extra_fields = {'jac': None, 'hess': None, 'hess_inv': None}
        for key in extra_fields:
            if hasattr(opt, key):
                extra_fields[key] = getattr(opt, key)
        return opt.x, extra_fields

    def pso(self, n_particles=100, n_iterations=100, restart_from_init=False, n_threads=1):
        from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
        from lenstronomy.Sampling.Pool.pool import choose_pool
        pool = choose_pool(mpi=False, processes=n_threads, use_dill=True)
        lowers, uppers = self._param_class.bounds
        if np.any(np.isnan(lowers)):
            raise ValueError("PSO needs lower and upper bounds, i.e. prior distributions with a finite support")
        optimizer = ParticleSwarmOptimizer(self.log_likelihood(x), 
                                           lowers, uppers, n_particles, pool=pool)
        init_params = self._param_class.initial_values(as_kwargs=False, original=restart_from_init)
        optimizer.set_global_best(init_params, [0]*len(init_params), - self.loss(init_params))
        start = time.time()
        best_fit, [chi2_list, pos_list, vel_list] = optimizer.optimize(n_iterations)
        runtime = time.time() - start
        logL = float(chi2_list[-1])
        extra_fields = {'chi2_list': chi2_list, 'pos_list': pos_list, 'vel_list': vel_list}
        self._param_class.set_best_fit(best_fit)
        return best_fit, logL, extra_fields, runtime


class Metrics(object):
    def __init__(self, func):
        self.loss_history = []
        self.param_history = []
        self._func = func
        
    def __call__(self, v):
        self.loss_history.append(self._func(v))
        self.param_history.append(v)
