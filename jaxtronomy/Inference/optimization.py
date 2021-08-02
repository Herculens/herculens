import time
from scipy import optimize
import numpy as np
from scipy.optimize import Bounds
import optax

from jaxtronomy.Inference.inference_base import InferenceBase

__all__ = ['Optimizer']


class Optimizer(InferenceBase):
    """Class that handles optimization tasks, i.e. finding best-fit point estimates of parameters
    It currently handles:
    - a subset of scipy.optimize.minimize routines, using first and second order derivatives when required
    - a particle swarm optimizer (PSO), implemented in lenstronomy
    """

    _supported_scipy_methods = ['Nelder-Mead', 'BFGS', 'Newton-CG', 'trust-krylov', 'trust-exact', 'trust-constr']

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

    def minimize(self, method='BFGS', maxiter=None, restart_from_init=False, use_exact_hessian_if_allowed=False):
        # TODO: should we call once / a few times all jitted functions before optimization, to potentially speed things up?
        init_params = self._param.initial_values(as_kwargs=False, original=restart_from_init)
        self._metrics = MinimizeMetrics(self.loss, method)
        start = time.time()
        best_fit, extra_fields = self._run_scipy_minimizer(init_params, method, maxiter, self._metrics,
                                                           use_exact_hessian_if_allowed)
        runtime = time.time() - start
        if self._metrics.loss_history == []:
            raise ValueError("The loss history does not contain any value")
        logL_best_fit = - float(self._metrics.loss_history[-1])
        self._param.set_best_fit(best_fit)
        return best_fit, logL_best_fit, extra_fields, runtime

    def _run_scipy_minimizer(self, x0, method, maxiter, callback, exact_hessian):
        if method not in self._supported_scipy_methods:
            raise ValueError(f"Minimize method '{method}' is not supported.")
        # here we only put select the kwargs related to the chosen method 
        extra_kwargs = {}
        if method in ['BFGS']:
            extra_kwargs['jac'] = self.gradient
        elif method in ['Newton-CG', 'trust-krylov', 'trust-exact', 'trust-constr']:
            extra_kwargs['jac'] = self.gradient
            if method == 'trust-exact' or exact_hessian is True:
                extra_kwargs['hess'] = self.hessian
            else:
                extra_kwargs['hessp'] = self.hessian_vec_prod
            if method == 'trust-constr':
                extra_kwargs['bounds'] = Bounds(*self._param.bounds)
        if maxiter is not None:
            extra_kwargs['options'] = {'maxiter': maxiter}
        res = optimize.minimize(self.loss, x0, method=method, callback=callback, 
                                **extra_kwargs)
        extra_fields = {'result_class': res, 'jac': None, 'hess': None, 'hess_inv': None}
        for key in extra_fields:
            if hasattr(res, key):
                extra_fields[key] = getattr(res, key)
        return res.x, extra_fields

    def optax(self, max_iterations, init_learning_rate, restart_from_init=False):
        # Exponential decay of the learning rate
        scheduler = optax.exponential_decay(
            init_value=init_learning_rate, 
            transition_steps=max_iterations,
            decay_rate=0.99)

        # Combining gradient transforms using `optax.chain`
        optim = optax.chain(
            #optax.clip_by_global_norm(1.0),  # clip by the gradient by the global norm
            optax.scale_by_belief(),  # use the updates from AdaBelief optimizer
            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler
            optax.scale(-1.)  # because gradient *descent*
        )

        # Initialise optimizer state
        init_params = self._param.initial_values(as_kwargs=False, original=restart_from_init)
        opt_state = optim.init(init_params)

        # Gradient descent loop
        start_time = time.time()
        for _ in range(max_iterations):
            updates, opt_state = optim.update(self.gradient(params), opt_state, params)
            params = optax.apply_updates(params, updates)
        runtime = time.time() - start_time
        best_fit = params
        logL_best_fit = self.loss(best_fit)
        extra_fields = {}
        return best_fit, logL_best_fit, extra_fields, runtime

    def pso(self, n_particles=100, n_iterations=100, restart_from_init=False, n_threads=1):
        """legacy optimization method from lenstronomy, mainly for comparison"""
        from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
        from lenstronomy.Sampling.Pool.pool import choose_pool
        pool = choose_pool(mpi=False, processes=n_threads, use_dill=True)
        lowers, uppers = self._param.bounds
        if np.any(np.isnan(lowers)):
            raise ValueError("PSO needs lower and upper bounds, i.e. prior distributions with a finite support")
        optimizer = ParticleSwarmOptimizer(self.log_likelihood, 
                                           lowers, uppers, n_particles, pool=pool)
        init_params = self._param.initial_values(as_kwargs=False, original=restart_from_init)
        optimizer.set_global_best(init_params, [0]*len(init_params), - self.loss(init_params))
        start = time.time()
        best_fit, [chi2_list, pos_list, vel_list] = optimizer.optimize(n_iterations)
        runtime = time.time() - start
        logL_best_fit = float(chi2_list[-1])
        extra_fields = {'chi2_list': chi2_list, 'pos_list': pos_list, 'vel_list': vel_list}
        self._param.set_best_fit(best_fit)
        return best_fit, logL_best_fit, extra_fields, runtime


class MinimizeMetrics(object):
    """simple callable class used as callback in scipy.optimize.minimize method"""
    
    def __init__(self, func, method):
        self.loss_history = []
        self.param_history = []
        self._func = func
        if method == 'trust-constr':
            self._call = self._call_2args
        else:
            self._call = self._call_1arg

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)
        
    def _call_1arg(self, x):
        self.loss_history.append(self._func(x))
        self.param_history.append(x)

    def _call_2args(self, x, state):
        # Input state parameter is necessary for 'trust-constr' method
        # You can use it to stop execution early by returning True
        self.loss_history.append(self._func(x))
        self.param_history.append(x)
