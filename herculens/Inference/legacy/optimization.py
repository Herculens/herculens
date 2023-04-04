# Handles different method to optimize a loss function
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal', 'austinpeel'


import time
import warnings
import numpy as np
import jax
import optax
from scipy import optimize
from scipy.optimize import Bounds
from tqdm import tqdm
from copy import deepcopy

from herculens.Inference.legacy.base_inference import Inference

__all__ = ['Optimizer']


class Optimizer(Inference):
    """Class that handles optimization tasks, i.e. finding best-fit point estimates of parameters
    It currently handles:
    - a subset of scipy.optimize.minimize routines, using first and second order derivatives when required
    - a particle swarm optimizer (PSO), implemented in lenstronomy
    """

    _supported_scipy_methods = ['Nelder-Mead', 'BFGS', 'Newton-CG', 'trust-krylov', 'trust-exact', 'trust-constr']

    # @property
    # def loss_history(self):
    #     if not hasattr(self, '_metrics'):
    #         raise ValueError("You must run the optimizer at least once to access the history")
    #     return self._metrics.loss_history

    # @property
    # def param_history(self):
    #     if not hasattr(self, '_metrics'):
    #         raise ValueError("You must run the optimizer at least once to access the history")
    #     return self._metrics.param_history

    def minimize(self, method='BFGS', maxiter=None, init_params=None,
                 restart_from_init=False, use_exact_hessian_if_allowed=False,
                 multi_start_from_prior=False, num_multi_start=1, seed_multi_start=None,
                 progress_bar=True, return_param_history=False):
        # TODO: should we call once / a few times all jitted functions before optimization, to potentially speed things up?
        metrics = MinimizeMetrics(self._loss, method, with_param_history=return_param_history)
        if multi_start_from_prior is False:
            num_multi_start = 1
        if num_multi_start == 1:
            if init_params is None:
                init_params = self._param.current_values(as_kwargs=False, restart=restart_from_init, copy=True)
            init_samples = np.asarray(init_params)[None, :]
        else:
            init_samples = self._param.draw_prior_samples(num_samples=num_multi_start, 
                                                          seed=seed_multi_start)
        exact_hessian = use_exact_hessian_if_allowed

        start = time.time()
        best_fit_list = []
        logL_best_fit_list = []
        loss_history_list = []
        param_history_list = []
        extra_fields_list = []
        for n in self._for_loop(range(num_multi_start), progress_bar, 
                                total=num_multi_start, 
                                desc=f"minimize.{method}"):
            init_params_n = init_samples[n, :]
            best_fit_n, extra_fields_n = self._run_scipy_minimizer(init_params_n, 
                                                                   method, maxiter, metrics,
                                                                   exact_hessian)
            if metrics.loss_history == []:
                warnings.warn("The loss history does not contain any value")
            best_fit_list.append(best_fit_n)
            logL_best_fit_list.append(self.log_probability(best_fit_n))
            loss_history_list.append(metrics.get_loss_history())
            param_history_list.append(metrics.get_param_history())
            extra_fields_list.append(extra_fields_n)

        # select the best fit among the multi start runs
        if num_multi_start > 1:
            index = np.argmax(logL_best_fit_list)
        else:
            index = 0
        best_fit = best_fit_list[index]
        logL_best_fit = logL_best_fit_list[index]
        extra_fields  = extra_fields_list[index]

        runtime = time.time() - start

        extra_fields['best_fit_index'] = index
        extra_fields['loss_history'] = loss_history_list[index]
        extra_fields['loss_history_list'] = loss_history_list
        if return_param_history is True:
            extra_fields['param_history'] = param_history_list[index]
            extra_fields['param_history_list'] = param_history_list  # maybe too memory consuming?
        self._param.set_best_fit(best_fit)
        return best_fit, logL_best_fit, extra_fields, runtime

    def _run_scipy_minimizer(self, x0, method, maxiter, callback, exact_hessian):
        callback.reset()
        if method not in self._supported_scipy_methods:
            raise ValueError(f"Minimize method '{method}' is not supported.")
        # here we only put select the kwargs related to the chosen method 
        extra_kwargs = {}
        if method in ['BFGS']:
            extra_kwargs['jac'] = self._loss.gradient
        elif method in ['Newton-CG', 'trust-krylov', 'trust-exact', 'trust-constr']:
            extra_kwargs['jac'] = self._loss.gradient
            if method == 'trust-exact' or exact_hessian is True:
                extra_kwargs['hess'] = self._loss.hessian
            else:
                extra_kwargs['hessp'] = self._loss.hessian_vec_prod
            if method == 'trust-constr':
                extra_kwargs['bounds'] = Bounds(*self._param.bounds)
        if maxiter is not None:
            extra_kwargs['options'] = {'maxiter': maxiter}
        res = optimize.minimize(self._loss, x0, method=method, callback=callback, 
                                **extra_kwargs)
        extra_fields = {'result_class': res, 'jac': None, 'hess': None, 'hess_inv': None}
        for key in extra_fields:
            if hasattr(res, key):
                extra_fields[key] = getattr(res, key)
        return res.x, extra_fields

    def optax(self, algorithm='adabelief', max_iterations=100, min_iterations=None,
              init_learning_rate=1e-2, schedule_learning_rate=True, 
              restart_from_init=False, stop_at_loss_increase=False, 
              progress_bar=True, return_param_history=False):
        if min_iterations is None:
            min_iterations = max_iterations
        if schedule_learning_rate is True:
            # Exponential decay of the learning rate
            scheduler = optax.exponential_decay(
                init_value=init_learning_rate, 
                decay_rate=0.99, # TODO: this has never been fine-tuned (taken from optax examples)
                transition_steps=max_iterations)

            if algorithm.lower() == 'adabelief':
                scale_algo = optax.scale_by_belief()
            elif algorithm.lower() == 'radam':
                scale_algo = optax.scale_by_radam()
            elif algorithm.lower() == 'adam':
                scale_algo = optax.scale_by_adam()
            else:
                raise ValueError(f"Optax algorithm '{algorithm}' is not supported")

            # Combining gradient transforms using `optax.chain`
            optim = optax.chain(
                #optax.clip_by_global_norm(1.0),  # clip by the gradient by the global norm # TODO: what is this used for?
                scale_algo,  # use the updates from the chosen optimizer
                optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler
                optax.scale(-1.)  # because gradient *descent*
            )
        else:
            if algorithm.lower() == 'adabelief':
                optim = optax.adabelief(init_learning_rate)
            elif algorithm.lower() == 'radam':
                optim = optax.radam(init_learning_rate)
            elif algorithm.lower() == 'adam':
                optim = optax.adam(init_learning_rate)
            else:
                raise ValueError(f"Optax algorithm '{algorithm}' is not supported")

        # Initialise optimizer state
        params = self._param.current_values(as_kwargs=False, restart=restart_from_init, copy=True)
        opt_state = optim.init(params)
        prev_params, prev_loss_val = params, 1e10

        @jax.jit
        def gradient_step(params, opt_state):
            #loss_val, grads = jax.value_and_grad(self._loss)(params)
            loss_val, grads = self._loss.value_and_gradient(params)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        # Gradient descent loop
        param_history = []
        loss_history = []
        start_time = time.time()
        for i in self._for_loop(range(max_iterations), progress_bar, 
                                total=max_iterations, 
                                desc=f"optax.{algorithm}"):
            params, opt_state, loss_val = gradient_step(params, opt_state)
            if stop_at_loss_increase and i > min_iterations and loss > prev_loss:
                params, loss_val = prev_params, prev_loss_val
                break
            else:
                loss_history.append(loss_val)
                prev_params, prev_loss_val = params, loss_val
            if return_param_history is True:
                param_history.append(params)
        runtime = time.time() - start_time
        best_fit = params
        logL_best_fit = self.log_probability(best_fit)
        extra_fields = {'loss_history': np.array(loss_history)}  # TODO: use optax.second_order module to compute diagonal of Hessian?
        if return_param_history is True:
            extra_fields['param_history'] = param_history
        self._param.set_best_fit(best_fit)
        return best_fit, logL_best_fit, extra_fields, runtime

    def gradient_descent(self, num_iterations=100, step_size=1e-2, restart_from_init=False):
        import jax.numpy as jnp
        # Initialise optimizer state
        params = self._param.current_values(as_kwargs=False, restart=restart_from_init, copy=True)
        
        # Gradient descent loop
        start_time = time.time()
        for _ in range(num_iterations):
            params -= step_size * self._loss.gradient(params)
        runtime = time.time() - start_time
        best_fit = params
        logL_best_fit = self.log_probability(best_fit)
        extra_fields = {}  # TODO: use optax.second_order module to compute diagonal of Hessian?
        self._param.set_best_fit(best_fit)
        return best_fit, logL_best_fit, extra_fields, runtime

    def pso(self, n_particles=100, n_iterations=100, restart_from_init=False, n_threads=1):
        """particle swarm optimization, implemented in lenstronomy"""
        from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
        from lenstronomy.Sampling.Pool.pool import choose_pool
        
        pool = choose_pool(mpi=False, processes=n_threads, use_dill=True)
        lowers, uppers = self._param.bounds
        if np.any(np.isnan(lowers)):
            raise ValueError("PSO needs lower and upper bounds, i.e. prior distributions with a finite support")
        optimizer = ParticleSwarmOptimizer(self.log_likelihood, 
                                           lowers, uppers, n_particles, pool=pool)
        init_params = self._param.current_values(as_kwargs=False, restart=restart_from_init)
        optimizer.set_global_best(init_params, [0]*len(init_params), - self._loss(init_params))
        start = time.time()
        best_fit, [chi2_list, pos_list, vel_list] = optimizer.optimize(n_iterations)
        runtime = time.time() - start
        logL_best_fit = float(chi2_list[-1])
        extra_fields = {'chi2_list': chi2_list, 'pos_list': pos_list, 'vel_list': vel_list}
        self._param.set_best_fit(best_fit)
        return best_fit, logL_best_fit, extra_fields, runtime


    @staticmethod
    def _for_loop(iterable, progress_bar_bool, **tqdm_kwargs):
        if progress_bar_bool is True:
            return tqdm(iterable, **tqdm_kwargs)
        else:
            return iterable


class MinimizeMetrics(object):
    """simple callable class used as callback in scipy.optimize.minimize method"""
    
    def __init__(self, func, method, with_param_history=False):
        self._func = func
        if method == 'trust-constr':
            self._call = self._call_2args
        else:
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
