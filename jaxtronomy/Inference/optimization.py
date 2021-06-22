import time
from scipy import optimize
from functools import partial
from jax import jit, grad, jacfwd, jacrev

__all__ = ['Optimizer']


class Optimizer(object):
    """"""

    def __init__(self, loss_fn, param_class):
        self._loss_fn = loss_fn
        self._param_class = param_class

    @partial(jit, static_argnums=(0,))  # because first argument is 'self'
    def loss(self, args):
        return self._loss_fn(self._param_class.args2kwargs(args))

    @partial(jit, static_argnums=(0,))
    def jacobian(self, args):
        return grad(self.loss)(args)

    @partial(jit, static_argnums=(0,))
    def hessian(self, args):
        return jacfwd(jacrev(self.loss))(args)

    #TODO
    # def hessian_vector_product(self, args):
    #     return 

    @property
    def loss_history(self):
        if not hasattr(self, '_metrics'):
            raise ValueError("You muts run the optimizer at least once to access the history")
        return self._metrics.loss_history

    @property
    def param_history(self):
        if not hasattr(self, '_metrics'):
            raise ValueError("You muts run the optimizer at least once to access the history")
        return self._metrics.param_history

    def minimize(self, method='BFGS', restart_from_init=False):
        init_params = self._param_class.initial_values(as_kwargs=False, original=restart_from_init)
        self._metrics = Metrics(self.loss)
        start = time.time()
        best_fit, extra_fields = self._run_minimizer_scipy(init_params, method, self._metrics)
        runtime = time.time() - start
        logL = - float(self._metrics.loss_history[-1])
        self._param_class.set_best_fit(best_fit)
        print("AAAA", best_fit)
        return best_fit, logL, extra_fields, runtime

    def _run_minimizer_scipy(self, x0, method, callback):
        if method in ['Nelder-Mead']:  # TODO: add methods
            extra_kwargs = {}
        elif method in ['BFGS']:  # TODO: add methods
            extra_kwargs = {'jac': self.jacobian}
        elif method in ['Newton-CG', 'trust-krylov']:  # TODO: add methods
            extra_kwargs = {'jac': self.jacobian, 'hess': self.hessian}
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
        optimizer = ParticleSwarmOptimizer(jit(lambda x: - self.loss(x)),
                                           self._param_class._lowers, self._param_class._uppers, 
                                           n_particles, pool=pool)
        init_params = self._param_class.initial_values(as_kwargs=False, original=restart_from_init)
        optimizer.set_global_best(init_params, [0]*len(init_params), - self.loss(init_params))
        start = time.time()
        best_fit, [chi2_list, pos_list, vel_list] = optimizer.optimize(n_iterations)
        runtime = time.time() - start
        logL = float(chi2_list[-1])
        extra_fields = {'chi2_list': chi2_list, 'pos_list': pos_list, 'vel_list': vel_list}
        print("MAIS NOE", best_fit)
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
