import time
import numpy as np
from functools import partial
from jax.random import PRNGKey
from numpyro.infer import MCMC, HMC, NUTS
#from numpyro.infer.util import ParamInfo

from herculens.Inference.inference_base import InferenceBase

# ref: https://bayesianbrad.github.io/posts/2019_hmc.html
# - q is the position, which are variables we are interested in
# - p is the momentum
# The potential energy U(q) will be the minus of the log of the probability density for the distribution 
# of the position variables we wish to sample, plus any constant that is convenient.
# The kinetic energy K(p) will represents the dynamics of our variables.
# A popular form is the Euclidean kinetic energy 1/2 * p^T.(M^-1).p, where M is symmetric, positive definite and typically diagonal.

__all__ = ['Sampler']


class Sampler(InferenceBase):
    """Class that handles sampling tasks, i.e. approximating posterior distributions of parameters.
    It currently supports:
    - Hamiltonian Monte Carlo from numpyro
    - Ensemble Affine Invariant MCMC from emcee
    """

    def hmc(self, num_warmup=100, num_samples=100, num_chains=1, restart_from_init=False,
            sampler_type='NUTS', seed=0, progress_bar=True, sampler_kwargs={}):
        rng_key = PRNGKey(seed)

        if sampler_type.lower() == 'hmc':
            kernel = HMC(potential_fn=self.potential_fn, kinetic_fn=self.kinetic_fn, 
                         **sampler_kwargs)
        elif sampler_type.lower() == 'nuts': # NUTS stands for 'no U-turn sampler'
            kernel = NUTS(potential_fn=self.potential_fn, kinetic_fn=self.kinetic_fn, 
                          **sampler_kwargs)
        
        init_params = self._param.current_values(as_kwargs=False, restart=restart_from_init)
        # alternative way to provide initial parameters through a NamedTuple:
        #init_params = ParamInfo(init_params, potential_fn(init_params), kinetic_fn(init_params))
        num_dims = len(init_params)
        start = time.time()
        samples, extra_fields = self._run_numpyro_mcmc(kernel, init_params, rng_key, 
                                                       num_warmup, num_samples, num_chains, progress_bar)
        runtime = time.time() - start
        logL = - extra_fields['potential_energy']
        samples = np.asarray(samples)
        expected_shape = (num_samples*num_chains, num_dims)
        if samples.T.shape == expected_shape:  # this happens sometimes...
            samples = samples.T
            #raise RuntimeError(f"HMC samples do not have correct shape, {samples.shape} instead of {expected_shape}")
        logL = np.asarray(logL)
        self._param.set_posterior(samples)
        return samples, logL, extra_fields, runtime

    @staticmethod
    def _run_numpyro_mcmc(kernel, init_params, rng_key, num_warmup, num_samples, num_chains, progress_bar):
        # NOTE: disabling the progress-bar can speed up the sampling
        if num_chains > 1:
            init_params = np.repeat(np.expand_dims(init_params, axis=0), num_chains, axis=0)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, 
                    num_chains=num_chains, progress_bar=progress_bar)
        mcmc.run(rng_key, init_params=init_params, extra_fields=('potential_energy', 'energy', 'r', 'accept_prob'))
        #mcmc.print_summary(exclude_deterministic=False)
        samples = mcmc.get_samples()
        extra_fields = mcmc.get_extra_fields()
        return samples, extra_fields

    def mcmc(self, log_likelihood_fn, init_stds, walker_ratio=10, num_warmup=100, num_samples=100, 
             restart_from_init=False, num_threads=1, progress_bar=True):
        """legacy sampling method from lenstronomy, mainly for comparison.
        Warning: `log_likelihood_fn` needs to be non.jitted (emcee pickles this function, which is incomptabile with JAX objetcs)
        """ 
        from emcee import EnsembleSampler
        from lenstronomy.Sampling.Pool.pool import choose_pool
        pool = choose_pool(mpi=False, processes=num_threads, use_dill=True)
        init_means = self._param.current_values(as_kwargs=False, restart=restart_from_init)
        num_dims = len(init_means)
        num_walkers = int(walker_ratio * num_dims)
        init_params = self._init_ball(np.asarray(init_means), np.asarray(init_stds), 
                                      size=num_walkers, dist='normal')
        start = time.time()
        sampler = EnsembleSampler(num_walkers, num_dims, log_likelihood_fn,
                                  pool=pool, backend=None)
        sampler.run_mcmc(init_params, num_warmup + num_samples, progress=progress_bar)
        runtime = time.time() - start
        samples = sampler.get_chain(discard=num_warmup, thin=1, flat=True)
        logL = sampler.get_log_prob(flat=True, discard=num_warmup, thin=1)
        extra_fields = None
        self._param.set_posterior(samples)
        return samples, logL, extra_fields, runtime

    @staticmethod
    def _init_ball(p0, std, size=1, dist='uniform'):
        """
        [from lenstronomy]

        Produce a ball of walkers around an initial parameter value.
        this routine is from the emcee package as it became deprecated there

        :param p0: The initial parameter values (array).
        :param std: The axis-aligned standard deviation (array).
        :param size: The number of samples to produce.
        :param dist: string, specifies the distribution being sampled, supports 'uniform' and 'normal'

        """
        assert(len(p0) == len(std))
        if dist == 'uniform':
            return np.vstack([p0 + std * np.random.uniform(low=-1, high=1, size=len(p0))
                             for i in range(size)])
        elif dist == 'normal':
            return np.vstack([p0 + std * np.random.normal(loc=0, scale=1, size=len(p0))
                              for i in range(size)])
        else:
            raise ValueError('distribution %s not supported. Chose among "uniform" or "normal".' % dist)
