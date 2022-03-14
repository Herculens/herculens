import time
import numpy as np
from functools import partial
import jax

# inference packages
import blackjax.hmc as blackjax_hmc
import blackjax.nuts as blackjax_nuts
import blackjax.stan_warmup as stan_warmup
from numpyro.infer import HMC as numpyro_HMC
from numpyro.infer import NUTS as numpyro_NUTS
from numpyro.infer import MCMC as numpyro_MCMC
#from numpyro.infer.util import ParamInfo
from emcee import EnsembleSampler

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

    def hmc_blackjax(self, num_warmup=100, num_samples=100, num_chains=1, 
                     restart_from_init=False, sampler_type='NUTS', use_stan_warmup=True,
                     step_size=1e-3, inv_mass_matrix=None, num_integ_steps=30, 
                     seed=0):
        rng_key = jax.random.PRNGKey(seed)
        logprob = self.loss
        init_positions = self._param.current_values(as_kwargs=False, restart=restart_from_init)
        if inv_mass_matrix is None:
            # default inverse mass matrix is only 1s
            inv_mass_matrix = np.ones(self._param.num_parameters)

        start = time.time()

        if use_stan_warmup is True:
            rng_key, rng_subkey = jax.random.split(rng_key)
            # here for simplicity we use standard HMC
            # TODO: use NUTS also for Stan warmup?
            init_state_warmup = blackjax_hmc.new_state(init_positions, logprob)
            num_integ_steps_warmup = 30
            kernel_generator = lambda step_size, inverse_mass_matrix: blackjax_hmc.kernel(
                logprob, step_size, inverse_mass_matrix, num_integ_steps_warmup
            )
            # update step size and invser mass matrix during warmup with Stan
            warmup_state, (step_size, inv_mass_matrix), warmup_info = stan_warmup.run(
                rng_subkey,
                kernel_generator,
                init_state_warmup,
                num_warmup,
            )
            # reset number of samples so we don't warmup again in the final inference
            num_warmup = 0
        
        if sampler_type.lower() == 'hmc':
            new_state_func = blackjax_hmc.new_state
            kernel = blackjax_hmc.kernel(logprob, step_size, inv_mass_matrix, num_integ_steps)
        elif sampler_type.lower() == 'nuts':
            new_state_func = blackjax_nuts.new_state
            kernel = blackjax_nuts.kernel(logprob, step_size, inv_mass_matrix)
        else:
            raise ValueError(f"Sampler/kernel type '{sampler_type}' is not supported ('NUTS' or 'HMC' only).")

        if num_chains == 1:
            init_states = new_state_func(init_positions, logprob)
        else:
            init_positions = np.repeat(np.expand_dims(init_positions, axis=0), num_chains, axis=0)
            init_states = jax.vmap(new_state_func, in_axes=(0, None))(init_positions, logprob)
        # run the inference
        states, infos = self._run_blackjax_inference(
            rng_key, kernel, init_states, num_warmup + num_samples, num_chains
        )
        samples = states.position.block_until_ready()
        logL = infos.energy
        runtime = time.time() - start

        if len(samples.shape) == 3:  # basically if num_chains > 1
            # flatten the multiple chains
            s0, s1, s2 = samples.shape
            samples = samples.reshape(s0*s1, s2)
            logL = logL.flatten()

        self._param.set_posterior(samples)
        extra_fields = {
            'step_size': step_size,
            'inverse_mass_matrix': inv_mass_matrix,
            'mean_acceptance_rate': np.mean(infos.acceptance_probability),
            'mean_perc_divergent': 100. * np.mean(infos.is_divergent),
            #'infos': infos,
        }
        return samples, logL, extra_fields, runtime

    @staticmethod
    def _run_blackjax_inference(rng_key, kernel, initial_state, 
                                     num_samples, num_chains):
        def one_step_single_chain(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)
        def one_step_multi_chains(states, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            states, infos = jax.vmap(kernel)(keys, states)
            return states, (states, infos)

        keys = jax.random.split(rng_key, num_samples)
        if num_chains == 1:
            _, (states, infos) = jax.lax.scan(one_step_single_chain, initial_state, keys)
        else:
            _, (states, infos) = jax.lax.scan(one_step_multi_chains, initial_state, keys)
        return states, infos

    def hmc_numpyro(self, num_warmup=100, num_samples=100, num_chains=1, 
                    restart_from_init=False, sampler_type='NUTS', 
                    seed=0, progress_bar=True, sampler_kwargs={}):
        rng_key = jax.random.PRNGKey(seed)

        if sampler_type.lower() == 'hmc':
            kernel = numpyro._HMC(potential_fn=self.potential_fn, kinetic_fn=self.kinetic_fn, 
                                  **sampler_kwargs)
        elif sampler_type.lower() == 'nuts': # NUTS stands for 'no U-turn sampler'
            kernel = numpyro._NUTS(potential_fn=self.potential_fn, kinetic_fn=self.kinetic_fn, 
                                   **sampler_kwargs)
        else:
            raise ValueError(f"Sampler/kernel type '{sampler_type}' is not supported ('NUTS' or 'HMC' only).")
        
        init_params = self._param.current_values(as_kwargs=False, restart=restart_from_init)
        # alternative way to provide initial parameters through a NamedTuple:
        #init_params = ParamInfo(init_params, potential_fn(init_params), kinetic_fn(init_params))
        num_dims = len(init_params)
        start = time.time()
        samples, extra_fields = self._run_numpyro_inference(kernel, init_params, rng_key, 
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
    def _run_numpyro_inference(kernel, init_params, rng_key, num_warmup, num_samples, num_chains, progress_bar):
        # NOTE: disabling the progress-bar can speed up the sampling
        if num_chains > 1:
            init_params = np.repeat(np.expand_dims(init_params, axis=0), num_chains, axis=0)
        mcmc = numpyro_MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, 
                            num_chains=num_chains, progress_bar=progress_bar)
        mcmc.run(rng_key, init_params=init_params, extra_fields=('potential_energy', 'energy', 'r', 'accept_prob'))
        #mcmc.print_summary(exclude_deterministic=False)
        samples = mcmc.get_samples(group_by_chain=False)
        extra_fields = mcmc.get_extra_fields()
        return samples, extra_fields

    def mcmc_emcee(self, log_likelihood_fn, init_stds, walker_ratio=10, 
                   num_warmup=100, num_samples=100, 
                   restart_from_init=False, num_threads=1, progress_bar=True):
        """legacy sampling method from lenstronomy, mainly for comparison.
        Warning: `log_likelihood_fn` needs to be non.jitted (emcee pickles this function, which is incomptabile with JAX objetcs)
        """ 
        from lenstronomy.Sampling.Pool.pool import choose_pool  # TODO: remove this dependence
        pool = choose_pool(mpi=False, processes=num_threads, use_dill=True)
        init_means = self._param.current_values(as_kwargs=False, restart=restart_from_init)
        num_dims = len(init_means)
        num_walkers = int(walker_ratio * num_dims)
        init_params = self._init_emcee_ball(np.asarray(init_means), 
                                            np.asarray(init_stds), 
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
    def _init_emcee_ball(p0, std, size=1, dist='uniform'):
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
