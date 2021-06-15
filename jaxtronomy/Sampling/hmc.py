import time
import jax
from numpyro.infer import MCMC, HMC, NUTS
#from numpyro.infer.util import ParamInfo

# ref: https://bayesianbrad.github.io/posts/2019_hmc.html
# - q is the position, which are variables we are interested in
# - p is the momentum
# The potential energy U(q) will be the minus of the log of the probability density for the distribution 
# of the position variables we wish to sample, plus any constant that is convenient.
# The kinetic energy K(p) will represents the dynamics of our variables.
# A popular form is K(p) = p^T M^{−1} p, where M is symmetric, positive definite and typically diagonal.

def _run_mcmc(kernel, init_params, mcmc_key, num_warmup, num_samples):
    print(f"Initial parameters: {init_params} {init_params.shape}", flush=True)
    # NOTE: num_chains > 1, init_params should be of shape (num_chains, num_dims) instead of (num_dims,)
    #init_params = np.repeat(np.expand_dims(init_params, axis=0), 3, axis=0)
    # NOTE 2: disabling the progress-bar can speed up the sampling
    mcmc = MCMC(kernel, num_warmup, num_samples, num_chains=1, progress_bar=True)
    mcmc.run(mcmc_key, init_params=init_params, extra_fields=('potential_energy', 'energy', 'r', 'accept_prob'))
    mcmc.print_summary(exclude_deterministic=False)
    samples = mcmc.get_samples()
    extra_fields = mcmc.get_extra_fields()
    return samples, extra_fields

def run_inference(potential_fn, init_params, sampler_type='nuts', num_warmup=100, num_samples=100, seed=18, sampler_kwargs={}):
    hmc_key = jax.random.PRNGKey(seed)

    # below, `kinetic_fn=None` will use the default Euclidean kinetic energy 1/2 * p^T.(M^-1).p    
    # standard HMC sampler
    if sampler_type.lower() == 'hmc':
        kernel = HMC(potential_fn=potential_fn, kinetic_fn=None, **sampler_kwargs)
    
    # 'no U-turn sampler' HMC sampler
    elif sampler_type.lower() == 'nuts':
        kernel = NUTS(potential_fn=potential_fn, kinetic_fn=None, **sampler_kwargs)
        
    # custom sampler (example with traditional Metropolis-Hastings sampler)
    #elif sampler_type.lower() == 'mcmc':
    #    kernel = MetropolisHastings(potential_fn, step_size=0.1)
    
    # alternative way to provide initial parameters:
    #init_params = ParamInfo(init_params, potential_fn(init_params), kinetic_fn(init_params))
    
    start = time.time()
    samples, extra_fields = _run_mcmc(kernel, init_params, hmc_key, num_warmup, num_samples)
    runtime = time.time() - start
    return samples, extra_fields, runtime


#####
# example of how to create our own sampler kernel
#####

#from numpyro.infer.mcmc import MCMCKernel
#import numpyro.distributions as dist
#from jax import random
#from collections import namedtuple
#MHState = namedtuple("MHState", ["u", "rng_key"])
#class MetropolisHastings(MCMCKernel):
#    sample_field = "u"
#
#    def __init__(self, potential_fn, step_size=0.1):
#        self.potential_fn = potential_fn
#        self.step_size = step_size
#
#    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
#        return MHState(init_params, rng_key)
#
#    def sample(self, state, model_args, model_kwargs):
#        u, rng_key = state
#        rng_key, key_proposal, key_accept = random.split(rng_key, 3)
#        u_proposal = dist.Normal(u, self.step_size).sample(key_proposal)
#        accept_prob = jnp.exp(self.potential_fn(u) - self.potential_fn(u_proposal))
#        u_new = jnp.where(dist.Uniform().sample(key_accept) < accept_prob, u_proposal, u)
#        return MHState(u_new, rng_key)

