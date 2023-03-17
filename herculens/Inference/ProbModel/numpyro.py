# Defines the model of a strong lens
# 
# Copyright (c) 2022, herculens developers and contributors
# based on the ImSim module from lenstronomy (version 1.9.3)

__author__ = 'aymgal'


from functools import partial
import jax
import jax.numpy as jnp
import numpyro
from numpyro import handlers
from numpyro.infer import util
from numpyro.distributions.transforms import biject_to
from numpyro.distributions import constraints
from numpyro.distributions.util import sum_rightmost

from herculens.Inference.ProbModel.base_model import BaseProbModel


__all__ = ['NumpyroModel']


def _unconstrain_reparam(params, site):
    """added support for numpyro.param sites"""
    name = site["name"]
    if name in params:
        p = params[name]
        
        if site["type"] == "param":
            constraint = site["kwargs"].get("constraint", constraints.real)
            with util.helpful_support_errors(site):
                transform = biject_to(constraint)
            
            len_event_shape = len(site["kwargs"]["event_dim"])

        else:
            support = site["fn"].support
            with util.helpful_support_errors(site):
                transform = biject_to(support)
            
            # in scan, we might only want to substitute an item at index i, rather than the whole sequence
            i = site["infer"].get("_scan_current_index", None)
            if i is not None:
                event_dim_shift = transform.codomain.event_dim - t.domain.event_dim
                expected_unconstrained_dim = len(site["fn"].shape()) - event_dim_shift
                # check if p has additional time dimension
                if jnp.ndim(p) > expected_unconstrained_dim:
                    p = p[i]

            if support is constraints.real or (
                isinstance(support, constraints.independent)
                and support.base_constraint is constraints.real
            ):
                return p

            len_event_shape = len(site["fn"].event_shape)

        value = transform(p)

        # NB: we add the determinant term only for sampled sites
        # and only transformed parameter site values above 
        if site["type"] == "sample":
            log_det = transform.log_abs_det_jacobian(p, value)
            log_det = sum_rightmost(
                log_det, jnp.ndim(log_det) - jnp.ndim(value) + len_event_shape
            )
            numpyro.factor("_{}_log_det".format(name), log_det)

        return value


def _transform_fn(model, model_args, model_kwargs, params, invert):
    """
    Transforms parameter values between constrained <-> unconstrained spaces.
    It supports numpyro.param sites
    """
    substituted_model = handlers.substitute(model, params)
    model_trace = handlers.trace(substituted_model).get_trace(*model_args, **model_kwargs)
    values, inv_transforms = {}, {}
    for k, v in model_trace.items():
        if v["type"] == "param":
            values[k] = v["value"]
            constraint = v["kwargs"].pop("constraint", constraints.real)
            with util.helpful_support_errors(v):
                inv_transforms[k] = biject_to(constraint)
        elif (
            v["type"] == "sample"
            and not v["is_observed"]
            and not v["fn"].support.is_discrete
        ):
            values[k] = v["value"]
            with util.helpful_support_errors(v):
                inv_transforms[k] = biject_to(v["fn"].support)
    params_const = util.transform_fn(
        inv_transforms,
        {k: v for k, v in values.items()},
        invert=invert,
    )
    return params_const

def unconstrain_fn(model, model_args, model_kwargs, params):
    return _transform_fn(model, model_args, model_kwargs, params, True)

def constrain_fn(model, model_args, model_kwargs, params):
    return _transform_fn(model, model_args, model_kwargs, params, False)


def potential_energy(model, model_args, model_kwargs, params):
    """
    (EXPERIMENTAL INTERFACE) Computes potential energy of a model given unconstrained params.
    Under the hood, we will transform these unconstrained parameters to the values
    belong to the supports of the corresponding priors in `model`.

    :param model: a callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: unconstrained parameters of `model`.
    :return: potential energy given unconstrained parameters.
    """
    substituted_model = handlers.substitute(
        model, substitute_fn=partial(_unconstrain_reparam, params)
    )
    # no param is needed for log_density computation because we already substitute
    log_joint, model_trace = util.log_density(
        substituted_model, model_args, model_kwargs, {}
    )
    return -log_joint


class NumpyroModel(BaseProbModel):
    """Defines a numpyro model based on a LensImage instance"""

    @property
    def num_parameters(self):
        if not hasattr(self, '_num_param'):
            num_param = 0
            for site in self.get_trace().values():
                if (site['type'] == 'sample' and not site['is_observed']
                    or site['type'] == 'param'):
                    num_param += site['value'].size
            self._num_param = num_param
        return self._num_param

    def log_prob(self, params, constrained=True):
        """returns the logarithm of the data likelihood plus the logarithm of the prior"""
        if constrained is True:
            # do this for optimisation in constrained space
            log_prob, model_trace = util.log_density(self.model, (), {}, params)
        else:
            # do this for optimisation in unconstrained space
            log_prob = - potential_energy(self.model, (), {}, params)
            #params_c = self.constrain(params)
            #log_prob, model_trace = util.log_density(self.model, (), {}, params_c)
        return log_prob
    
    def log_likelihood(self, params):
        # returns the logarithm of the data likelihood
        return util.log_likelihood(self.model, params, batch_ndims=0)['obs']

    def seeded_model(self, seed=0):
        return handlers.seed(self.model, jax.random.PRNGKey(seed))
    
    def get_trace(self, seed=0):
        return handlers.trace(self.seeded_model(seed=seed)).get_trace()

    def get_sample(self, seed=0):
        trace = self.get_trace(seed=seed)
        return {site['name']: site['value'] for site in trace.values() if not site.get('is_observed', False)}

    def sample_prior(self, num_samples, seed=0):
        batch_ndims = 0 if num_samples else 1
        predictive = util.Predictive(self.model, 
                                     num_samples=num_samples, 
                                     batch_ndims=batch_ndims)
        samples = predictive(jax.random.PRNGKey(seed))
        del samples['obs']
        return samples

    def render_model(self):
        return numpyro.render_model(self.model)

    def params2kwargs(self, params, transform=False, constrained=True):
        """
        If transform=True, parameters will get transformed to their codomain.
        If constrained=True, assumes that the input are in constrained space,
        otherwise they are assumed to be in constrained values
        """
        if transform is False:
            params_ = params
        elif constrained is True:
            params_ = self.unconstrain(params)
        else:
            params_ = self.constrain(params)
        return self._params2kwargs(params_)

    def constrain(self, params):
        return constrain_fn(self.model, (), {}, params)

    def unconstrain(self, params):
        return unconstrain_fn(self.model, (), {}, params)
