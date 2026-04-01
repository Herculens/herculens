# Unit tests for numpyro_to_nifty_prior and the truncated-distribution transforms
# in herculens.Util.jifty_util.
#
# Strategy
# --------
# Each transform is tested at two levels:
#   1. Statistical correctness: draw N samples from N(0,1), push through the
#      transform, and compare empirical moments / bounds to the analytical
#      reference from scipy.stats.
#   2. JAX-compatibility: the transform must survive jit compilation and
#      reverse-mode differentiation (jax.value_and_grad).
#
# The end-to-end numpyro_to_nifty_prior function is tested with a small model
# that exercises every supported distribution type.

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from scipy.stats import truncnorm, truncnorm as scipy_truncnorm

from herculens.Util.jifty_util import (
    NormalTransform,
    LognormalTransform,
    UniformTransform,
    LogUniformTransform,
    TruncatedNormalTransform,
    TruncatedLognormalTransform,
    numpyro_to_nifty_prior,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SAMPLES = 5_000
RNG = np.random.default_rng(42)
RTOL_MEAN = 0.03   # 3 % relative tolerance for empirical mean comparisons
ATOL_STD  = 0.05   # absolute tolerance for empirical std comparisons


def _std_normal_samples(n=N_SAMPLES):
    return RNG.standard_normal(n)


def _apply_scalar_transform(transform, xi_samples, key):
    """Push an array of standard-normal scalars through a jft.Model transform."""
    return np.array([float(transform({key: jnp.array(x)})[key]) for x in xi_samples])


# ---------------------------------------------------------------------------
# LogUniformTransform
# ---------------------------------------------------------------------------

class TestLogUniformTransform:

    def test_samples_within_bounds(self):
        low, high = 0.1, 10.0
        t = LogUniformTransform(low, high, 'p')
        xi = _std_normal_samples()
        samples = _apply_scalar_transform(t, xi, 'p')
        assert np.all(samples >= low - 1e-9), "Samples below lower bound"
        assert np.all(samples <= high + 1e-9), "Samples above upper bound"

    def test_log_uniform_distribution(self):
        """Samples must be approximately uniform in log space."""
        low, high = 0.1, 10.0
        t = LogUniformTransform(low, high, 'p')
        xi = _std_normal_samples(20_000)
        samples = _apply_scalar_transform(t, xi, 'p')
        log_samples = np.log(samples)
        # Log-space mean and std of Uniform(log(low), log(high))
        log_low, log_high = np.log(low), np.log(high)
        expected_log_mean = (log_low + log_high) / 2.
        expected_log_std  = (log_high - log_low) / np.sqrt(12.)
        assert abs(log_samples.mean() - expected_log_mean) < 0.05
        assert abs(log_samples.std()  - expected_log_std)  < 0.05

    def test_jit_compatible(self):
        t = LogUniformTransform(0.1, 10.0, 'p')
        fn = jax.jit(lambda x: t({'p': x})['p'])
        result = fn(jnp.array(0.0))
        assert jnp.isfinite(result)

    def test_differentiable(self):
        t = LogUniformTransform(0.1, 10.0, 'p')
        fn = lambda x: t({'p': x})['p']
        val, grad = jax.value_and_grad(fn)(jnp.array(0.0))
        assert jnp.isfinite(val)
        assert jnp.isfinite(grad)
        assert grad > 0.0, "Transform should be monotonically increasing"


# ---------------------------------------------------------------------------
# TruncatedNormalTransform
# ---------------------------------------------------------------------------

class TestTruncatedNormalTransform:

    def test_two_sided_samples_within_bounds(self):
        loc, scale, low, high = 0.5, 0.3, 0.0, 1.0
        t = TruncatedNormalTransform(loc, scale, low, high, 'p')
        xi = _std_normal_samples()
        samples = _apply_scalar_transform(t, xi, 'p')
        assert np.all(samples >= low - 1e-9), "Samples below lower bound"
        assert np.all(samples <= high + 1e-9), "Samples above upper bound"

    def test_two_sided_moments(self):
        loc, scale, low, high = 0.5, 0.3, 0.0, 1.0
        a, b = (low - loc) / scale, (high - loc) / scale
        expected_mean = scipy_truncnorm.mean(a, b, loc=loc, scale=scale)
        expected_std  = scipy_truncnorm.std( a, b, loc=loc, scale=scale)

        t = TruncatedNormalTransform(loc, scale, low, high, 'p')
        xi = _std_normal_samples()
        samples = _apply_scalar_transform(t, xi, 'p')

        assert abs(samples.mean() - expected_mean) < RTOL_MEAN * abs(expected_mean) + 1e-6
        assert abs(samples.std()  - expected_std)  < ATOL_STD

    def test_left_truncated_only(self):
        """One-sided lower bound: high = +inf."""
        loc, scale, low = 0.0, 1.0, 1.0
        t = TruncatedNormalTransform(loc, scale, low, np.inf, 'p')
        xi = _std_normal_samples()
        samples = _apply_scalar_transform(t, xi, 'p')
        assert np.all(samples >= low - 1e-9)

        a = (low - loc) / scale
        expected_mean = scipy_truncnorm.mean(a, np.inf, loc=loc, scale=scale)
        assert abs(samples.mean() - expected_mean) < RTOL_MEAN * abs(expected_mean) + 1e-6

    def test_right_truncated_only(self):
        """One-sided upper bound: low = -inf."""
        loc, scale, high = 0.0, 1.0, -0.5
        t = TruncatedNormalTransform(loc, scale, -np.inf, high, 'p')
        xi = _std_normal_samples()
        samples = _apply_scalar_transform(t, xi, 'p')
        assert np.all(samples <= high + 1e-9)

    def test_jit_compatible(self):
        t = TruncatedNormalTransform(0.5, 0.3, 0.0, 1.0, 'p')
        fn = jax.jit(lambda x: t({'p': x})['p'])
        result = fn(jnp.array(0.0))
        assert jnp.isfinite(result)

    def test_differentiable(self):
        t = TruncatedNormalTransform(0.5, 0.3, 0.0, 1.0, 'p')
        fn = lambda x: t({'p': x})['p']
        val, grad = jax.value_and_grad(fn)(jnp.array(0.0))
        assert jnp.isfinite(val)
        assert jnp.isfinite(grad)
        assert grad > 0.0, "Transform should be monotonically increasing"


# ---------------------------------------------------------------------------
# TruncatedLognormalTransform
# ---------------------------------------------------------------------------

class TestTruncatedLognormalTransform:

    # numpyro LogNormal(loc, scale): loc and scale are log-space parameters.
    # mean of X = exp(loc + scale^2/2), std of X = mean * sqrt(exp(scale^2) - 1)

    def test_samples_within_bounds(self):
        loc, scale, low, high = 0.0, 0.5, 0.5, 3.0
        t = TruncatedLognormalTransform(loc, scale, low, high, 'p')
        xi = _std_normal_samples()
        samples = _apply_scalar_transform(t, xi, 'p')
        assert np.all(samples >= low - 1e-9)
        assert np.all(samples <= high + 1e-9)

    def test_samples_positive(self):
        """Even with low=0 (treated as no lower bound), all samples must be > 0."""
        loc, scale = 0.0, 0.5
        t = TruncatedLognormalTransform(loc, scale, 0.0, np.inf, 'p')
        xi = _std_normal_samples()
        samples = _apply_scalar_transform(t, xi, 'p')
        assert np.all(samples > 0.0)

    def test_left_truncated_moments(self):
        """With only a lower bound, compare log-space mean to truncated normal in log-space."""
        loc, scale, low = 0.0, 0.5, 1.0   # low in original space → log(1.0)=0 in log-space
        t = TruncatedLognormalTransform(loc, scale, low, np.inf, 'p')
        xi = _std_normal_samples()
        samples = _apply_scalar_transform(t, xi, 'p')

        # In log-space, this is a left-truncated normal with a=0, b=inf
        a = (np.log(low) - loc) / scale   # = 0.0
        expected_log_mean = scipy_truncnorm.mean(a, np.inf, loc=loc, scale=scale)
        empirical_log_mean = np.log(samples).mean()
        assert abs(empirical_log_mean - expected_log_mean) < RTOL_MEAN * abs(expected_log_mean) + 1e-4

    def test_jit_compatible(self):
        t = TruncatedLognormalTransform(0.0, 0.5, 0.5, 3.0, 'p')
        fn = jax.jit(lambda x: t({'p': x})['p'])
        result = fn(jnp.array(0.0))
        assert jnp.isfinite(result)

    def test_differentiable(self):
        t = TruncatedLognormalTransform(0.0, 0.5, 0.5, 3.0, 'p')
        fn = lambda x: t({'p': x})['p']
        val, grad = jax.value_and_grad(fn)(jnp.array(0.0))
        assert jnp.isfinite(val)
        assert jnp.isfinite(grad)
        assert grad > 0.0


# ---------------------------------------------------------------------------
# numpyro_to_nifty_prior — end-to-end
# ---------------------------------------------------------------------------

class TestNumpyrToNiftyPrior:

    def _make_model(self):
        """Small numpyro model covering all supported distribution types."""
        class _Model:
            def model(self):
                numpyro.sample('a_normal',    dist.Normal(0.5, 0.2))
                numpyro.sample('b_lognorm',   dist.LogNormal(0.0, 0.3))
                numpyro.sample('c_uniform',   dist.Uniform(1.0, 3.0))
                numpyro.sample('d_logunif',   dist.LogUniform(0.1, 10.0))
                numpyro.sample('e_trunc2',    dist.TruncatedNormal(0.5, 0.2, low=0.0, high=1.0))
                numpyro.sample('f_trunc_lo',  dist.TruncatedNormal(0.0, 1.0, low=0.5))
        return _Model()

    def test_returns_valid_transform(self):
        import nifty.re as jft
        model = self._make_model()
        prior = numpyro_to_nifty_prior(model.model)
        assert hasattr(prior, 'domain'), "Prior transform must have a .domain attribute"
        # Domain should contain all 6 parameters
        assert set(prior.domain.keys()) == {'a_normal', 'b_lognorm', 'c_uniform',
                                             'd_logunif', 'e_trunc2', 'f_trunc_lo'}

    def test_transform_callable(self):
        model = self._make_model()
        prior = numpyro_to_nifty_prior(model.model)
        # Domain values are ShapeWithDtype instances; use their .shape directly
        xi = {k: jnp.zeros(v.shape) for k, v in prior.domain.items()}
        result = prior(xi)
        assert set(result.keys()) == set(prior.domain.keys())
        for v in result.values():
            assert jnp.isfinite(v), "Transform output must be finite at xi=0"

    def test_truncated_normal_bounds_respected(self):
        """TruncatedNormal samples from the prior must stay within [low, high]."""
        import nifty.re as jft
        class _M:
            def model(self):
                numpyro.sample('x', dist.TruncatedNormal(0.5, 0.2, low=0.0, high=1.0))
        prior = numpyro_to_nifty_prior(_M().model)
        xi_vals = RNG.standard_normal(1000)
        samples = np.array([float(prior({'x': jnp.array(xi)})['x']) for xi in xi_vals])
        assert np.all(samples >= 0.0 - 1e-9)
        assert np.all(samples <= 1.0 + 1e-9)

    def test_truncated_lognormal_bounds_respected(self):
        """TruncatedLogNormal samples must stay within [low, high].

        Note: numpyro's TwoSidedTruncatedDistribution only accepts a limited set
        of base distributions (Normal, Cauchy, …) and does NOT accept LogNormal.
        TruncatedLognormalTransform is therefore tested directly here, independently
        of numpyro_to_nifty_prior (which would only be reached if a future numpyro
        version allows constructing such a distribution).
        """
        t = TruncatedLognormalTransform(0.0, 0.5, 0.5, 3.0, 'x')
        xi_vals = RNG.standard_normal(1000)
        samples = np.array([float(t({'x': jnp.array(xi)})['x']) for xi in xi_vals])
        assert np.all(samples >= 0.5 - 1e-9)
        assert np.all(samples <= 3.0 + 1e-9)

    def test_unsupported_distribution_raises(self):
        """An unsupported distribution type must raise NotImplementedError."""
        class _M:
            def model(self):
                numpyro.sample('x', dist.Laplace(0.0, 1.0))
        with pytest.raises(NotImplementedError):
            numpyro_to_nifty_prior(_M().model)

    def test_verbose_output(self, capsys):
        class _M:
            def model(self):
                numpyro.sample('x', dist.TruncatedNormal(0.0, 1.0, low=-1.0, high=1.0))
        numpyro_to_nifty_prior(_M().model, verbose=True)
        captured = capsys.readouterr()
        assert 'TruncatedNormal' in captured.out
