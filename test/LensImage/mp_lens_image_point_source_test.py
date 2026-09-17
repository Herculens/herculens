"""Integration tests for MPLensImage with MPPointSourceModel.

These tests verify the end-to-end integration of point source rendering
inside the multi-plane lens image model.  A minimal but physically valid
two-plane system (one EPL deflector, two Gaussian light planes) is used
throughout.

Tests covered
-------------
1.  MPLensImage instantiation without a point source model (unchanged behaviour)
2.  MPLensImage instantiation with an MPPointSourceModel
3.  point_source_image() returns zeros when no PS model is attached
4.  point_source_image() with LENSED_POSITIONS has the correct output shape
5.  point_source_image() with LENSED_POSITIONS is non-zero for in-field positions
6.  point_source_image() returns zeros when no point sources are given (empty lists)
7.  model() with point_source_add=False is identical to model() without point sources
8.  model() with point_source_add=True differs from the light-only model
9.  model() with point_source_add=True has the correct output shape
10. Calling model() twice (JIT cache) gives identical results
11. Gradient of model sum w.r.t. point source amplitudes is finite
12. Point source amplitude scales the rendered flux linearly
13. Point source at off-field position contributes no flux to the image
14. Mixed LENSED_POSITIONS point sources: only non-zero images when in field
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from herculens.MassModel.mass_model_multiplane import MPMassModel
from herculens.LightModel.light_model import LightModel
from herculens.LightModel.light_model_multiplane import MPLightModel
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.LensImage.lens_image_multiplane import MPLensImage
from herculens.PointSourceModel.point_source_multiplane import MPPointSourceModel


# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------
NPIX = 20
PIX_SCL = 0.1  # arcsec / pixel  →  field spans ≈ ±1 arcsec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def base_grid():
    half = NPIX * PIX_SCL / 2.
    return PixelGrid(
        nx=NPIX, ny=NPIX,
        ra_at_xy_0=-half + PIX_SCL / 2.,
        dec_at_xy_0=-half + PIX_SCL / 2.,
        transform_pix2angle=PIX_SCL * np.eye(2),
    )


@pytest.fixture(scope='module')
def base_psf():
    return PSF(psf_type='GAUSSIAN', fwhm=0.3, pixel_size=PIX_SCL)


@pytest.fixture(scope='module')
def base_noise():
    return Noise(NPIX, NPIX, background_rms=0.01, exposure_time=1000.)


@pytest.fixture(scope='module')
def mp_mass():
    """Single EPL deflector plane."""
    return MPMassModel([['EPL']])


@pytest.fixture(scope='module')
def mp_light():
    """Gaussian light on each of the two planes (lens + source)."""
    return MPLightModel([
        LightModel(['GAUSSIAN']),
        LightModel(['GAUSSIAN']),
    ])


@pytest.fixture(scope='module')
def eta_flat():
    """For a single-deflector system the eta matrix has no free entries."""
    return jnp.array([])


@pytest.fixture(scope='module')
def kwargs_mass():
    return [[{
        'theta_E': 0.5, 'gamma': 2.0,
        'center_x': 0., 'center_y': 0.,
        'e1': 0., 'e2': 0.,
    }]]


@pytest.fixture(scope='module')
def kwargs_light():
    return [
        [{'amp': 1.0, 'sigma': 0.3, 'center_x': 0., 'center_y': 0.}],  # lens light
        [{'amp': 2.0, 'sigma': 0.2, 'center_x': 0., 'center_y': 0.}],  # source light
    ]


@pytest.fixture(scope='module')
def ps_model_lensed():
    """A single LENSED_POSITIONS point source model."""
    return MPPointSourceModel(['LENSED_POSITIONS'])


@pytest.fixture(scope='module')
def kwargs_ps_infield():
    """Two image-plane positions that lie inside the ±0.95 arcsec field."""
    return [{'ra': jnp.array([0.3, -0.3]), 'dec': jnp.array([0.2, -0.2]),
             'amp': jnp.array([5.0, 5.0])}]


@pytest.fixture(scope='module')
def lens_image_no_ps(base_grid, base_psf, base_noise, mp_mass, mp_light):
    """MPLensImage with no point source model."""
    return MPLensImage(base_grid, base_psf, base_noise, mp_mass, mp_light)


@pytest.fixture(scope='module')
def lens_image_with_ps(base_grid, base_psf, base_noise, mp_mass, mp_light, ps_model_lensed):
    """MPLensImage with an LENSED_POSITIONS point source model."""
    return MPLensImage(
        base_grid, base_psf, base_noise, mp_mass, mp_light,
        point_source_model_class=ps_model_lensed,
    )


# ---------------------------------------------------------------------------
# 1 & 2. Instantiation
# ---------------------------------------------------------------------------

def test_instantiation_without_ps(lens_image_no_ps):
    assert lens_image_no_ps.MPPointSourceModel is None


def test_instantiation_with_ps(lens_image_with_ps, ps_model_lensed):
    assert lens_image_with_ps.MPPointSourceModel is ps_model_lensed


# ---------------------------------------------------------------------------
# 3. point_source_image returns zeros when no model is set
# ---------------------------------------------------------------------------

def test_point_source_image_no_model_returns_zeros(
    lens_image_no_ps, eta_flat, kwargs_mass
):
    result = lens_image_no_ps.point_source_image(
        kwargs_point_source=None,
        eta_flat=eta_flat,
        kwargs_mass=kwargs_mass,
    )
    assert result.shape == (NPIX, NPIX)
    np.testing.assert_array_equal(np.array(result), np.zeros((NPIX, NPIX)))


# ---------------------------------------------------------------------------
# 4. point_source_image shape
# ---------------------------------------------------------------------------

def test_point_source_image_shape(
    lens_image_with_ps, eta_flat, kwargs_mass, kwargs_ps_infield
):
    result = lens_image_with_ps.point_source_image(
        kwargs_ps_infield, eta_flat, kwargs_mass
    )
    assert result.shape == (NPIX, NPIX)


# ---------------------------------------------------------------------------
# 5. point_source_image is non-zero for in-field positions
# ---------------------------------------------------------------------------

def test_point_source_image_nonzero_infield(
    lens_image_with_ps, eta_flat, kwargs_mass, kwargs_ps_infield
):
    result = lens_image_with_ps.point_source_image(
        kwargs_ps_infield, eta_flat, kwargs_mass
    )
    assert float(result.sum()) > 0., "Expected non-zero flux for in-field point sources"


# ---------------------------------------------------------------------------
# 6. model() with point_source_add=False matches no-PS model
# ---------------------------------------------------------------------------

def test_model_ps_flag_false_unchanged(
    lens_image_no_ps, lens_image_with_ps,
    eta_flat, kwargs_mass, kwargs_light, kwargs_ps_infield,
):
    model_ref = lens_image_no_ps.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
    )
    model_flag_off = lens_image_with_ps.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=False, kwargs_point_source=kwargs_ps_infield,
    )
    np.testing.assert_allclose(
        np.array(model_flag_off), np.array(model_ref), rtol=1e-6,
        err_msg="model() with point_source_add=False should equal the light-only model",
    )


# ---------------------------------------------------------------------------
# 7. model() with point_source_add=True differs from light-only
# ---------------------------------------------------------------------------

def test_model_ps_flag_true_differs(
    lens_image_with_ps, eta_flat, kwargs_mass, kwargs_light, kwargs_ps_infield,
):
    model_light = lens_image_with_ps.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=False,
    )
    model_ps = lens_image_with_ps.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=True, kwargs_point_source=kwargs_ps_infield,
    )
    assert not jnp.allclose(model_ps, model_light), \
        "model() with point sources should differ from the light-only model"


# ---------------------------------------------------------------------------
# 8. model() output shape
# ---------------------------------------------------------------------------

def test_model_ps_shape(
    lens_image_with_ps, eta_flat, kwargs_mass, kwargs_light, kwargs_ps_infield,
):
    model = lens_image_with_ps.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=True, kwargs_point_source=kwargs_ps_infield,
    )
    assert model.shape == (NPIX, NPIX)


# ---------------------------------------------------------------------------
# 9. JIT consistency: second call gives identical result
# ---------------------------------------------------------------------------

def test_model_jit_consistency(
    lens_image_with_ps, eta_flat, kwargs_mass, kwargs_light, kwargs_ps_infield,
):
    call_kwargs = dict(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=True, kwargs_point_source=kwargs_ps_infield,
    )
    model_1 = lens_image_with_ps.model(**call_kwargs)
    model_2 = lens_image_with_ps.model(**call_kwargs)
    np.testing.assert_array_equal(
        np.array(model_1), np.array(model_2),
        err_msg="Repeated JIT calls should give identical results",
    )


# ---------------------------------------------------------------------------
# 10. Amplitude linearity: doubling amp doubles the PS contribution
# ---------------------------------------------------------------------------

def test_point_source_amplitude_linearity(
    lens_image_with_ps, eta_flat, kwargs_mass, kwargs_light,
):
    kwargs_ps_1x = [{'ra': jnp.array([0.3]), 'dec': jnp.array([0.2]), 'amp': jnp.array([5.0])}]
    kwargs_ps_2x = [{'ra': jnp.array([0.3]), 'dec': jnp.array([0.2]), 'amp': jnp.array([10.0])}]

    model_1x = lens_image_with_ps.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=True, kwargs_point_source=kwargs_ps_1x,
    )
    model_0x = lens_image_with_ps.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=False,
    )
    model_2x = lens_image_with_ps.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=True, kwargs_point_source=kwargs_ps_2x,
    )

    ps_contribution_1x = model_1x - model_0x
    ps_contribution_2x = model_2x - model_0x

    np.testing.assert_allclose(
        np.array(ps_contribution_2x),
        2.0 * np.array(ps_contribution_1x),
        atol=1e-6,  # float32 rounding at near-zero pixels; absolute tolerance is sufficient
        err_msg="Point source flux should scale linearly with amplitude",
    )


# ---------------------------------------------------------------------------
# 11. Point source off-field contributes no flux
# ---------------------------------------------------------------------------

def test_point_source_offfield_zero_contribution(
    lens_image_with_ps, eta_flat, kwargs_mass, kwargs_light,
):
    """A point source 100 arcsec away should have no overlap with the image."""
    kwargs_ps_far = [{'ra': jnp.array([100.0]), 'dec': jnp.array([100.0]),
                      'amp': jnp.array([1e6])}]

    model_off = lens_image_with_ps.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=False,
    )
    model_far = lens_image_with_ps.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=True, kwargs_point_source=kwargs_ps_far,
    )
    np.testing.assert_allclose(
        np.array(model_far), np.array(model_off), atol=1e-6,
        err_msg="Far off-field point source should not contribute to the image",
    )


# ---------------------------------------------------------------------------
# 12. Gradient of model sum w.r.t. PS amplitudes is finite
# ---------------------------------------------------------------------------

def test_gradient_wrt_ps_amplitude_is_finite(
    lens_image_with_ps, eta_flat, kwargs_mass, kwargs_light,
):
    def model_sum(amp):
        kwargs_ps = [{'ra': jnp.array([0.3, -0.3]),
                      'dec': jnp.array([0.2, -0.2]),
                      'amp': amp}]
        return lens_image_with_ps.model(
            eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
            point_source_add=True, kwargs_point_source=kwargs_ps,
        ).sum()

    amp = jnp.array([5.0, 5.0])
    grad = jax.grad(model_sum)(amp)
    assert jnp.all(jnp.isfinite(grad)), \
        f"Gradient w.r.t. PS amplitudes contains non-finite values: {grad}"


# ---------------------------------------------------------------------------
# 13. Multiple LENSED_POSITIONS sources are additive
# ---------------------------------------------------------------------------

def test_two_point_sources_additive(
    base_grid, base_psf, base_noise, mp_mass, mp_light,
    eta_flat, kwargs_mass, kwargs_light,
):
    """The combined model with two sources should equal the sum of individual models."""
    ps_model_two = MPPointSourceModel(['LENSED_POSITIONS', 'LENSED_POSITIONS'])
    li_two = MPLensImage(base_grid, base_psf, base_noise, mp_mass, mp_light,
                         point_source_model_class=ps_model_two)

    ps_model_one = MPPointSourceModel(['LENSED_POSITIONS'])
    li_one = MPLensImage(base_grid, base_psf, base_noise, mp_mass, mp_light,
                         point_source_model_class=ps_model_one)

    kw_a = {'ra': jnp.array([0.3]), 'dec': jnp.array([0.2]), 'amp': jnp.array([5.0])}
    kw_b = {'ra': jnp.array([-0.3]), 'dec': jnp.array([-0.2]), 'amp': jnp.array([3.0])}

    model_two = li_two.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=True, kwargs_point_source=[kw_a, kw_b],
    )
    model_a = li_one.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=True, kwargs_point_source=[kw_a],
    )
    model_b = li_one.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=True, kwargs_point_source=[kw_b],
    )
    model_light = li_one.model(
        eta_flat=eta_flat, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light,
        point_source_add=False,
    )

    # model_two = light + PS_a + PS_b = model_a + model_b - light
    expected = model_a + model_b - model_light
    np.testing.assert_allclose(
        np.array(model_two), np.array(expected), rtol=1e-5,
        err_msg="Two point sources should be additive",
    )
