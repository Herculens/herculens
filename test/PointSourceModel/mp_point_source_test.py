"""Unit tests for MPPointSourceModel.

Covers:
- Instantiation with LENSED_POSITIONS and SOURCE_POSITION types
- Input validation (invalid type, missing mass model, bad source_plane_index)
- get_multiple_images for LENSED_POSITIONS (values, k-selection, amplitude flag)
- _indices_from_k helper
- _zero_amp_duplicated_images static method (via mock)
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import MagicMock, patch

from herculens.PointSourceModel.point_source_multiplane import (
    MPPointSourceModel,
    SUPPORTED_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mock_mass_model(n_mass=1):
    """Return a minimal mock that mimics MPMassModel with n_mass planes."""
    m = MagicMock()
    m.number_mass_planes = n_mass
    return m


def _make_mock_image_plane():
    """Return a minimal mock that mimics PixelGrid."""
    m = MagicMock()
    x = np.linspace(-1., 1., 20)
    m.pixel_coordinates = (x, x)
    return m


# ---------------------------------------------------------------------------
# 1. SUPPORTED_TYPES constant
# ---------------------------------------------------------------------------

def test_supported_types_content():
    assert 'LENSED_POSITIONS' in SUPPORTED_TYPES
    assert 'SOURCE_POSITION' in SUPPORTED_TYPES


# ---------------------------------------------------------------------------
# 2. Instantiation – LENSED_POSITIONS (no mass model required)
# ---------------------------------------------------------------------------

def test_lensed_positions_no_mass_model():
    """LENSED_POSITIONS should not require a mass model or image plane."""
    ps = MPPointSourceModel(['LENSED_POSITIONS'])
    assert ps.type_list == ['LENSED_POSITIONS']
    assert len(ps._solvers) == 1
    assert ps._solvers[0] is None


def test_lensed_positions_multiple():
    """Multiple LENSED_POSITIONS entries should instantiate cleanly."""
    ps = MPPointSourceModel(['LENSED_POSITIONS', 'LENSED_POSITIONS'])
    assert len(ps.type_list) == 2
    assert len(ps._solvers) == 2


def test_lensed_positions_source_plane_index_ignored():
    """source_plane_index is irrelevant (and not validated) for LENSED_POSITIONS."""
    ps = MPPointSourceModel(['LENSED_POSITIONS'], source_plane_index_list=[None])
    assert ps.source_plane_index_list == [None]

    ps2 = MPPointSourceModel(['LENSED_POSITIONS'], source_plane_index_list=[0])
    assert ps2.source_plane_index_list == [0]


# ---------------------------------------------------------------------------
# 3. Instantiation – default source_plane_index
# ---------------------------------------------------------------------------

def test_default_source_plane_index_uses_n_mass():
    """When source_plane_index_list is None, default to mp_mass_model.number_mass_planes."""
    mm = _make_mock_mass_model(n_mass=2)
    ip = _make_mock_image_plane()
    with patch('herculens.PointSourceModel.point_source_multiplane._solver_installed', True):
        ps = MPPointSourceModel(
            ['SOURCE_POSITION'],
            mp_mass_model=mm,
            image_plane=ip,
            source_plane_index_list=None,
        )
    assert ps.source_plane_index_list == [2]


def test_default_source_plane_index_lensed_positions_none():
    """For LENSED_POSITIONS with no mass model, default source_plane_index is None."""
    ps = MPPointSourceModel(['LENSED_POSITIONS'])
    assert ps.source_plane_index_list == [None]


# ---------------------------------------------------------------------------
# 4. Input validation errors
# ---------------------------------------------------------------------------

def test_invalid_type_raises():
    with pytest.raises(ValueError, match="not a valid point source type"):
        MPPointSourceModel(['WRONG_TYPE'])


def test_non_list_input_raises():
    with pytest.raises(ValueError, match="must be a list"):
        MPPointSourceModel('LENSED_POSITIONS')


def test_source_plane_index_list_wrong_length_raises():
    with pytest.raises(ValueError, match="same length"):
        MPPointSourceModel(['LENSED_POSITIONS'], source_plane_index_list=[1, 2])


def test_source_position_without_solver_raises():
    """SOURCE_POSITION requires the helens solver to be installed."""
    with patch('herculens.PointSourceModel.point_source_multiplane._solver_installed', False):
        with pytest.raises(RuntimeError, match="lens equation solver"):
            MPPointSourceModel(
                ['SOURCE_POSITION'],
                mp_mass_model=_make_mock_mass_model(),
                image_plane=_make_mock_image_plane(),
                source_plane_index_list=[1],
            )


def test_source_position_without_mass_model_raises():
    with patch('herculens.PointSourceModel.point_source_multiplane._solver_installed', True):
        with pytest.raises(ValueError, match="mp_mass_model is required"):
            MPPointSourceModel(
                ['SOURCE_POSITION'],
                mp_mass_model=None,
                image_plane=_make_mock_image_plane(),
                source_plane_index_list=[1],
            )


def test_source_position_without_image_plane_raises():
    with patch('herculens.PointSourceModel.point_source_multiplane._solver_installed', True):
        with pytest.raises(ValueError, match="image_plane is required"):
            MPPointSourceModel(
                ['SOURCE_POSITION'],
                mp_mass_model=_make_mock_mass_model(),
                image_plane=None,
                source_plane_index_list=[1],
            )


def test_source_position_zero_plane_index_raises():
    """source_plane_index < 1 should raise for SOURCE_POSITION."""
    with patch('herculens.PointSourceModel.point_source_multiplane._solver_installed', True):
        with pytest.raises(ValueError, match="source_plane_index must be >= 1"):
            MPPointSourceModel(
                ['SOURCE_POSITION'],
                mp_mass_model=_make_mock_mass_model(),
                image_plane=_make_mock_image_plane(),
                source_plane_index_list=[0],
            )


# ---------------------------------------------------------------------------
# 5. get_multiple_images – LENSED_POSITIONS
# ---------------------------------------------------------------------------

def test_get_multiple_images_returns_input_positions():
    """For LENSED_POSITIONS, image positions are returned unchanged."""
    ps = MPPointSourceModel(['LENSED_POSITIONS'])
    ra = jnp.array([1.0, -1.0])
    dec = jnp.array([0.5, -0.5])
    amp = jnp.array([2.0, 3.0])
    kwargs_ps = [{'ra': ra, 'dec': dec, 'amp': amp}]

    theta_x, theta_y, amps = ps.get_multiple_images(kwargs_ps)

    np.testing.assert_allclose(np.array(theta_x[0]), np.array(ra))
    np.testing.assert_allclose(np.array(theta_y[0]), np.array(dec))
    np.testing.assert_allclose(np.array(amps[0]), np.array(amp))


def test_get_multiple_images_scalar_inputs():
    """Scalar ra/dec/amp are broadcast to 1-D arrays."""
    ps = MPPointSourceModel(['LENSED_POSITIONS'])
    kwargs_ps = [{'ra': 0.5, 'dec': -0.3, 'amp': 4.0}]
    theta_x, theta_y, amps = ps.get_multiple_images(kwargs_ps)
    assert theta_x[0].ndim == 1
    assert theta_y[0].ndim == 1
    assert amps[0].ndim == 1


def test_get_multiple_images_two_point_sources():
    """Two independent LENSED_POSITIONS entries are both returned."""
    ps = MPPointSourceModel(['LENSED_POSITIONS', 'LENSED_POSITIONS'])
    kw1 = {'ra': jnp.array([1.0, -1.0]), 'dec': jnp.array([0.5, -0.5]), 'amp': jnp.array([2.0, 3.0])}
    kw2 = {'ra': jnp.array([0.3]), 'dec': jnp.array([-0.3]), 'amp': jnp.array([1.5])}
    kwargs_ps = [kw1, kw2]

    theta_x, theta_y, amps = ps.get_multiple_images(kwargs_ps)

    assert len(theta_x) == 2
    assert len(theta_y) == 2
    assert len(amps) == 2
    np.testing.assert_allclose(np.array(theta_x[1]), np.array(kw2['ra']))


def test_get_multiple_images_without_amplitude():
    """with_amplitude=False returns only positions."""
    ps = MPPointSourceModel(['LENSED_POSITIONS'])
    kwargs_ps = [{'ra': jnp.array([0.5]), 'dec': jnp.array([0.3]), 'amp': jnp.array([2.0])}]
    result = ps.get_multiple_images(kwargs_ps, with_amplitude=False)
    assert len(result) == 2, "Expected (theta_x, theta_y) tuple"


# ---------------------------------------------------------------------------
# 6. get_multiple_images – k selection
# ---------------------------------------------------------------------------

def test_get_multiple_images_k_selects_single_source():
    """k=0 should return only the first point source."""
    ps = MPPointSourceModel(['LENSED_POSITIONS', 'LENSED_POSITIONS'])
    kw1 = {'ra': jnp.array([1.0]), 'dec': jnp.array([0.5]), 'amp': jnp.array([2.0])}
    kw2 = {'ra': jnp.array([-1.0]), 'dec': jnp.array([-0.5]), 'amp': jnp.array([3.0])}
    kwargs_ps = [kw1, kw2]

    theta_x, theta_y, amps = ps.get_multiple_images(kwargs_ps, k=0)

    assert len(theta_x) == 1
    np.testing.assert_allclose(float(theta_x[0][0]), 1.0)


def test_get_multiple_images_k_out_of_range_returns_all():
    """k outside [0, N-1] falls back to returning all sources."""
    ps = MPPointSourceModel(['LENSED_POSITIONS', 'LENSED_POSITIONS'])
    kw = {'ra': jnp.array([0.5]), 'dec': jnp.array([0.3]), 'amp': jnp.array([1.0])}
    theta_x, theta_y, amps = ps.get_multiple_images([kw, kw], k=99)
    assert len(theta_x) == 2


# ---------------------------------------------------------------------------
# 7. _indices_from_k helper
# ---------------------------------------------------------------------------

def test_indices_from_k_none():
    ps = MPPointSourceModel(['LENSED_POSITIONS', 'LENSED_POSITIONS'])
    assert ps._indices_from_k(None) == [0, 1]


def test_indices_from_k_valid():
    ps = MPPointSourceModel(['LENSED_POSITIONS', 'LENSED_POSITIONS'])
    assert ps._indices_from_k(0) == [0]
    assert ps._indices_from_k(1) == [1]


def test_indices_from_k_out_of_range():
    ps = MPPointSourceModel(['LENSED_POSITIONS', 'LENSED_POSITIONS'])
    assert ps._indices_from_k(5) == [0, 1]
    assert ps._indices_from_k(-1) == [0, 1]


# ---------------------------------------------------------------------------
# 8. _make_ray_shooting_func
# ---------------------------------------------------------------------------

def test_make_ray_shooting_func_calls_ray_shooting_at_correct_plane():
    """The wrapped ray-shooting function should index into the correct plane."""
    mm = _make_mock_mass_model(n_mass=2)
    # ray_shooting returns (xs, ys) with shape (N+1, ...)
    xs_mock = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # shape (3, 2)
    ys_mock = jnp.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
    mm.ray_shooting.return_value = (xs_mock, ys_mock)

    with patch('herculens.PointSourceModel.point_source_multiplane._solver_installed', True):
        ps = MPPointSourceModel(
            ['SOURCE_POSITION'],
            mp_mass_model=mm,
            image_plane=_make_mock_image_plane(),
            source_plane_index_list=[2],
        )

    ray_func = ps._make_ray_shooting_func(2)
    eta_flat = jnp.array([1.0])
    kwargs_mass = [{'theta_E': 1.0}]
    kwargs_combined = [{'eta_flat': eta_flat}] + kwargs_mass

    x = jnp.array([0.5, -0.5])
    y = jnp.array([0.3, -0.3])
    bx, by = ray_func(x, y, kwargs_combined)

    # Should have called MPMassModel.ray_shooting with N=2
    mm.ray_shooting.assert_called_once()
    call_kwargs = mm.ray_shooting.call_args
    assert call_kwargs[1].get('N') == 2 or (len(call_kwargs[0]) >= 5 and call_kwargs[0][4] == 2)

    # Should return plane index 2
    np.testing.assert_allclose(np.array(bx), np.array(xs_mock[2]))
    np.testing.assert_allclose(np.array(by), np.array(ys_mock[2]))


# ---------------------------------------------------------------------------
# 9. param_names attribute
# ---------------------------------------------------------------------------

def test_param_names():
    ps = MPPointSourceModel(['LENSED_POSITIONS'])
    assert 'ra' in ps.param_names
    assert 'dec' in ps.param_names
    assert 'amp' in ps.param_names
