# Utility functions
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal'


import copy
import numpy as np
import jax.numpy as jnp
from scipy.ndimage import morphology
from scipy.spatial import Delaunay
from scipy import ndimage
from skimage import measure
from jax import config
from functools import partial

from herculens.LensImage.lens_image import LensImage, LensImage3D
from herculens.LensImage.lens_image_future import LensImage as LensImageFuture
from herculens.LensImage.lens_image_multiplane import MPLensImage
from herculens.LensImage.lensing_operator import LensingOperator


def critical_lines_caustics(lens_image, kwargs_mass, eta_flat=None, supersampling=5, 
                            return_lens_centers=False, k_plane=None):
    if config.read('jax_enable_x64') is not True:
        print("WARNING: JAX's 'jax_enable_x64' is not enabled; "
              "computation of critical lines and caustics might be inaccurate.")
    if isinstance(lens_image, (LensImage, LensImage3D)):
        mass_model = lens_image.MassModel
        inv_mag_fn = partial(mass_model.inverse_magnification, kwargs=kwargs_mass)
        ray_shooting_fn = partial(mass_model.ray_shooting, kwargs=kwargs_mass)
        multiplane_mode = False
    elif isinstance(lens_image, MPLensImage):
        if k_plane is None:
            # we take the furthest plane as the default
            k_plane = lens_image.MPLightModel.number_light_planes
        mass_model = lens_image.MPMassModel
        if eta_flat is None:
            raise ValueError("The `eta_flat` parameter must be provided when using MPLensImage.")
        inv_mag_fn = lambda x, y: mass_model.inverse_magnification(x, y, eta_flat=eta_flat, kwargs=kwargs_mass)[k_plane]
        ray_shooting_fn = lambda x, y: [rs[k_plane] for rs in mass_model.ray_shooting(x, y, eta_flat=eta_flat, kwargs=kwargs_mass)]
        multiplane_mode = True
    else:
        raise TypeError("lens_image must be an instance of LensImage, LensImage3D or MPLensImage.")
    
    # evaluate the inverse magnification
    grid = lens_image.Grid.create_model_grid(pixel_scale_factor=1./supersampling)
    x_grid_img, y_grid_img = grid.pixel_coordinates
    inv_mag_tot = inv_mag_fn(x_grid_img, y_grid_img)
    inv_mag_tot = np.array(inv_mag_tot, dtype=np.float64)

    # find contours corresponding to infinite magnification
    contours = measure.find_contours(inv_mag_tot, 0.)

    crit_lines, caustics = [], []
    for contour in contours:
        # extract the lines
        cline_x, cline_y = contour[:, 1], contour[:, 0]
        # convert to model coordinates
        cline_x, cline_y = grid.map_pix2coord(cline_x, cline_y)
        crit_lines.append((np.array(cline_x), np.array(cline_y)))
        # find corresponding caustics through ray shooting
        caust_x, caust_y = ray_shooting_fn(cline_x, cline_y)
        caustics.append((np.array(caust_x), np.array(caust_y)))

    # can also returns the lens components centroids for convenience
    if return_lens_centers:
        cxs, cys = [], []
        if multiplane_mode is True:
            k_main_lens = 0
            for kw in kwargs_mass[k_main_lens]:
                if 'center_x' in kw:
                    cxs.append(kw['center_x'])
                    cys.append(kw['center_y'])
        else:
            for kw in kwargs_mass:
                if 'center_x' in kw:
                    cxs.append(kw['center_x'])
                    cys.append(kw['center_y'])
        return crit_lines, caustics, (np.array(cxs), np.array(cys))
    return crit_lines, caustics


def shear_deflection_field(lens_image, kwargs_lens, num_pixels=20, shear_type='external_single'):
    """Return a coarse shear field for plotting.

    Parameters
    ----------
    shear_type : str
        'external_single' (default) — shear from the first SHEAR/SHEAR_GAMMA_PSI profile only.
        'external_all' — sum of shear from all SHEAR/SHEAR_GAMMA_PSI and PixelatedFixed profiles.
        'total' — total shear from all mass profiles (via MassModel.gamma).
    """
    if shear_type == 'total':
        return total_shear_deflection_field(lens_image, kwargs_lens, num_pixels=num_pixels)
    from herculens.MassModel.Profiles.shear import Shear, ShearGammaPsi
    from herculens.MassModel.Profiles.pixelated import PixelatedFixed
    profile_list = lens_image.MassModel.profile_type_list
    if shear_type == 'external_all':
        selected_indices = [
            i for i, p in enumerate(profile_list)
            if (p in ('SHEAR', 'SHEAR_GAMMA_PSI') or
                isinstance(p, (Shear, ShearGammaPsi, PixelatedFixed)))
        ]
        if not selected_indices:
            return None
        grid = lens_image.Grid.create_model_grid(num_pixels=num_pixels)
        x_grid_img, y_grid_img = grid.pixel_coordinates
        gamma_x = np.zeros_like(x_grid_img)
        gamma_y = np.zeros_like(y_grid_img)
        for k in selected_indices:
            gx_k, gy_k = lens_image.MassModel.gamma(x_grid_img, y_grid_img, kwargs_lens, k=k)
            gamma_x = gamma_x + gx_k
            gamma_y = gamma_y + gy_k
        return (np.array(x_grid_img), np.array(y_grid_img), gamma_x, gamma_y)
    # 'external_single': find the first SHEAR/SHEAR_GAMMA_PSI profile
    shear_idx, shear_profile_type = None, None
    num_profiles = len(profile_list)
    assert num_profiles == len(kwargs_lens)
    for i in range(num_profiles):
        if (profile_list[i] == 'SHEAR' or isinstance(profile_list[i], Shear)):
            shear_idx = i
            shear_profile_type = 'SHEAR'
            break
        elif (profile_list[i] == 'SHEAR_GAMMA_PSI' or isinstance(profile_list[i], ShearGammaPsi)):
            shear_idx = i
            shear_profile_type = 'SHEAR_GAMMA_PSI'
            break
    if shear_idx is None:
        return None
    if shear_profile_type == 'SHEAR_GAMMA_PSI':
        phi_ext, gamma_ext = kwargs_lens[shear_idx]['psi_ext'], kwargs_lens[shear_idx]['gamma_ext']
    else:
        # imports are here to avoid issues with circular imports
        from herculens.Util.param_util import shear_cartesian2polar_numpy
        phi_ext, gamma_ext = shear_cartesian2polar_numpy(
            kwargs_lens[shear_idx]['gamma1'],
            kwargs_lens[shear_idx]['gamma2'],
        )
    grid = lens_image.Grid.create_model_grid(num_pixels=num_pixels)
    x_grid_img, y_grid_img = grid.pixel_coordinates
    gamma_x = gamma_ext * np.cos(phi_ext)
    gamma_y = gamma_ext * np.sin(phi_ext)
    if gamma_x.size == 1:
        gamma_x = np.full_like(x_grid_img, float(gamma_x))
        gamma_y = np.full_like(y_grid_img, float(gamma_y))
    return (np.array(x_grid_img), np.array(y_grid_img), gamma_x, gamma_y)


def total_shear_deflection_field(lens_image, kwargs_mass, eta_flat=None, num_pixels=20, k_plane=-1):
    if isinstance(lens_image, (LensImage, LensImage3D)):
        mass_model = lens_image.MassModel
        gamma_fn = partial(mass_model.gamma, kwargs=kwargs_mass)
        multiplane_mode = False
    elif isinstance(lens_image, MPLensImage):
        mass_model = lens_image.MPMassModel
        if eta_flat is None:
            raise ValueError("The `eta_flat` parameter must be provided when using MPLensImage.")
        gamma_fn = partial(mass_model.gamma, eta_flat=eta_flat, kwargs=kwargs_mass)
        multiplane_mode = True
    else:
        raise TypeError("lens_image must be an instance of LensImage, LensImage3D or MPLensImage.")
    grid = lens_image.Grid.create_model_grid(num_pixels=num_pixels)
    x_grid_img, y_grid_img = grid.pixel_coordinates
    gamma_x, gamma_y = gamma_fn(x_grid_img, y_grid_img)
    if multiplane_mode is True:
        gamma_x = gamma_x[k_plane]
        gamma_y = gamma_y[k_plane]
    if gamma_x.size == 1:
        gamma_x = np.full_like(x_grid_img, float(gamma_x))
        gamma_y = np.full_like(y_grid_img, float(gamma_y))
    return (np.array(x_grid_img), np.array(y_grid_img), gamma_x, gamma_y)


def _get_parameters(parameters):
    # For backward compatibility, we need to handle both the old (legacy) Parameters class
    # and the simpler way to provide parameters (which is a simple dictionary)
    if isinstance(parameters, dict):
        kwargs_params = parameters
    else:
        from herculens.Inference.legacy.parameters import Parameters as LegacyParameters
        if not isinstance(parameters, LegacyParameters):
            raise TypeError("The 'parameters' argument must be either a dictionary or a Parameters instance.")
        kwargs_params = copy.deepcopy(parameters.current_values(as_kwargs=True))
    return kwargs_params


def mask_from_source_area(lens_image, parameters):
    src_idx = lens_image.SourceModel.pixelated_index
    kwargs_param_mask = _get_parameters(parameters)
    pixels = kwargs_param_mask['kwargs_source'][src_idx]['pixels']
    pixels = np.zeros_like(pixels)
    pixels[3:-3, 3:-3] = 1.  # so the source plane is ones except in a small margin all around
    kwargs_param_mask['kwargs_source'][src_idx]['pixels'] = jnp.array(pixels)
    model_mask = lens_image.source_surface_brightness(kwargs_param_mask['kwargs_source'],
                                                      kwargs_lens=kwargs_param_mask['kwargs_lens'],
                                                      unconvolved=True, de_lensed=False)
    model_mask = np.array(model_mask)
    model_mask[model_mask < 0.1] = 0.
    model_mask[model_mask >= 0.1] = 1.
    model_mask = morphology.binary_opening(model_mask, iterations=10)
    model_mask = morphology.binary_dilation(model_mask, iterations=3)
    return model_mask


def mask_from_lensed_source(lens_image, parameters=None, source_model=None,
                            threshold=0.1, smoothing=0, kwargs_numerics=None):
    # imports are here to avoid issues with circular imports
    from herculens.LensImage.lens_image import LensImage
    from herculens.LightModel.light_model import LightModel

    if parameters is None and source_model is None:
        raise ValueError("You must provide a source model "
                         "if no parameters are provided.")
    if parameters is not None:
        kwargs_param = _get_parameters(parameters)
        source_model = lens_image.source_surface_brightness(kwargs_param['kwargs_source'], 
                                                            de_lensed=True, unconvolved=True)
    source_model = np.array(source_model)
    if smoothing > 0:
        source_model = ndimage.gaussian_filter(source_model, sigma=smoothing)
    binary_source = source_model / source_model.max()
    binary_source[binary_source < threshold] = 0.
    binary_source[binary_source >= threshold] = 1.
    grid = copy.deepcopy(lens_image.Grid)
    lens_image_pixel = LensImage(grid, lens_image.PSF, 
                                 noise_class=lens_image.Noise,
                                 lens_mass_model_class=lens_image.MassModel,
                                 source_model_class=LightModel(['PIXELATED']),
                                 lens_light_model_class=lens_image.LensLightModel,
                                 kwargs_numerics=kwargs_numerics)
    kwargs_param_mask = copy.deepcopy(kwargs_param)
    kwargs_param_mask['kwargs_source'] = [{'pixels': jnp.array(binary_source)}]
    model_mask = lens_image_pixel.source_surface_brightness(kwargs_param_mask['kwargs_source'], 
                                                      kwargs_lens=kwargs_param_mask['kwargs_lens'],
                                                      unconvolved=True, de_lensed=False)
    model_mask = np.array(model_mask)
    model_mask[model_mask < 0.1] = 0.
    model_mask[model_mask >= 0.1] = 1.
    #model_mask = morphology.binary_opening(model_mask, iterations=10)
    #model_mask = morphology.binary_dilation(model_mask, iterations=3)
    return model_mask, binary_source


def pixelated_region_from_sersic(kwargs_sersic, force_square=False, use_major_axis=False,
                                 min_width=1.0, min_height=1.0, scaling=1.0):
    # imports are here to avoid issues with circular imports
    from herculens.Util import param_util

    # TODO: support arbitrary smooth source profile
    c_x = kwargs_sersic['center_x']
    c_y = kwargs_sersic['center_y']
    r_eff = kwargs_sersic['R_sersic']
    if 'e1' in kwargs_sersic:
        e1 = kwargs_sersic['e1']
        e2 = kwargs_sersic['e2']
    else:
        e1 = e2 = 0.
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    a = r_eff / np.sqrt(q)  # semi-major axis
    r = r_eff * np.sqrt(q)  # product average radius
    print(f"r_eff={r_eff:.2f}, r={r:.2f}, a={a:.2f}")
    diameter = 2*a if use_major_axis else 2*r
    width = diameter * np.abs(np.cos(phi)) * scaling
    height = diameter * np.abs(np.sin(phi)) * scaling
    width = max(min_width, width)
    height = max(min_height, height)
    if force_square is True:
        width = max(width, height)
        height = width
    # the following dict is consistent with arguments of PixelGrid.create_model_grid()
    kwargs_pixelated_grid = {
        'grid_center': (float(c_x), float(c_y)),
        'grid_shape': (float(width), float(height)),
    }
    return kwargs_pixelated_grid


def pixelated_region_from_arc_mask(arc_mask, image_grid, mass_model, mass_params):
    # We first design a hypothetical source plane with sufficiently high resolution
    nx, ny = image_grid.num_pixel_axes
    nx_src, ny_src = (nx//3, ny//3)
    high_res_source_grid = image_grid.create_model_grid(pixel_scale_factor=0.5, grid_shape=(nx_src, ny_src))
    
    # Then build the lensing operator based on the image/source grids and mass model
    lensing_op = LensingOperator(mass_model, image_grid, high_res_source_grid)
    lensing_op.compute_mapping(kwargs_lens=mass_params)
    
    # De-lens the arc mask
    arc_mask_source = np.array(lensing_op.image2source_2d(arc_mask))
    arc_mask_source = np.where(arc_mask_source < 0.1, 0., 1.)  # only contains 0 and 1 now

    # Find the minimum bounding box that encloses the mask in source plane (square here)
    # - get the extrema 
    rows = np.max(arc_mask_source, axis=0)
    print(rows)
    row_low, row_high = np.argmax(rows), np.argmax(rows[::-1])
    print("R", row_low, row_high)
    cols = np.max(arc_mask_source, axis=1)
    print(cols)
    col_low, col_high = np.argmax(cols), np.argmax(cols[::-1])
    print("C", col_low, col_high)
    # - get the pixel coordinates corresponding to these extrema
    x_low, y_low = high_res_source_grid.map_pix2coord(row_low, col_low)
    x_high, y_high = high_res_source_grid.map_pix2coord(row_high, col_high)
    print(x_low, y_low)
    print(x_high, y_high)

    # Get the grid parameters
    new_grid_shape = (abs(x_high - x_low), abs(y_high - y_low))
    new_grid_center = (0.5*(x_low+x_high), 0.5*(y_low+y_high))
    kwargs_pixelated_grid = {
        'grid_center': new_grid_center,
        'grid_shape': new_grid_shape,
    }

    # Create the new, reduced source plane grid
    return kwargs_pixelated_grid


def source_plane_mask_from_arc_mask(
        lens_image, 
        kwargs_lens,
        alpha=1,
        dilation_iterations=0,
        source_arc_mask=None,
    ):
    """
    Compute a boolean mask in the source plane identifying pixels constrained
    by the image-plane arc mask, using an alpha-shape approach based on
    Delaunay triangulation (no external geometry packages required).

    The mask includes interior regions not directly mapped from the arc but
    enclosed by the traced boundary — these pixels are regularised by the
    source prior (e.g. CorrelatedField) rather than excluded.

    Parameters
    ----------
    lens_image : LensImage
        Herculens LensImage instance. Must have source_arc_mask and a pixelated
        source model with an associated pixel_grid.
    mass_params : list of dict
        Lens mass model kwargs.
    alpha : float or None
        Circumradius threshold in arcsec. Delaunay triangles with circumradius
        <= alpha are accepted as part of the alpha shape. Increase to fill larger interior gaps;
        decrease to tighten the mask around the arc traces.
    dilation_iterations : int
        Number of binary dilation iterations applied after the triangle test
        (default 0 — usually not needed since the polygon approach already
        fills interior gaps).
    source_arc_mask : 2D bool array, optional
        If provided, use this arc mask instead of the one in lens_image.source_arc_mask.

    Returns
    -------
    source_mask : 2D bool array
        Shape matches lens_image.SourceModel.pixel_grid.num_pixel_axes.
        True for source pixels that fall inside the alpha-shape polygon.
    """
    if lens_image.source_arc_mask is None and source_arc_mask is None:
        raise ValueError("LensImage has no source_arc_mask set.")
    if lens_image.SourceModel.pixel_grid is None:
        raise ValueError("LensImage has no pixelated source grid (source model is not pixelated).")

    # --- Step 1: ray-shoot arc mask pixels to source plane ---
    if source_arc_mask is None:
        source_arc_mask = lens_image.source_arc_mask
    arc_mask = np.array(source_arc_mask).astype(bool)
    x_img, y_img = lens_image.Grid.pixel_coordinates        # (ny, nx)
    x_pts = x_img[arc_mask]
    y_pts = y_img[arc_mask]

    beta_x, beta_y = lens_image.MassModel.ray_shooting(x_pts, y_pts, kwargs_lens)
    pts = np.column_stack([np.array(beta_x), np.array(beta_y)])  # (N, 2)

    # --- Step 2: Delaunay triangulation + circumradius filtering ---
    tri = Delaunay(pts)
    simplices = tri.simplices                               # (M, 3)

    A = pts[simplices[:, 0]]                               # (M, 2)
    B = pts[simplices[:, 1]]
    C = pts[simplices[:, 2]]
    a = np.linalg.norm(B - C, axis=1)
    b = np.linalg.norm(A - C, axis=1)
    c = np.linalg.norm(A - B, axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        circumradius = np.where(area > 0, (a * b * c) / (4.0 * area), np.inf)

    accepted = simplices[circumradius <= alpha]             # (K, 3)
    tri_pts = pts[accepted]                                 # (K, 3, 2)

    # --- Step 3: vectorized point-in-triangle test for each source pixel ---
    # Use get_source_coordinates so that the adaptive grid case (adaptive_grid=True)
    # correctly returns the dynamically computed source extent, rather than the
    # placeholder grid that spans the full image FOV.
    x_src, y_src, _ = lens_image.get_source_coordinates(kwargs_lens)
    x_src = np.array(x_src)
    y_src = np.array(y_src)
    test_pts = np.column_stack([x_src.ravel(), y_src.ravel()])  # (P, 2)

    source_mask_flat = np.zeros(len(test_pts), dtype=bool)
    batch_size = 64
    for i in range(0, len(tri_pts), batch_size):
        Ab = tri_pts[i:i + batch_size, 0, :]               # (b, 2)
        Bb = tri_pts[i:i + batch_size, 1, :]
        Cb = tri_pts[i:i + batch_size, 2, :]
        v0 = Cb - Ab                                        # (b, 2)
        v1 = Bb - Ab
        v2 = test_pts[:, None, :] - Ab[None, :, :]         # (P, b, 2)
        dot00 = (v0 * v0).sum(-1)                           # (b,)
        dot01 = (v0 * v1).sum(-1)
        dot11 = (v1 * v1).sum(-1)
        dot02 = (v2 * v0[None]).sum(-1)                     # (P, b)
        dot12 = (v2 * v1[None]).sum(-1)
        denom = dot00 * dot11 - dot01 ** 2                  # (b,)
        inv = np.where(np.abs(denom) > 1e-30,
                       1.0 / np.maximum(np.abs(denom), 1e-30), 0.0)
        u = (dot11[None] * dot02 - dot01[None] * dot12) * inv[None]
        v = (dot00[None] * dot12 - dot01[None] * dot02) * inv[None]
        source_mask_flat |= ((u >= 0) & (v >= 0) & (u + v <= 1)).any(axis=1)

    source_mask = source_mask_flat.reshape(x_src.shape)

    if dilation_iterations > 0:
        source_mask = morphology.binary_dilation(source_mask, iterations=dilation_iterations)

    return source_mask


def border_mask(image, border_fraction=0.1):
    """Returns a mask (2D binary array) filled with zeros on each edge
    with width equal to `border_fraction` times the shape of the input image.
    The remaining values are ones.

    Parameters
    ----------
    image : np.ndarray
        Input 2D array
    border_fraction : float, optional
        Fraction of the shape along a given dimension that should be masked,
        by default 0.1 (i.e. 10% of the width/height on each side).
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    if border_fraction < 0 or border_fraction > 0.5:
        raise ValueError("border_fraction must be between 0 and 0.5.")
    ny, nx = image.shape
    mask = np.ones_like(image, dtype=float)
    x_border = int(border_fraction * nx / 2.)
    y_border = int(border_fraction * ny / 2.)
    mask[:, :x_border] = 0.0
    mask[:, -x_border:] = 0.0
    mask[:y_border, :] = 0.0
    mask[-y_border:, :] = 0.0
    return mask


def estimate_model_covariance(lens_image, parameters, samples, return_cross_covariance=False):
    model_samples = []
    for sample in samples:
        kwargs_sample = parameters.args2kwargs(sample)
        model_sample = lens_image.model(**kwargs_sample)
        model_map_shape = model_sample.shape
        model_samples.append(model_sample.flatten())
    model_samples = np.array(model_samples)
    
    # variance map (as a 2D image)
    model_var_map = np.var(model_samples, axis=0).reshape(*model_map_shape)
    
    # full (auto) covariance matrix from the model samples
    model_cov = np.cov(model_samples, rowvar=False)

    # computation of the cross-covariance matrix from model and data samples
    if return_cross_covariance:
        data_mean_proxy = lens_image.model(**parameters.best_fit_values(as_kwargs=True))   # best-fit model as a proxy for the unknown data 'mean'
        data_var = lens_image.Noise.C_D_model(data_mean_proxy, force_recompute=True)
        data_mean_proxy = data_mean_proxy.flatten()
        data_cov = np.diag(data_var.flatten())
        data_samples = draw_samples_from_covariance(data_mean_proxy, data_cov, num_samples=len(model_samples))
        data_vector  = np.mean(data_samples - data_samples.mean(axis=0), axis=0)
        model_vector = np.mean(model_samples - model_samples.mean(axis=0), axis=0)
        data_model_cross_cov = np.outer(data_vector, model_vector)
        return model_var_map, model_cov, data_model_cross_cov
    else:
        return model_var_map, model_cov


def draw_samples_from_covariance(mean, covariance, num_samples=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.multivariate_normal(mean, covariance, size=num_samples)
    return samples

