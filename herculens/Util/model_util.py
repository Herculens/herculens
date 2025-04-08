# Utility functions
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal'


import copy
import numpy as np
import jax.numpy as jnp
from scipy.ndimage import morphology
from scipy import ndimage
from skimage import measure
from jax import config
from functools import partial

from herculens.LensImage.lens_image import LensImage, LensImage3D
from herculens.LensImage.lens_image_multiplane import MPLensImage
from herculens.LensImage.lensing_operator import LensingOperator


def critical_lines_caustics(lens_image, kwargs_mass, eta_flat=None, supersampling=5, 
                            return_lens_centers=False, k_plane=None):
    if config.read('jax_enable_x64') is not True:
        print("WARNING: JAX's 'jax_enable_x64' is not enabled; "
              "computation of critical lines and caustics might be inaccurate.")
    if isinstance(lens_image, (LensImage, LensImage3D)):
        mass_model = lens_image.MassModel
        inv_mag_fn = lambda x, y: 1. / partial(mass_model.magnification, kwargs=kwargs_mass)(x, y)
        ray_shooting_fn = partial(mass_model.ray_shooting, kwargs=kwargs_mass)
        multiplane_mode = False
    elif isinstance(lens_image, MPLensImage):
        if k_plane is None:
            # we take the furthest plane as the default
            k_plane = lens_image.MPLightModel.number_light_planes - 1
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


def shear_deflection_field(lens_image, kwargs_lens, num_pixels=20):
    from herculens.MassModel.Profiles.shear import Shear, ShearGammaPsi
    shear_idx, shear_type = None, None
    num_profiles = len(lens_image.MassModel.profile_type_list)
    assert num_profiles == len(kwargs_lens)
    for i in range(num_profiles):
        if (lens_image.MassModel.profile_type_list[i] == 'SHEAR' or
            isinstance(lens_image.MassModel.profile_type_list[i], Shear)):
            shear_idx = i
            shear_type = 'SHEAR'
            break
        elif (lens_image.MassModel.profile_type_list[i] == 'SHEAR_GAMMA_PSI' or
              isinstance(lens_image.MassModel.profile_type_list[i], ShearGammaPsi)):
            shear_idx = i
            shear_type = 'SHEAR_GAMMA_PSI'
            break
    if shear_idx is None:
        return None
    if shear_type == 'SHEAR_GAMMA_PSI':
        phi_ext, gamma_ext = kwargs_lens[shear_idx]['phi_ext'], kwargs_lens[shear_idx]['gamma_ext']
    else:
        # imports are here to avoid issues with circular imports
        from herculens.Util.param_util import shear_cartesian2polar_numpy
        phi_ext, gamma_ext = shear_cartesian2polar_numpy(
            kwargs_lens[shear_idx]['gamma1'], 
            kwargs_lens[shear_idx]['gamma2'],
        )
    grid = lens_image.Grid.create_model_grid(num_pixels=num_pixels)
    x_grid_img, y_grid_img = grid.pixel_coordinates
    gamma_x = gamma_ext*np.cos(phi_ext)
    gamma_y = gamma_ext*np.sin(phi_ext)
    if gamma_x.size == 1:
        gamma_x = np.full_like(x_grid_img, float(gamma_x))
        gamma_y = np.full_like(y_grid_img, float(gamma_y))
    return (np.array(x_grid_img), np.array(y_grid_img), gamma_x, gamma_y)


def total_shear_deflection_field(lens_image, kwargs_mass, eta_flat=None, num_pixels=20, k_plane=-1):
    if isinstance(lens_image, (LensImage, LensImage3D)):
        mass_model = lens_image.MassModel
        gamma_fn = partial(mass_model.gamma, kwargs_lens=kwargs_mass)
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

