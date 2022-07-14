# Utility functions to read simulation products from MOLET (Vernardos et al. 2022)
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal'


import os
import warnings
import numpy as np
from astropy.io import fits

from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Util.util import read_json


def read_molet_simulation(molet_path, simu_dir, 
                          use_true_noise_map=False, cut_psf=None,
                          input_file='molet_input.json', 
                          intrument_index=0, instrument_name=None,
                          subtract_offset=True, use_supersampled_psf=False,
                          align_coordinates=True):
    """
    utility method for getting the PixelGrid class from MOLET settings

    If True, align_coordinates will add an offset to the coordinates equal to a supersampled (x10) pixel.
    """
    # load the settings
    input_settings = read_json(os.path.join(molet_path, simu_dir, input_file))
    if instrument_name is None:
        instrument_name = input_settings['instruments'][intrument_index]['name']
        warnings.warn(f"Using MOLET instrument '{instrument_name}'.")
    
    assert input_settings['instruments'][intrument_index]['name'] == instrument_name, "Instrument names are not consistent."
    
    instru_settings = read_json(os.path.join(molet_path, 'instrument_modules', instrument_name, 'specs.json'))
    noise_props = read_json(os.path.join(molet_path, simu_dir, 'output', f'{instrument_name}_noise_properties.json'))
    
    # load data array
    data, data_hdr = fits.getdata(os.path.join(molet_path, simu_dir, 'output', f'OBS_{instrument_name}.fits'), header=True)
    data = data.astype(float)

    # if any offset value was added within MOLET, subtract it back
    offset = 0.
    if 'min_noise' in noise_props:
        if 'offset' not in noise_props:
            warnings.warn("Assuming the constant offset is `abs(min_noise)`!")
        else:
            offset = float(noise_props['offset'])
    
    if subtract_offset is True:
        data -= offset
        if offset != 0.:
            warnings.warn(f"An offset of {offset:.3f} was subtracted from the original MOLET simulation.")

    mass_profiles = input_settings['lenses'][0]['mass_model']
    for mass_profile in mass_profiles:
        if mass_profile['type'] == 'pert':
            pert_file = mass_profile['pars']['filepath']
            pert_path = os.path.join(molet_path, simu_dir, 'input_files', pert_file)
            try:
                dpsi_map = fits.getdata(pert_path, header=False)
            except Exception as e:
                warnings.warn(f"Error when accessing potential perturbation fits file at '{pert_path}':\n{e}")
            else:
                dpsi_map = dpsi_map.astype(float)
        else:
            dpsi_map = None
    
    # load required settings values
    fov_xmin_input = float(input_settings['instruments'][intrument_index]['field-of-view_xmin'])
    # fov_xmax = float(input_settings['instruments'][intrument_index]['field-of-view_xmax'])
    # fov_ymin = float(input_settings['instruments'][intrument_index]['field-of-view_ymin'])
    # fov_ymax = float(input_settings['instruments'][intrument_index]['field-of-view_ymax'])
    fov_xmin = float(data_hdr['XMIN'])
    fov_xmax = float(data_hdr['XMAX'])
    fov_ymin = float(data_hdr['YMIN'])
    fov_ymax = float(data_hdr['YMAX'])
    assert fov_xmin == fov_xmin_input, "Input and output FoV are not consistent."
    
    # the following follows VKL conventions for defining the coordinates grid
    pixel_size = float(instru_settings['resolution'])
    transform_pix2angle = pixel_size * np.eye(2)  # here we assume pixels are square

    width  = fov_xmax - fov_xmin
    height = fov_ymax - fov_ymin
    step_x = pixel_size
    step_y = pixel_size
    #Nx = int(width / step_x + width % step_x)
    #Ny = int(height / step_y + width % step_y)
    Nx = round(width / step_x)
    Ny = round(height / step_y)
    # if Nx % 2 == 1:
    #     Nx += 1
    # if Ny % 2 == 1:
    #     Ny += 1
    ra_at_xy_0 = -width/2. + step_x/2.
    dec_at_xy_0 = -height/2. + step_y/2.
    if align_coordinates:
        half_super_pixel = pixel_size / 10. / 2.  # MOLET uses 10x supersampling
        ra_at_xy_0 += half_super_pixel
        dec_at_xy_0 += half_super_pixel

    # setup the grid class
    kwargs_pixel = {'nx': Nx, 'ny': Ny,
                    'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
                    'transform_pix2angle': transform_pix2angle}
    pixel_grid = PixelGrid(**kwargs_pixel)

    #assert pixel_grid.extent == [fov_xmin + step_x/2., fov_xmax - step_x/2., fov_ymin + step_y/2., fov_ymax - step_y/2.], "Check FoV in MOLET settings."
    assert pixel_grid.pixel_coordinates[0].shape == data.shape, "Shape of image not consistent with coordinates grid."

    # setup the noise class
    noise_type = input_settings['instruments'][intrument_index]['noise']['type']
    if noise_type == 'PoissonNoise':
        background_rms = float(noise_props['sigma_bg'])
        if use_true_noise_map:
            noise_map = fits.getdata(os.path.join(molet_path, simu_dir, 'output', f'{instrument_name}_sigma_map.fits'), header=False)
            noise_map = noise_map.astype(float)
            exp_time  = None
        else:
            noise_map = None
            exp_time  = float(input_settings['instruments'][intrument_index]['noise']['texp'])
    elif noise_type == 'UniformGaussian':
        background_rms = float(noise_props['sigma'])
        exp_time  = None
        noise_map = None
    else:
        raise ValueError(f"Unknown type of noise '{noise_type}'.")
    noise = Noise(Nx, Ny, background_rms=background_rms, exposure_time=exp_time, noise_map=noise_map)

    # if it exists, get the super-sampled PSF that was used for the mock
    super_psf_path = os.path.join(molet_path, simu_dir, 'output', 'supersampled_psf.fits')
    if os.path.exists(super_psf_path):
        psf_kernel_super = fits.getdata(super_psf_path, header=False)
        psf_kernel_super = psf_kernel_super.astype(float)
    else:
        psf_kernel_super = None

    # setup the psf class
    psf_kernel = fits.getdata(os.path.join(molet_path, 'instrument_modules', instrument_name, 'psf.fits'), header=False)
    psf_kernel = psf_kernel.astype(float)
    true_psf_width = float(instru_settings['psf']['width'])
    expe_psf_width = psf_kernel.shape[0] * pixel_size
    if use_supersampled_psf:
        if not len(psf_kernel_super) % 2 == 0:
            psf = PSF(psf_type='PIXEL', kernel_point_source=psf_kernel_super, 
                      point_source_supersampling_factor=10)
        else:
            psf = None
            warnings.warn("Could not prepare the supersampled 'PIXEL' PSF instance as the supersampled PSF is even-sized.")
    else:
        if round(true_psf_width, 7) == round(expe_psf_width, 7):  # means it is not a supersampled PSF
            if cut_psf is not None:
                psf_kernel = psf_kernel[cut_psf:-cut_psf, cut_psf:-cut_psf]
                psf_kernel /= psf_kernel.sum()
            psf = PSF(psf_type='PIXEL', kernel_point_source=psf_kernel)
        else:
            psf = None
            warnings.warn("Could not prepare the 'PIXEL' PSF instance as the PSF in the instrument module is supersampled (what is the supersampling factor?).")
        
    # get specific noise realisation
    noise_real = fits.getdata(os.path.join(molet_path, simu_dir, 'output', f'{instrument_name}_noise_realization.fits'), header=False)
    noise_real = noise_real.astype(float)
    if subtract_offset is True:
        noise_real -= offset

    # load supersampled source and corresponding extent
    source_super, source_hdr = fits.getdata(os.path.join(molet_path, simu_dir, 'output', f'{instrument_name}_source_super.fits'), header=True)
    source_super = source_super.astype(float)
    src_fov_xmin = float(source_hdr['XMIN'])
    src_fov_xmax = float(source_hdr['XMAX'])
    src_fov_ymin = float(source_hdr['YMIN'])
    src_fov_ymax = float(source_hdr['YMAX'])
    # src_extent = (src_fov_xmin, src_fov_xmax, src_fov_ymin, src_fov_ymax)
    src_width  = src_fov_xmax - src_fov_xmin
    src_height = src_fov_ymax - src_fov_ymin
    src_pixel_size = src_width / source_super.shape[0]
    pixel_scale_factor = src_pixel_size / pixel_size
    grid_shape = (src_width, src_height)
    grid_center = ((src_fov_xmin + src_fov_xmax) / 2., (src_fov_ymin + src_fov_ymax) / 2.)
    pixel_grid.create_model_grid(grid_center=grid_center, 
                                 grid_shape=grid_shape, 
                                 pixel_scale_factor=pixel_scale_factor, 
                                 conserve_extent=False,
                                 name='MOLET_source_super')
    #pixel_grid.create_model_grid_simple(source_super.shape, src_extent, name='MOLET_source_super')
    assert pixel_grid.model_pixel_shape('MOLET_source_super') == source_super.shape
    np.testing.assert_almost_equal(pixel_grid.model_pixel_extent('MOLET_source_super'),
                                   [src_fov_xmin + src_pixel_size/2., src_fov_xmax - src_pixel_size/2.,
                                    src_fov_ymin + src_pixel_size/2., src_fov_ymax - src_pixel_size/2.])

    # flux normalisation
    source_super *= pixel_size**2

    return pixel_grid, noise, psf, data, dpsi_map, noise_real, source_super, psf_kernel_super
