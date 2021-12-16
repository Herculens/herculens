import os
import re
import json
import warnings
import numpy as np
from astropy.io import fits

from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF
from herculens.Coordinates.pixel_grid import PixelGrid


def read_json(input_path):
    with open(input_path,'r') as f:
        input_str = f.read()
        input_str = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "", input_str)
        input_str = re.sub(re.compile("//.*?\n" ), "", input_str)
        json_in   = json.loads(input_str)
    return json_in

def read_molet_simulation(molet_path, simu_dir, instrument_name, 
                          use_true_noise_map=False, cut_psf=None,
                          input_file='molet_input.json', intrument_index=0,
                          return_noise_real=False):
    """utility method for getting the PixelGrid class from MOLET settings"""
    # load the settings
    input_settings = read_json(os.path.join(molet_path, simu_dir, input_file))
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
    pixel_size = float(instru_settings['resolution'])
    
    # the following follows VKL conventions for defining the coordinates grid
    width  = fov_xmax - fov_xmin
    height = fov_ymax - fov_ymin
    step_x = pixel_size
    step_y = pixel_size
    Nx = int(width / step_x + width % step_x)
    Ny = int(height / step_y + width % step_y)
    ra_at_xy_0 = -width/2. + step_x/2.
    dec_at_xy_0 = -height/2. + step_y/2.
    # here we assume pixels are square
    transform_pix2angle = pixel_size * np.eye(2)

    # setup the grid class
    kwargs_pixel = {'nx': Nx, 'ny': Ny,
                    'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
                    'transform_pix2angle': transform_pix2angle}
    pixel_grid = PixelGrid(**kwargs_pixel)

    # setup the noise class
    if input_settings['instruments'][intrument_index]['noise']['type'] == 'PoissonNoise':
        background_rms = float(noise_props['sigma_bg'])
        exp_time = float(input_settings['instruments'][intrument_index]['noise']['texp'])
        if use_true_noise_map:
            noise_map = fits.getdata(os.path.join(molet_path, simu_dir, 'output', f'{instrument_name}_sigma_map.fits'), header=False)
            noise_map = noise_map.astype(float)
        else:
            noise_map = None
    else:
        background_rms = float(noise_props['sigma'])
        exp_time = None
        noise_map = None
    noise = Noise(Nx, Ny, background_rms=background_rms, exposure_time=exp_time, noise_map=noise_map)

    # setup the psf class
    psf_kernel = fits.getdata(os.path.join(molet_path, 'instrument_modules', instrument_name, 'psf.fits'), header=False)
    psf_kernel = psf_kernel.astype(float)
    if cut_psf is not None:
        psf_kernel = psf_kernel[-cut_psf:cut_psf, -cut_psf:cut_psf]
        psf_kernel /= psf_kernel.sum()
    psf = PSF(psf_type='PIXEL', kernel_point_source=psf_kernel)

    # sanity checks
    assert input_settings['instruments'][intrument_index]['name'] == instrument_name
    assert fov_xmin == fov_xmin_input
    assert pixel_grid.extent == [fov_xmin + step_x/2., fov_xmax - step_x/2., fov_ymin + step_y/2., fov_ymax - step_y/2.]
    assert pixel_grid.pixel_coordinates[0].shape == data.shape

    if return_noise_real:
        noise_real = fits.getdata(os.path.join(molet_path, simu_dir, 'output', f'{instrument_name}_noise_realization.fits'), header=False)
        noise_real = noise_real.astype(float)
        noise_real -= offset
        return pixel_grid, noise, psf, data, dpsi_map, noise_real
    else:
        return pixel_grid, noise, psf, data, dpsi_map
