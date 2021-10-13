import numpy as np

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.LightModel.light_model import LightModel
from herculens.LensModel.lens_model import LensModel
from herculens.LensImage.lens_image import LensImage


# Coordinate grid
npix = 100
pix_scl = 0.08  # arcsec / pixel
half_size = npix * pix_scl / 2
ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2
transform_pix2angle = pix_scl * np.eye(2)
kwargs_grid = {'nx': npix, 'ny': npix,
                'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                'transform_pix2angle': transform_pix2angle}
pixel_grid = PixelGrid(**kwargs_grid)

# Noise
exp_time = 100
sigma_bkd = 0.05
kwargs_noise = {'background_rms': sigma_bkd, 'exposure_time': exp_time}
noise = Noise(npix, npix, **kwargs_noise)

# Seeing
kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 0.3}
psf = PSF(**kwargs_psf)

# Source light
source_light_model_list = ['SERSIC']
source_light_model = LightModel(source_light_model_list)
kwargs_source = [{'amp': 10.0, 'R_sersic': 1.2, 'n_sersic': 1.5, 'center_x': 0.4, 'center_y': 0.15}]

# Lens mass
lens_mass_model_list = ['SIE']
lens_mass_model = LensModel(lens_mass_model_list)
kwargs_lens = [{'theta_E': 1.6, 'e1': 0.15, 'e2': -0.04, 'center_x': 0.0, 'center_y': 0.0}]

# Lens light
lens_light_model_list = ['SERSIC']
lens_light_model = LightModel(lens_light_model_list)
kwargs_lens_light = [{'amp': 10.0, 'R_sersic': 1.8, 'n_sersic': 2.5, 'center_x': 0.0, 'center_y': 0.0}]

# Lens image object
kwargs_numerics = {'supersampling_factor': 1}
lens_image = LensImage(pixel_grid, psf, noise_class=noise,
                       lens_model_class=lens_mass_model,
                       source_model_class=source_light_model,
                       lens_light_model_class=lens_light_model,
                       kwargs_numerics=kwargs_numerics)

# Generate model image
model = lens_image.model(kwargs_lens=kwargs_lens,
                         kwargs_source=kwargs_source,
                         kwargs_lens_light=kwargs_lens_light)

# Generate simulated image
data = lens_image.simulation(compute_true_noise_map=True,
                             kwargs_lens=kwargs_lens,
                             kwargs_source=kwargs_source,
                             kwargs_lens_light=kwargs_lens_light)

def get_input_values():
    return dict(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_lens_light)

def get_building_blocks():
    return (pixel_grid, psf, noise, lens_mass_model, source_light_model, lens_light_model)

def get_lens_image():
    return lens_image

def get_model_and_data():
    return model, data
