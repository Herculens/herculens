# Handles coordinate systems
# 
# Copyright (c) 2022, herculens developers and contributors

__author__ = 'aymgal'


import os
import shutil
import math
import numpy as np
from astropy.io import fits

from herculens.Inference.legacy.parameters import Parameters as HerculensParameters
from herculens.Util import param_util
from herculens.Util.jax_util import unjaxify_kwargs

from coolest.template.lazy import *


# NOTE: `h2c` is a shorthand for `herculens2coolest`


def create_directory(directory, empty_dir_bool):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"COOLEST-info: Created directory {directory}")
    elif empty_dir_bool:
        shutil.rmtree(directory)
        os.makedirs(directory)
        print(f"COOLEST-info: Emptied directory {directory}")
    elif not os.listdir(directory):
        print(f"COOLEST-info: Directory {directory} already empty; nothing to be done.")
    else:
        return False
    return True


def create_observation(data, lens_image, json_dir=None,
                       noise_type='NoiseMap', model_noise_map=None, 
                       kwargs_obs=None, kwargs_noise=None):
    # saves the observed lens image on disk
    obs_file_name = "obs.fits"
    if json_dir is not None:
        obs_fits_path = os.path.join(json_dir, obs_file_name)
    else:
        obs_fits_path = obs_file_name
    save_image_to_fits(obs_fits_path, data, 
                       header_cards=[('COOLEST', "Observed lens image")])
    model_grid = lens_image.Grid
    if model_grid.x_is_inverted or model_grid.y_is_inverted:
        raise NotImplementedError("Grid orientation not yet supported")
    extent = model_grid.extent
    pix_scl = model_grid.pixel_width
    fov_x = [extent[0] - pix_scl/2., extent[1] + pix_scl/2.]
    fov_y = [extent[2] - pix_scl/2., extent[3] + pix_scl/2.]
    pixels = PixelatedRegularGrid(obs_file_name,
                                  field_of_view_x=fov_x,
                                  field_of_view_y=fov_y,
                                  fits_file_dir=json_dir)

    if kwargs_noise is None:
        kwargs_noise = {}
    if noise_type == 'NoiseMap':
        if model_noise_map is None:
            raise ValueError(f"A noise map must be provided for noise type {noise_type}")
        noise_map_file_name = "noise_map.fits"
        if json_dir is not None:
            noise_map_fits_path = os.path.join(json_dir, noise_map_file_name)
        else:
            noise_map_fits_path = noise_map_file_name
        save_image_to_fits(noise_map_fits_path, model_noise_map, 
                           header_cards=[('COOLEST', "Model noise map")])
        noise_map = PixelatedRegularGrid(noise_map_file_name,
                                         field_of_view_x=fov_x,
                                         field_of_view_y=fov_y,
                                         fits_file_dir=json_dir)
        noise = NoiseMap(noise_map, **kwargs_noise)
        assert math.isclose(pixels.pixel_size, noise_map.pixel_size,
                            rel_tol=1e-09, abs_tol=0.0)
    else:
        raise NotImplementedError(f"Noise type {noise_type} not yet supported")
    
    exp_map = lens_image.Noise.exposure_map
    if isinstance(exp_map, (int, float)):
        pass
    elif isinstance(exp_map, np.ndarray) and exp_map.ndim == 2:
        exp_map_file_name = "exposure_map"
        exp_map = PixelatedRegularGrid(exp_map_file_name,
                                        field_of_view_x=fov_x,
                                        field_of_view_y=fov_y,
                                        fits_file_dir=json_dir)
    else:
        raise NotImplementedError("Exposure time / map can only be a float or integerer, or a 2D array")
    
    if kwargs_obs is None:
        kwargs_obs = {}
    observation = Observation(pixels=pixels,
                              exposure_time=exp_map,
                              noise=noise,
                              **kwargs_obs)
    return observation


def create_instrument(lens_image, observation, json_dir=None,
                      psf_type='PixelatedPSF',
                      psf_description=None,
                      kwargs_psf=None):
    if psf_type == 'PixelatedPSF':
        super_factor = lens_image.PSF.kernel_supersampling_factor
        if super_factor > 1:
            psf_kernel = lens_image.PSF.kernel_point_source_supersampled(super_factor)
        elif super_factor == 1:
            psf_kernel = lens_image.PSF.kernel_point_source
        else:
            raise ValueError(f"PSF supersampling factor of {super_factor} is unknown "
                             f"for `psf_type` {psf_type}")
        psf_file_name = "psf.fits"
        if json_dir is not None:
            psf_fits_path = os.path.join(json_dir, psf_file_name)
        else:
            psf_fits_path = psf_file_name
        save_image_to_fits(psf_fits_path, psf_kernel, 
                           header_cards=[('COOLEST', "Pixelated PSF kernel")])
        pixel_size = lens_image.Grid.pixel_width / super_factor
        nx, ny = psf_kernel.shape
        fov_x = (0, nx * pixel_size)
        fov_y = (0, ny * pixel_size)
        pixels = PixelatedRegularGrid(psf_file_name,
                                      field_of_view_x=fov_x,
                                      field_of_view_y=fov_y,
                                      fits_file_dir=json_dir)
        if psf_description is None:
            psf_description = ""
        psf = PixelatedPSF(pixels, description=psf_description)
    else:
        raise NotImplementedError(f"Noise type {noise_type} not yet supported")

    if kwargs_psf is None:
        kwargs_psf = {}
    instrument = Instrument(pixel_size,
                            psf=psf,
                            **kwargs_psf)

    assert math.isclose(observation.pixels.pixel_size, instrument.pixel_size,
                        rel_tol=1e-09, abs_tol=0.0)

    return instrument


def save_image_to_fits(path, image, header_cards=[], overwrite=True):
    header = fits.Header(cards=header_cards)
    fits.writeto(path, image, header, overwrite=overwrite)
    print(f"COOLEST-info: Saved image to FITS file {path}")


def update_lensing_entities(lens_image, lensing_entity_mapping, 
                            parameters=None, samples=None, 
                            current_entities=None, re_create_entities=True,
                            json_dir=None, fits_file_suffix="coolest"):
    """
    lensing_entity_mapping: list of 2-tuples of the following format:
        ('name_of_the_entity', kwargs_mapping)
    where kwargs_mapping is settings for update_mass_field_model and update_galaxy_model functions. 
    """
    # TODO: check if multi-plane lensing

    if re_create_entities is False:
        if current_entities is None or len(current_entities) == 0:
            print("COOLEST-info: Since no lensing entities have been found nor provided, "
                         "new ones will be created based on the model.")
            re_create_entities = True
        else:
            print("COOLEST-info: Current lensing entities will be updated.")
    else:
        print("COOLEST-info: New lensing entities will be created based on the model.")

    if parameters is not None and isinstance(parameters, dict):
        parameters = unjaxify_kwargs(parameters)
    if samples is not None and isinstance(samples, dict):
        samples = unjaxify_kwargs(samples)

    # initialize list of lensing entities
    entities = []

    # iterate over the lensing entities (galaxies or external shears)
    for i, (entity_name, kwargs_mapping) in enumerate(lensing_entity_mapping):
        if kwargs_mapping is None:
            continue  # this entity is ignored (neither created nor updated!)
        
        entity_type = kwargs_mapping.pop('type')

        if re_create_entities is True:
            current_entity = None
        else:
            current_entity = current_entities[i]
        
        if entity_type == 'MassField':
            entity = update_mass_field_model(lens_image, entity_name,
                                             mass_field=current_entity, 
                                             parameters=parameters,
                                             samples=samples,
                                             **kwargs_mapping)
        elif entity_type == 'Galaxy':
            entity = update_galaxy_model(lens_image, entity_name, 
                                         galaxy=current_entity, 
                                         parameters=parameters,
                                         samples=samples,
                                         file_dir=json_dir,
                                         fits_file_suffix=fits_file_suffix,
                                         **kwargs_mapping)
        else:
            raise ValueError(f"Unknown lensing entity type '{entity_type}'.")

        if re_create_entities:
            entities.append(entity)
        else:
            # in that case the current entity has been updated in place
            pass

    return current_entities if not re_create_entities else LensingEntityList(*entities)


def update_mass_field_model(lens_image, name, mass_field=None,
                            parameters=None, samples=None,
                            mass_profile_indices=None, 
                            redshift=None):
    # external shear
    mass_profiles_all = lens_image.MassModel.profile_type_list
    mass_profiles_in  = [mass_profiles_all[i] for i in mass_profile_indices]
    if mass_profiles_in not in (['SHEAR'], ['SHEAR_GAMMA_PSI']):
        raise NotImplementedError("Mass fields other than a single external shear are not yet supported.")
    mass_profiles_out = h2c_extshear_profiles(mass_profiles_in)
    
    # instantiate the external shear
    if mass_field is None:
        mass_field = MassField(name, mass_model=MassModel(*mass_profiles_out), redshift=redshift)
    else:
        assert isinstance(mass_field, MassField), "Inconsistent entity type"

    kwargs_all, kwargs_all_samples = get_parameters_and_samples(parameters, samples)
        
    if kwargs_all is not None or kwargs_all_samples is not None:
        # add point estimate values to the MassField object
        for ic, ih in enumerate(mass_profile_indices):
            profile_name = mass_profiles_in[ic]
            if profile_name == 'SHEAR':
                g1g2_param = True
            elif profile_name == 'SHEAR_GAMMA_PSI':
                g1g2_param = False
            if kwargs_all is not None:
                h2c_extshear_values(mass_field.mass_model[ic], 
                                    kwargs_all['kwargs_lens'][ih], 
                                    g1g2_param=g1g2_param)
            if kwargs_all_samples is not None:
                h2c_extshear_posteriors(mass_field.mass_model[ic], 
                                        kwargs_all_samples['kwargs_lens'][ih], 
                                        g1g2_param=g1g2_param)

    return mass_field


def update_galaxy_model(lens_image, name, galaxy=None,
                        parameters=None, samples=None,
                        mass_profile_indices=None, light_profile_indices=None,
                        lensed=None, redshift=None, 
                        file_dir=None, fits_file_suffix="coolest"):
    # mass model
    if mass_profile_indices is not None:
        mass_profiles_all = lens_image.MassModel.profile_type_list
        mass_profiles_in  = [mass_profiles_all[i] for i in mass_profile_indices]
        mass_profiles_out = h2c_mass_profiles(mass_profiles_in)
    else:
        mass_profiles_out = []

    # light model
    if light_profile_indices is not None:
        if lensed:
            light_profiles_all = lens_image.SourceModel.profile_type_list
        else:
            light_profiles_all = lens_image.LensLightModel.profile_type_list
        light_profiles_in  = [light_profiles_all[i] for i in light_profile_indices]
        light_profiles_out = h2c_light_profiles(light_profiles_in)
    else:
        light_profiles_out = []
    
    if galaxy is None:
        # instantiate the galaxy
        galaxy = Galaxy(name,
                        mass_model=MassModel(*mass_profiles_out), 
                        light_model=LightModel(*light_profiles_out),
                        redshift=redshift)
    else:
        assert isinstance(galaxy, Galaxy), "Inconsistent entity type"

    kwargs_all, kwargs_all_samples = get_parameters_and_samples(parameters, samples)

    if kwargs_all is not None or kwargs_all_samples is not None:
        if mass_profile_indices is not None:
            update_galaxy_mass_model(galaxy, lens_image, kwargs_all, kwargs_all_samples,
                                     mass_profile_indices, mass_profiles_in, 
                                     file_dir=file_dir, fits_file_suffix=fits_file_suffix)
        if light_profile_indices is not None:
            update_galaxy_light_model(galaxy, lens_image, kwargs_all, kwargs_all_samples,
                                      light_profile_indices, light_profiles_in, lensed,
                                      file_dir=file_dir, fits_file_suffix=fits_file_suffix)

    return galaxy


def update_galaxy_mass_model(galaxy, lens_image, kwargs_all, kwargs_all_samples, 
                             profile_indices, profile_names, 
                             file_dir=None, fits_file_suffix="coolest"):
    # add point estimate values
    for ic, ih in enumerate(profile_indices):
        profile_name = profile_names[ic]
        if profile_name in ['SIS', 'SIE', 'EPL', 'PEMD', 'SPEMD']:
            if profile_name == 'SIS':
                isothermal, spherical = True, True
            elif profile_name == 'SIE':
                isothermal, spherical = True, False
            elif profile_name in ['EPL', 'PEMD']:
                isothermal, spherical = False, False
            elif profile_name == 'SPEMD':
                raise NotImplementedError("SPEMD mass profile is not yet supported")
            if kwargs_all is not None:
                h2c_powerlaw_values(galaxy.mass_model[ic], kwargs_all['kwargs_lens'][ih], 
                                    isothermal=isothermal, spherical=spherical)
            if kwargs_all_samples is not None:
                h2c_powerlaw_posteriors(galaxy.mass_model[ic], kwargs_all_samples['kwargs_lens'][ih], 
                                        isothermal=isothermal, spherical=spherical)
        else:
            raise NotImplementedError(f"'{profile_name}' not yet supported.")


def update_galaxy_light_model(galaxy, lens_image, kwargs_all, kwargs_all_samples, 
                              profile_indices, profile_names, lensed, 
                              file_dir=None, fits_file_suffix="coolest"):
    # get current values
    class_comp = lens_image.SourceModel if lensed else lens_image.LensLightModel
    light_comp = 'source' if lensed else 'lens_light'
    key = f'kwargs_{light_comp}'
    # add point estimate values
    for ic, ih in enumerate(profile_indices):
        if profile_names[ic] == 'SERSIC_ELLIPSE':
            if kwargs_all is not None:
                h2c_Sersic_values(galaxy.light_model[ic], kwargs_all[key][ih])
            if kwargs_all_samples is not None:
                h2c_Sersic_posteriors(galaxy.light_model[ic], kwargs_all_samples[key][ih])
        elif profile_names[ic] == 'SERSIC':
            if kwargs_all is not None:
                h2c_Sersic_values(galaxy.light_model[ic], kwargs_all[key][ih], spherical=True)
            if kwargs_all_samples is not None:
                h2c_Sersic_posteriors(galaxy.light_model[ic], kwargs_all_samples[key][ih], spherical=True)
        elif profile_names[ic] == 'SHAPELETS':
            if kwargs_all is not None:
                h2c_Shapelets_values(galaxy.light_model[ic], kwargs_all[key][ih],
                                     class_comp.func_list[ih])  # TODO: improve access to e.g. n_max 
            if kwargs_all_samples is not None:
                print(f"COOLEST-warning: Posterior stats for profile '{profile_names[ic]}' "
                      f"is not yet supported; samples are ignored.")
        elif profile_names[ic] == 'PIXELATED':
            if kwargs_all is not None:
                x_grid_model, y_grid_model, _ = lens_image.get_source_coordinates(kwargs_all['kwargs_lens'])
                h2c_pixelated_values(galaxy.light_model[ic], kwargs_all[key][ih],
                                     x_grid_model, y_grid_model,
                                     file_dir=file_dir,
                                     fits_file_suffix=light_comp+'-'+fits_file_suffix)  # TODO: improve access to e.g. pixel_grid 
            if kwargs_all_samples is not None:
                print(f"COOLEST-warning: Posterior stats for profile '{profile_names[ic]}' "
                      f"is not yet supported; samples are ignored.")
        else:
            raise NotImplementedError(f"'{profile_names[ic]}' not yet supported.")


def h2c_powerlaw_values(profile, kwargs, isothermal=False, spherical=False):
    theta_E  = check_type(kwargs['theta_E'])
    center_x = check_type(kwargs['center_x'])
    center_y = check_type(kwargs['center_y'])
    if spherical:
        phi, q = 0., 1.
    else:
        e1 = check_type(kwargs['e1'])
        e2 = check_type(kwargs['e2'])
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        phi = h2c_position_angle(phi)
        phi = check_type(phi)
        q = check_type(q)
    if not isothermal:
        gamma = check_type(kwargs['gamma'])
    profile.parameters['theta_E'].set_point_estimate(theta_E)
    profile.parameters['center_x'].set_point_estimate(center_x)
    profile.parameters['center_y'].set_point_estimate(center_y)
    profile.parameters['phi'].set_point_estimate(phi)
    profile.parameters['q'].set_point_estimate(q)
    if spherical:
        profile.parameters['phi'].fix()
        profile.parameters['q'].fix()
    if not isothermal:
        profile.parameters['gamma'].set_point_estimate(gamma)


def h2c_powerlaw_posteriors(profile, kwargs_samples, isothermal=False, spherical=False):
    profile.parameters['theta_E'].set_posterior(prepare_posterior(kwargs_samples['theta_E']))
    profile.parameters['center_x'].set_posterior(prepare_posterior(kwargs_samples['center_x']))
    profile.parameters['center_y'].set_posterior(prepare_posterior(kwargs_samples['center_y']))
    if spherical:
        phi, q = 0., 1.
    else:
        e1 = kwargs_samples['e1']
        e2 = kwargs_samples['e2']
        phi, q = param_util.ellipticity2phi_q_numpy(e1, e2)
        phi = h2c_position_angle(phi)
    profile.parameters['phi'].set_posterior(prepare_posterior(phi))
    profile.parameters['q'].set_posterior(prepare_posterior(q))
    if not isothermal:
        profile.parameters['gamma'].set_posterior(prepare_posterior(kwargs_samples['gamma']))


def h2c_Sersic_values(profile, kwargs, spherical=False):
    amp = check_type(kwargs['amp'])
    R_sersic = check_type(kwargs['R_sersic'])
    n_sersic = check_type(kwargs['n_sersic'])
    center_x = check_type(kwargs['center_x'])
    center_y = check_type(kwargs['center_y'])
    if spherical:
        phi, q = 0., 1.  # spherical case
    else:
        e1 = check_type(kwargs['e1'])
        e2 = check_type(kwargs['e2'])
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        phi = h2c_position_angle(phi)
        phi = check_type(phi)
        q = check_type(q)
    R_sersic = convert_major_axis_radius(R_sersic, q)
    profile.parameters['I_eff'].set_point_estimate(amp)
    profile.parameters['theta_eff'].set_point_estimate(R_sersic)
    profile.parameters['n'].set_point_estimate(n_sersic)
    profile.parameters['center_x'].set_point_estimate(center_x)
    profile.parameters['center_y'].set_point_estimate(center_y)
    profile.parameters['phi'].set_point_estimate(phi)
    profile.parameters['q'].set_point_estimate(q)
    if spherical:
        profile.parameters['phi'].fix()
        profile.parameters['q'].fix()


def h2c_Sersic_posteriors(profile, kwargs_samples, spherical=False):
    profile.parameters['I_eff'].set_posterior(prepare_posterior(kwargs_samples['amp']))
    profile.parameters['n'].set_posterior(prepare_posterior(kwargs_samples['n_sersic']))
    profile.parameters['center_x'].set_posterior(prepare_posterior(kwargs_samples['center_x']))
    profile.parameters['center_y'].set_posterior(prepare_posterior(kwargs_samples['center_y']))
    if spherical:
        phi, q = 0., 1.  # spherical case
    else:
        e1 = kwargs_samples['e1']
        e2 = kwargs_samples['e2']
        phi, q = param_util.ellipticity2phi_q_numpy(e1, e2)
        phi = h2c_position_angle(phi)
    R_sersic = kwargs_samples['R_sersic']
    R_sersic = convert_major_axis_radius(R_sersic, q)
    profile.parameters['theta_eff'].set_posterior(prepare_posterior(kwargs_samples['R_sersic']))
    profile.parameters['phi'].set_posterior(prepare_posterior(phi))
    profile.parameters['q'].set_posterior(prepare_posterior(q))


def h2c_Shapelets_values(profile, kwargs, profile_herculens):
    amps = check_type(kwargs['amps'])
    beta = check_type(kwargs['beta'])
    center_x = check_type(kwargs['center_x'])
    center_y = check_type(kwargs['center_y'])
    n_max = check_type(profile_herculens.maximum_order)
    profile.parameters['amps'].set_point_estimate(amps)
    profile.parameters['beta'].set_point_estimate(beta)
    profile.parameters['center_x'].set_point_estimate(center_x)
    profile.parameters['center_y'].set_point_estimate(center_y)
    profile.parameters['n_max'].set_point_estimate(int(n_max))
    profile.parameters['n_max'].fix()


def h2c_pixelated_values(profile, kwargs, x_grid, y_grid, 
                         file_dir=None, fits_file_suffix="coolest"):
    """Profile based on REGULAR grid of pixels"""
    pixel_values = check_type(kwargs['pixels'])
    if x_grid[0, 0] > x_grid[0, 1]:
        raise NotImplementedError("Only increasing x coordinates is supported so far")
    if y_grid[0, 0] > y_grid[1, 0]:
        raise NotImplementedError("Only increasing y coordinates is supported so far")
    pixel_scale = float(abs(x_grid[0, 0] - x_grid[0, 1]))
    pixel_scale_y = float(abs(y_grid[0, 0] - y_grid[1, 0]))
    print(f"COOLEST-info: assuming SQUARE pixels for the source "
          f"(pixel size along x = {pixel_scale}, along y = {pixel_scale_y}).")
    
    half_pix = pixel_scale / 2.
    extent = [x_grid[0, 0], x_grid[0, -1], y_grid[0, 0], y_grid[-1, 0]]
    fov_x = [extent[0]-half_pix, extent[1]+half_pix]
    fov_y = [extent[2]-half_pix, extent[3]+half_pix]

    transform_pix2angle = np.array([[pixel_scale, 0.], [0, pixel_scale]])
    matrix = transform_pix2angle / 3600.  # arcsec -> degree
    CD1_1 = float(matrix[0, 0])
    CD1_2 = float(matrix[0, 1])
    CD2_1 = float(matrix[1, 0])
    CD2_2 = float(matrix[1, 1])

    fits_filename = f'model-{fits_file_suffix}.fits'
    if file_dir is None:
        fits_path = fits_filename
    else:
        fits_path = os.path.join(file_dir, fits_filename)
    save_image_to_fits(fits_path, pixel_values, 
                       header_cards=[
                            ('COOLEST', "Pixelated surface brightness model"),
                            ('PIXSCALE', pixel_scale),
                            ('CD1_1', CD1_1),
                            ('CD1_2', CD1_2),
                            ('CD2_1', CD2_1),
                            ('CD2_2', CD2_2),
                       ])

    pixels = PixelatedRegularGrid(fits_filename, # relative path to fits file
                                  field_of_view_x=fov_x,
                                  field_of_view_y=fov_y,
                                  check_fits_file=True,
                                  fits_file_dir=file_dir)
    # set the pixel values
    profile.parameters['pixels'] = pixels
    # finally, just perform a consistency check
    recomputed_pixel_scale = profile.parameters['pixels'].pixel_size
    consistent =  math.isclose(pixel_scale, recomputed_pixel_scale, 
                               rel_tol=0.0, abs_tol=1e-06)
    if not consistent:
        raise ValueError(f"Inconsistent pixel scale values for profile '{profile.type}' "
                         f"({pixel_scale} vs. {recomputed_pixel_scale})")


def h2c_extshear_values(profile, kwargs, g1g2_param=False):
    if g1g2_param:
        gamma1, gamma2 = kwargs['gamma1'], kwargs['gamma2']
        phi_ext, gamma_ext = param_util.shear_cartesian2polar(gamma1, gamma2)
    else:
        phi_ext, gamma_ext = kwargs['phi_ext'], kwargs['gamma_ext']
    phi_ext   = check_type(phi_ext)
    gamma_ext = check_type(gamma_ext)
    phi_ext = h2c_position_angle(phi_ext)
    profile.parameters['phi_ext'].set_point_estimate(phi_ext)
    profile.parameters['gamma_ext'].set_point_estimate(gamma_ext)


def h2c_extshear_posteriors(profile, kwargs_samples, g1g2_param=False):
    if g1g2_param:
        gamma1, gamma2 = kwargs_samples['gamma1'], kwargs_samples['gamma2']
        phi_ext, gamma_ext = param_util.shear_cartesian2polar_numpy(gamma1, gamma2)
    else:
        phi_ext, gamma_ext = kwargs_samples['phi_ext'], kwargs_samples['gamma_ext']
    phi_ext = h2c_position_angle(phi_ext)
    profile.parameters['phi_ext'].set_posterior(prepare_posterior(phi_ext))
    profile.parameters['gamma_ext'].set_posterior(prepare_posterior(gamma_ext))


def h2c_extshear_profiles(profiles_herculens):
    profiles_coolest = []
    for i, profile_herculens in enumerate(profiles_herculens):
        if profile_herculens in ['SHEAR', 'SHEAR_GAMMA_PSI']:
            profiles_coolest.append('ExternalShear')
        else:
            raise ValueError(f"Unknown COOLEST mapping for mass profile '{profile_herculens}'.")
    return profiles_coolest


def h2c_mass_profiles(profiles_herculens):
    profiles_coolest = []
    for profile_herculens in profiles_herculens:
        if profile_herculens in ['SIS', 'SIE']:
            profiles_coolest.append('SIE')
        elif profile_herculens in ['NIE']:
            profiles_coolest.append('NIE')
        elif profile_herculens in ['PEMD', 'EPL']:
            profiles_coolest.append('PEMD')
        elif profile_herculens in ['SPEMD']:
            profiles_coolest.append('SPEMD')
        elif profile_herculens in ['PIXELATED']:
            profiles_coolest.append('PixelatedPotential')
        elif profile_herculens in ['SHEAR', 'SHEAR_GAMMA_PSI']:
            raise ValueError("External shear cannot be a galaxy mass profile.")
        else:
            raise ValueError(f"Unknown COOLEST mapping for mass profile '{profile_herculens}'.")
    return profiles_coolest


def h2c_light_profiles(profiles_herculens):
    profiles_coolest = []
    for profile_herculens in profiles_herculens:
        if profile_herculens in ['SERSIC', 'SERSIC_ELLIPSE']:
            profiles_coolest.append('Sersic')
        elif profile_herculens in ['UNIFORM']:
            profiles_coolest.append('Uniform')
        elif profile_herculens in ['SHAPELETS']:
            profiles_coolest.append('Shapelets')
        elif profile_herculens in ['SHAPELETS']:
            profiles_coolest.append('LensedPS')
        elif profile_herculens in ['PIXELATED']:
            profiles_coolest.append('PixelatedRegularGrid')
        else:
            raise ValueError(f"Unknown COOLEST mapping for light profile '{profile_herculens}'.")
    return profiles_coolest


def h2c_position_angle(value):
    """
    Transform an angle in radian from Herculens into an angle in degree in the COOLEST conventions.
    Based on @LyneVdV's implementation for lenstronomy.
    """
    value_conv = value + np.pi / 2.
    value_conv = value_conv * 180. / np.pi
    print("COOLEST-warning: Assuming x axis is positive towards the right when converting angles!")
    if is_iterable(value):
       median = np.median(value)
       for i, val in enumerate(value_conv):
            if median <= -90.:
                value_conv[i] += 180.
            elif median > 90.:
               value_conv[i] -= 180.
    else:
        if value_conv <= -90:
            value_conv += 180.
        elif value_conv > 90:
            value_conv -= 180.
    return value_conv


def prepare_posterior(samples):
    if isinstance(samples, (int, float)) or len(samples) == 1:
        mean = samples
        perc = None, None, None
    else:
        mean, perc = np.mean(samples), np.percentile(samples, q=[16, 50, 84])
    posterior = PosteriorStatistics(mean=check_type(mean),
                                    median=check_type(perc[1]),
                                    percentile_16th=check_type(perc[0]),
                                    percentile_84th=check_type(perc[2]))
    return posterior


def convert_major_axis_radius(r, q):
    return r * np.sqrt(q)


def check_type(value):
    if value is None:
        return None
    if is_iterable(value):
        value_valid = np.asarray(value)
    else:
        value_valid = float(value)
    return value_valid


def is_iterable(value):
    try:
        _ = iter(value)  # test if value is iterable
    except:
        return False
    else:
        return True

def get_parameters_and_samples(parameters, samples):
    kwargs_all = None
    kwargs_all_samples = None

    if parameters is not None:
        if isinstance(parameters, HerculensParameters):
            print("COOLEST-info: Using the legacy interface of the Parameters class")
            kwargs_all = parameters.best_fit_values(as_kwargs=True)
            if parameters.samples is not None:
                kwargs_all_samples = parameters.samples(as_kwargs=True, group_by_param=True)
            else:
                kwargs_all_samples = None

        elif isinstance(parameters, dict):
            kwargs_all = parameters
            kwargs_all_samples = None if samples is None else samples

    elif parameters is None and samples is not None:
        kwargs_all = None
        if isinstance(samples, dict):
            kwargs_all_samples = None if samples is None else samples
        else:
            raise ValueError(f"Unsupported type '{type(samples)}' for `samples`")
    
    return kwargs_all, kwargs_all_samples
