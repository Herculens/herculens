# Handles coordinate systems
# 
# Copyright (c) 2022, herculens developers and contributors

__author__ = 'aymgal'


import numpy as np
from astropy.io import fits

from herculens.Util import param_util

from coolest.template.classes.galaxy import Galaxy
from coolest.template.classes.external_shear import ExternalShear
from coolest.template.classes.mass_light_model import MassModel, LightModel
from coolest.template.classes.fits_file import FitsFile
# from coolest.template.classes.parameter import PointEstimate
from coolest.template.classes.probabilities import PosteriorStatistics


# Notes: `h2c` is a shorthand for `herculens2coolest`


def create_extshear_model(lens_image, name, parameters=None,
                          mass_profile_indices=None, 
                          redshift=None):
    # external shear
    mass_profiles_all = lens_image.MassModel.profile_type_list
    mass_profiles_in  = [mass_profiles_all[i] for i in mass_profile_indices]
    mass_profiles_out = h2c_extshear_profiles(mass_profiles_in)
    
    # instantiate the external shear
    extshear = ExternalShear(name, mass_model=MassModel(*mass_profiles_out), redshift=redshift)

    if parameters is not None:
        kwargs_list = parameters.best_fit_values(as_kwargs=True)['kwargs_lens']
        if parameters.samples is not None:
            kwargs_list_samples = parameters.samples(as_kwargs=True, group_by_param=True)['kwargs_lens']
        else:
            kwargs_list_samples = None

        # add point estimate values to the ExternalShear object
        for ic, ih in enumerate(mass_profile_indices):
            profile_name = mass_profiles_in[ic]
            if profile_name == 'SHEAR':
                g1g2_param = True
            elif profile_name == 'SHEAR_GAMMA_PSI':
                g1g2_param = False
            h2c_extshear_values(extshear.mass_model[ic], kwargs_list[ih], 
                                g1g2_param=g1g2_param)
            if kwargs_list_samples is not None:
                h2c_extshear_posteriors(extshear.mass_model[ic], kwargs_list_samples[ih], 
                                        g1g2_param=g1g2_param)

    return extshear


def create_galaxy_model(lens_image, name, parameters=None,
                        mass_profile_indices=None, 
                        light_profile_indices=None,
                        lensed=None, redshift=None):
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
    
    # instantiate the galaxy
    galaxy = Galaxy(name,
                    mass_model=MassModel(*mass_profiles_out), 
                    light_model=LightModel(*light_profiles_out),
                    redshift=redshift)

    if parameters is not None:
        kwargs_all = parameters.best_fit_values(as_kwargs=True)
        if parameters.samples is not None:
            kwargs_all_samples = parameters.samples(as_kwargs=True, group_by_param=True)
        else:
            kwargs_all_samples = None

        if mass_profile_indices is not None:
            update_galaxy_mass_model(galaxy, lens_image, kwargs_all, kwargs_all_samples,
                                     mass_profile_indices, mass_profiles_in)
        if light_profile_indices is not None:
            update_galaxy_light_model(galaxy, lens_image, kwargs_all, kwargs_all_samples,
                                      light_profile_indices, light_profiles_in, lensed)

    return galaxy


def update_galaxy_mass_model(galaxy, lens_image, kwargs_all, kwargs_all_samples, profile_indices, profile_names):
    kwargs_list = kwargs_all['kwargs_lens']
    kwargs_list_samples = None if kwargs_all_samples is None else kwargs_all_samples['kwargs_lens']
    # add point estimate values
    for ic, ih in enumerate(profile_indices):
        profile_name = profile_names[ic]
        if profile_name in ['SIS', 'SIE', 'EPL']:
            if profile_name == 'SIS':
                isothermal, spherical = True, True
            elif profile_name == 'SIE':
                isothermal, spherical = True, False
            elif profile_name == 'EPL':
                isothermal, spherical = False, False
            h2c_powerlaw_values(galaxy.mass_model[ic], kwargs_list[ih], 
                                isothermal=isothermal, spherical=spherical)
            if kwargs_list_samples is not None:
                h2c_powerlaw_posteriors(galaxy.mass_model[ic], kwargs_list_samples[ih], 
                                        isothermal=isothermal, spherical=spherical)
        else:
            raise NotImplementedError(f"'{profile_name}' not yet supported.")


def update_galaxy_light_model(galaxy, lens_image, kwargs_all, kwargs_all_samples, profile_indices, profile_names, lensed):
    # get current values
    if lensed:
        kwargs_list = kwargs_all['kwargs_source']
        kwargs_list_samples = None if kwargs_all_samples is None else kwargs_all_samples['kwargs_source']
    else:
        kwargs_list = kwargs_all['kwargs_lens_light']
        kwargs_list_samples = None if kwargs_all_samples is None else kwargs_all_samples['kwargs_lens_light']
    # add point estimate values
    for ic, ih in enumerate(profile_indices):
        if profile_names[ic] == 'SERSIC_ELLIPSE':
            h2c_Sersic_values(galaxy.light_model[ic], kwargs_list[ih])
            if kwargs_list_samples is not None:
                h2c_Sersic_posteriors(galaxy.light_model[ic], kwargs_list_samples[ih])
        elif profile_names[ic] == 'SERSIC':
            h2c_Sersic_values(galaxy.light_model[ic], kwargs_list[ih], spherical=True)
            if kwargs_list_samples is not None:
                h2c_Sersic_posteriors(galaxy.light_model[ic], kwargs_list_samples[ih], spherical=True)
        elif profile_names[ic] == 'SHAPELETS':
            h2c_Shapelets_values(galaxy.light_model[ic], kwargs_list[ih],
                                 lens_image.SourceModel.func_list[ih]),  # TODO: improve access to e.g. n_max 
        # elif profile_names[ic] == 'PIXELATED':
        #     h2c_pixelated_values(galaxy.light_model[ic], kwargs_list[ih],
        #                          lens_image.SourceModel.func_list[ih]),  # TODO: improve access to e.g. pixel_grid 
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


# def h2c_pixelated_values(profile, kwargs, profile_herculens):
#     pixel_values = check_type(kwargs['pixels'])
#     x_grid, y_grid = profile_herculens.pixel_grid.pixel_coordinates
#     pixel_scale = float(profile_herculens.pixel_grid.pixel_width)
#     matrix = profile_herculens.pixel_grid.transform_pix2angle / 3600.  # arcsec -> degree
#     CD1_1 = float(matrix[0, 0]) 
#     CD1_2 = float(matrix[0, 1])
#     CD2_1 = float(matrix[1, 0])
#     CD2_2 = float(matrix[1, 1])
    
#     primary_hdr = fits.Header()
#     primary_hdr['PIXSCALE'] = pixel_scale
#     primary_hdr['CD1_1'] = CD1_1
#     primary_hdr['CD1_2'] = CD1_2
#     primary_hdr['CD2_1'] = CD2_1
#     primary_hdr['CD2_2'] = CD2_2
#     primary_hdu = fits.PrimaryHDU(pixel_values, header=primary_hdr)  # or ImageHDU?
#     columns = fits.ColDefs([
#         fits.Column(name='id', format='J', array=np.arange(x_grid.size)),
#         fits.Column(name='x', format='D', array=x_grid.flatten()),
#         fits.Column(name='y', format='D', array=y_grid.flatten()),
#         fits.Column(name='flux', format='D', array=pixel_values.flatten())
#     ])
#     pixels_hdu = fits.BinTableHDU.from_columns(columns)
#     hdu_list = fits.HDUList([primary_hdu, pixels_hdu])

#     fits_filename = 'model_pixels.fits'
#     hdu_list.writeto(fits_filename, overwrite=True)
    
#     fits_path = fits_filename
#     profile.pixels = FitsFile(fits_path)


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
        phi_ext, gamma_ext = kwargs_samples['gamma1'], kwargs_samples['gamma2']
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
    value_conv = value * 180. / np.pi
    value_conv = value_conv - 90.
    if is_iterable(value):
        for i, val in enumerate(value_conv):
            if val <= -90.:
                value_conv[i] += 180.
    else:
        if value_conv <= -90:
            value_conv += 180.
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
