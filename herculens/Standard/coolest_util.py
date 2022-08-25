# Handles coordinate systems
# 
# Copyright (c) 2022, herculens developers and contributors

__author__ = 'aymgal'


import numpy as np
from astropy.io import fits

from herculens.Util import param_util

from lensmodelapi.api.galaxy import Galaxy
from lensmodelapi.api.external_shear import ExternalShear
from lensmodelapi.api.mass_light_model import MassModel, LightModel
from lensmodelapi.api.fits_file import FitsFile
# from lensmodelapi.api.parameter import PointEstimate
# from lensmodelapi.api.probabilities import PosteriorStatistics 


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
        # get current values
        kwargs_lens = parameters.current_values(as_kwargs=True)['kwargs_lens']
        # add point estimate values to the ExternalShear object
        for ic, ih in enumerate(mass_profile_indices):
            phi_ext, gamma_ext = h2c_extshear_values(mass_profiles_all[ih], kwargs_lens[ih])
            extshear.mass_model[ic].parameters['phi_ext'].set_point_estimate(phi_ext)
            extshear.mass_model[ic].parameters['gamma_ext'].set_point_estimate(gamma_ext)

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
        if mass_profile_indices is not None:
            update_galaxy_mass_model(galaxy, lens_image, parameters, mass_profile_indices, mass_profiles_in)
        if light_profile_indices is not None:
            update_galaxy_light_model(galaxy, lens_image, parameters, light_profile_indices, light_profiles_in, lensed)

    return galaxy


def update_galaxy_mass_model(galaxy, lens_image, parameters, profile_indices, profile_names):
    # get current values
    kwargs_list = parameters.current_values(as_kwargs=True)['kwargs_lens']
    # add point estimate values
    for ic, ih in enumerate(profile_indices):
        if profile_names[ic] == 'SIE':
            h2c_SIE_values(galaxy.mass_model[ic], kwargs_list[ih])
        elif profile_names[ic] == 'SIS':
            h2c_SIE_values(galaxy.mass_model[ic], kwargs_list[ih], spherical=True)
        elif profile_names[ic] in ['EPL', 'PEMD']:
            h2c_EPL_values(galaxy.mass_model[ic], kwargs_list[ih])
        else:
            raise NotImplementedError(f"'{profile_names[ic]}' not yet supported.")


def update_galaxy_light_model(galaxy, lens_image, parameters, profile_indices, profile_names, lensed):
    # get current values
    kwargs_all = parameters.current_values(as_kwargs=True)
    if lensed:
        kwargs_list = kwargs_all['kwargs_source']
    else:
        kwargs_list = kwargs_all['kwargs_lens_light']
    # add point estimate values
    for ic, ih in enumerate(profile_indices):
        if profile_names[ic] == 'SERSIC_ELLIPSE':
            h2c_Sersic_values(galaxy.light_model[ic], kwargs_list[ih])
        elif profile_names[ic] == 'SERSIC':
            h2c_Sersic_values(galaxy.light_model[ic], kwargs_list[ih], spherical=True)
        elif profile_names[ic] == 'SHAPELETS':
            h2c_Shapelets_values(galaxy.light_model[ic], kwargs_list[ih],
                                 lens_image.SourceModel.func_list[ih]),  # TODO: improve access to e.g. n_max 
        else:
            raise NotImplementedError(f"'{profile_names[ic]}' not yet supported.")


def h2c_SIE_values(profile, kwargs, spherical=False):
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
    profile.parameters['theta_E'].set_point_estimate(theta_E)
    profile.parameters['center_x'].set_point_estimate(center_x)
    profile.parameters['center_y'].set_point_estimate(center_y)
    profile.parameters['phi'].set_point_estimate(phi)
    profile.parameters['q'].set_point_estimate(q)
    if spherical:
        profile.parameters['phi'].fix()
        profile.parameters['q'].fix()


def h2c_EPL_values(profile, kwargs, spherical=False):
    theta_E  = check_type(kwargs['theta_E'])
    gamma    = check_type(kwargs['gamma'])
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
    profile.parameters['theta_E'].set_point_estimate(theta_E)
    profile.parameters['gamma'].set_point_estimate(gamma)
    profile.parameters['center_x'].set_point_estimate(center_x)
    profile.parameters['center_y'].set_point_estimate(center_y)
    profile.parameters['phi'].set_point_estimate(phi)
    profile.parameters['q'].set_point_estimate(q)
    if spherical:
        profile.parameters['phi'].fix()
        profile.parameters['q'].fix()


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
    profile.parameters['A'].set_point_estimate(amp)
    profile.parameters['R_sersic'].set_point_estimate(R_sersic)
    profile.parameters['n_sersic'].set_point_estimate(n_sersic)
    profile.parameters['center_x'].set_point_estimate(center_x)
    profile.parameters['center_y'].set_point_estimate(center_y)
    profile.parameters['phi'].set_point_estimate(phi)
    profile.parameters['q'].set_point_estimate(q)
    if spherical:
        profile.parameters['phi'].fix()
        profile.parameters['q'].fix()


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


def h2c_pixelated_values(profile, kwargs, profile_herculens):
    pixels = check_type(kwargs['pixels'])
    x_grid, y_grid = profile_herculens.pixel_grid.pixel_coordinates
    fits_filename = ''
    hdu_list = 
    profile.data = FitsFile(fits_path)


def h2c_extshear_values(profile_name, kwargs_profile):
    if profile_name == 'SHEAR':
        gamma1, gamma2 = kwargs_profile['gamma1'], kwargs_profile['gamma2']
        phi_ext, gamma_ext = param_util.shear_cartesian2polar(gamma1, gamma2)
    elif profile_name == 'SHEAR_GAMMA_PSI':
        phi_ext, gamma_ext = kwargs_profile['gamma1'], kwargs_profile['gamma2']
    phi_ext   = check_type(phi_ext)
    gamma_ext = check_type(gamma_ext)
    phi_ext = h2c_position_angle(phi_ext)
    return phi_ext, gamma_ext


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
