# Handles coordinate systems
# 
# Copyright (c) 2022, herculens developers and contributors

__author__ = 'aymgal'


from herculens.Util import param_util

from lensmodelapi.api.galaxy import Galaxy
from lensmodelapi.api.external_shear import ExternalShear
from lensmodelapi.api.mass_light_model import MassModel, LightModel
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
        for i, profile_name in enumerate(mass_profiles_in):
            phi_ext, gamma_ext = h2c_extshear_values(profile_name, kwargs_lens[herculens_idx])
            extshear.profiles[i].parameters['phi_ext'].set_point_estimate(float(phi_ext))
            extshear.profiles[i].parameters['gamma_ext'].set_point_estimate(float(gamma_ext))

    return extshear


def h2c_extshear_values(profile_name, kwargs_profile):
    if profile_name == 'SHEAR':
        raise


def create_galaxy_model(lens_image, name, 
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
    return galaxy


def h2c_extshear_profiles(profiles_herculens):
    profiles_coolest = []
    for profile_herculens in profiles_herculens:
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
