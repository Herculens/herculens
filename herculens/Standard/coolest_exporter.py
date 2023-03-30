# Handles coordinate systems
# 
# Copyright (c) 2022, herculens developers and contributors


__author__ =  'aymgal'


import os

import herculens
from herculens.Standard import coolest_util as util

from coolest.template.json import JSONSerializer
from coolest.template.classes.coordinates import CoordinatesOrigin
from coolest.template.classes.lensing_entity_list import LensingEntityList
from coolest.template.classes.likelihood_list import LikelihoodList
from coolest.template.classes.regularization_list import RegularizationList
# from lensmodelapi.api.cosmology import Cosmology


class COOLESTexporter(object):
    """Class that handles conversion from a Herculens model to the COOLEST file system"""

    _prefix = 'coolest-herculens'

    def __init__(self, output_basename, output_directory, input_coolest_file=None, 
                 empty_output_directory=False, **kwargs_serializer):
        if input_coolest_file is None:
            raise NotImplementedError("You must provide an input coolest file (for now)")
        if not os.path.isabs(input_coolest_file):
            input_coolest_file = os.path.abspath(input_coolest_file)
        output_coolest_file = os.path.join(output_directory, self._prefix+'-'+output_basename)
        if not os.path.isabs(output_coolest_file):
            output_coolest_file = os.path.abspath(output_coolest_file)
        self._input_coolest_file = input_coolest_file
        self._output_coolest_file = output_coolest_file
        self._output_dir = os.path.dirname(output_coolest_file)
        self._basename = output_basename
        self._kwargs_serializer = kwargs_serializer
        self._coolest = self._load_coolest_object()
        check = util.create_output_directory(self._output_dir, empty_output_directory)
        if check is False:
            raise ValueError("Output directory already exists and is not empty "
                             "(use `empty_output_directory=True` keyword argument)")

    @property
    def coolest_object(self):
        return self._coolest

    def save_on_disk(self, suffix=None, with_jsonpickle=False):
        # TODO: save a directory instead of just a json file
        output_file = self._output_coolest_file
        if suffix is not None:
            output_file += '-' + suffix
        serializer = JSONSerializer(output_file, obj=self._coolest, 
                                   **self._kwargs_serializer)
        if with_jsonpickle:
            serializer.dump_jsonpickle()
        else:
            serializer.dump_simple()

    def update_from_data(self, data, lens_image,
                         noise_type='NoiseMap', noise_map=None,
                         psf_type='PixelatedPSF', psf_description=None,
                         kwargs_obs=None, kwargs_noise=None, kwargs_psf=None):
        if 'mag_zero_point' not in kwargs_obs:
            kwargs_obs['mag_zero_point'] = self._coolest.observation.mag_zero_point
        if 'mag_sky_brightness' not in kwargs_obs:
            kwargs_obs['mag_sky_brightness'] = self._coolest.observation.mag_sky_brightness
        observation = util.create_observation(data, lens_image, 
                                              noise_type=noise_type,
                                              model_noise_map=noise_map, 
                                              json_dir=self._output_dir,
                                              fits_file_suffix=self._basename,
                                              kwargs_obs=kwargs_obs,
                                              kwargs_noise=kwargs_noise)
        instrument = util.create_instrument(lens_image, observation,
                                            json_dir=self._output_dir,
                                            fits_file_suffix=self._basename,
                                            psf_type=psf_type,
                                            psf_description=psf_description,
                                            kwargs_psf=kwargs_psf)
        # overwites the attributes
        self._coolest.observation = observation
        self._coolest.instrument = instrument

    def update_from_model(self, lens_image, lensing_entity_mapping, 
                          parameters=None, samples=None):
        lensing_entities = util.create_lensing_entities(lens_image, lensing_entity_mapping,
                                                        parameters=parameters, samples=samples,
                                                        json_dir=self._output_dir)
        # overwites the lensing entities of the COOLEST object
        self._coolest.lensing_entities = lensing_entities
        if parameters is not None:
            # set the COOLEST mode to MAP (i.e., maximum a-posteriori estimate)
            self._coolest.mode = "MAP"

    def update_from_loss(self, loss):
        # TODO: update LikelihoodList and RegularizationList from Herculens' loss
        raise NotImplementedError("update_from_loss() not yet implemented.")

    def update_metadata(self, **meta_kwargs):
        self._coolest.meta['modeling_code'] = f"Herculens (v{herculens.__version__})"
        self._coolest.meta.update(meta_kwargs)

    def _load_coolest_object(self):
        serializer = JSONSerializer(self._input_coolest_file, **self._kwargs_serializer)
        coolest_obj = serializer.load()
        if coolest_obj.standard.upper() != 'COOLEST':
            raise ValueError("The JSON file is not a COOLEST template file.")
        return coolest_obj
