# Handles coordinate systems
# 
# Copyright (c) 2022, herculens developers and contributors


__author__ =  'aymgal'


import os
from copy import deepcopy

import herculens
from herculens.Standard import coolest_util as util

from coolest.template.json import JSONSerializer
from coolest.api.util import get_coolest_object


class COOLESTexporter(object):
    """Class that handles conversion from a Herculens model to the COOLEST file system"""

    def __init__(self, output_basename, output_directory, input_coolest_file=None, 
                 empty_output_directory=False, **kwargs_serializer):
        if input_coolest_file is None:
            raise NotImplementedError("You must provide an input coolest file (for now)")
        if not os.path.isabs(input_coolest_file):
            self._input_coolest_file = os.path.abspath(input_coolest_file)
        else:
            self._input_coolest_file = input_coolest_file
        output_coolest_file = os.path.join(output_directory, output_basename)
        if not os.path.isabs(output_coolest_file):
            self._output_coolest_file = os.path.abspath(output_coolest_file)
        else:
            self._output_coolest_file = output_coolest_file
        self._output_dir = os.path.dirname(output_coolest_file)
        self._basename = output_basename
        self._kwargs_serializer = kwargs_serializer
        self._coolest = self._load_coolest_object()
        dir_bool = util.create_directory(self._output_dir, empty_output_directory)
        if dir_bool is False:
            print("COOLEST-warning: Output directory already exists and has not been emptied. "
                  "The template file might be directly updated.")

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
        print("COOLEST-info: successfully saved the updated template file.")

    def update_from_data(self, data, lens_image,
                         noise_type='NoiseMap', noise_map=None,
                         psf_type='PixelatedPSF', psf_description=None,
                         kwargs_obs=None, kwargs_noise=None, kwargs_instrument=None):
        if 'mag_zero_point' not in kwargs_obs:
            kwargs_obs['mag_zero_point'] = self._coolest.observation.mag_zero_point
        if 'mag_sky_brightness' not in kwargs_obs:
            kwargs_obs['mag_sky_brightness'] = self._coolest.observation.mag_sky_brightness
        observation = util.create_observation(data, lens_image, 
                                              noise_type=noise_type,
                                              model_noise_map=noise_map, 
                                              json_dir=self._output_dir,
                                              kwargs_obs=kwargs_obs,
                                              kwargs_noise=kwargs_noise)
        instrument = util.create_instrument(lens_image, observation,
                                            json_dir=self._output_dir,
                                            psf_type=psf_type,
                                            psf_description=psf_description,
                                            kwargs=kwargs_instrument)
        # overwites the attributes
        self._coolest.observation = observation
        self._coolest.instrument = instrument

    def update_from_model(self, lens_image, lensing_entity_mapping, 
                          parameters=None, samples=None, 
                          re_create_entities=False):
        lensing_entities = util.update_lensing_entities(lens_image, 
                                                        deepcopy(lensing_entity_mapping),
                                                        parameters=parameters, samples=samples,
                                                        re_create_entities=re_create_entities,
                                                        current_entities=self._coolest.lensing_entities,
                                                        json_dir=self._output_dir,
                                                        fits_file_suffix="bestfit")
        if re_create_entities is True:
            # overwites the lensing entities of the COOLEST object
            self._coolest.lensing_entities = lensing_entities
        if parameters is not None or samples is not None:
            # set the COOLEST mode to MAP (i.e., maximum a-posteriori estimate)
            self._coolest.mode = "MAP"

    def update_from_likelihood(self, lens_image,
                               likelihood_type="ImagingDataLikelihood",
                               likelihood_mask=None):
        likelihoods = util.create_likelihoods(
            lens_image, likelihood_type=likelihood_type,
            likelihood_mask=likelihood_mask,
            json_dir=self._output_dir,
        )
        self._coolest.likelihoods = likelihoods

    def update_from_wcs_coordinates(self, skycoord):
        self._coolest.coordinates_origin = util.skycoord_to_coolest(skycoord)

    def update_metadata(self, **meta_kwargs):
        self._coolest.meta['modeling_code'] = f"Herculens (v{herculens.__version__})"
        self._coolest.meta.update(meta_kwargs)

    def delete_metadata(self, meta_key):
        res = self._coolest.meta.pop(meta_key, None)
        if res is None:
            print(f"COOLEST-warning: key '{meta_key}' has not been found in the COOLEST metadata.")
        else:
            print(f"COOLEST-info: successfully removed key '{meta_key}' from COOLEST metadata.")

    def _load_coolest_object(self):
        coolest_obj = get_coolest_object(self._input_coolest_file, **self._kwargs_serializer)
        return coolest_obj
