# Handles coordinate systems
# 
# Copyright (c) 2022, herculens developers and contributors


__author__ =  'aymgal'


import os
import shutil

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

    def __init__(self, output_basename, output_directory, input_coolest_file=None, 
                 empty_output_directory=False, **kwargs_serializer):
        if input_coolest_file is None:
            raise NotImplementedError("You must provide an input coolest file (for now)")
        if not os.path.isabs(input_coolest_file):
            input_coolest_file = os.path.abspath(input_coolest_file)
        output_coolest_file = os.path.join(output_directory, output_basename)
        if not os.path.isabs(output_coolest_file):
            output_coolest_file = os.path.abspath(output_coolest_file)
        self._input_coolest_file = input_coolest_file
        self._output_coolest_file = output_coolest_file
        self._output_dir = os.path.dirname(output_coolest_file)
        self._basename = output_basename
        self._kwargs_serializer = kwargs_serializer
        self._coolest = self._load_coolest_object()
        self._create_output_directory(empty_output_directory)

    def _create_output_directory(self, empty_dir):
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
            print(f"Created directory {self._output_dir}")
        elif empty_dir:
            shutil.rmtree(self._output_dir)
            os.makedirs(self._output_dir)
            print(f"Emptied directory {self._output_dir}")
        elif not os.listdir(self._output_dir):
            print(f"Directory {self._output_dir} already empty; nothing to be done.")
        else:
            raise ValueError("Output directory already exists and is not empty "
                             "(use `empty_output_directory=True` for cleaning it)")

    @property
    def coolest_object(self):
        return self._coolest

    def dump_json(self, suffix=None, with_jsonpickle=False):
        # TODO: save a directory instead of just a json file
        output_file = self._output_coolest_file + '-herculens'
        if suffix is not None:
            output_file += '_' + suffix
        serializer = JSONSerializer(output_file, obj=self._coolest, 
                                   **self._kwargs_serializer)
        if with_jsonpickle:
            serializer.dump_jsonpickle()
        else:
            serializer.dump_simple()

    def update_from_data(self, data, lens_image,
                         noise_type='NoiseMap', 
                         noise_map=None):
        observation = util.create_observation(data, lens_image, 
                                              noise_type=noise_type,
                                              model_noise_map=noise_map, 
                                              json_dir=self._output_dir,
                                              fits_file_prefix=self._basename)
        #instrument = util.create_instrument(lens_image, 
                                            #json_dir=self._output_dir)
        # overwites the attributes
        self._coolest.observation = observation
        #self._coolest.instrument = instrument

    def update_from_model(self, lens_image, lensing_entity_mapping, 
                          parameters=None, samples=None):
        lensing_entities = util.create_lensing_entities(lens_image, lensing_entity_mapping,
                                                        parameters=parameters, samples=samples,
                                                        json_dir=self._output_dir)
        # overwites the lensing entities of the COOLEST object
        self._coolest.lensing_entities = lensing_entities

    def update_from_loss(self, loss):
        # TODO: update LikelihoodList and RegularizationList from Herculens' loss
        raise NotImplementedError("update_from_loss() not yet implemented.")

    def update_metadata(self, mode='MAP', **meta_kwargs):
        self._coolest.meta['mode'] = mode.upper()
        self._coolest.meta['code_name'] = 'Herculens'
        self._coolest.meta['code_version'] = herculens.__version__
        self._coolest.meta.update(meta_kwargs)

    def _load_coolest_object(self):
        serializer = JSONSerializer(self._input_coolest_file, **self._kwargs_serializer)
        coolest_obj = serializer.load()
        if coolest_obj.standard.upper() != 'COOLEST':
            raise ValueError("The JSON file is not a COOLEST template file.")
        return coolest_obj
