# Handles coordinate systems
# 
# Copyright (c) 2022, herculens developers and contributors


__author__ =  'aymgal'


import os

import herculens
from herculens.Standard import coolest_util as util
from herculens.Util.jax_util import unjaxify_kwargs

from coolest.template.json import JSONSerializer
from coolest.template.classes.coordinates import CoordinatesOrigin
from coolest.template.classes.lensing_entity_list import LensingEntityList
from coolest.template.classes.likelihood_list import LikelihoodList
from coolest.template.classes.regularization_list import RegularizationList
# from lensmodelapi.api.cosmology import Cosmology


class COOLESTexporter(object):
    """Class that handles conversion from a Herculens model to the COOLEST file system"""

    def __init__(self, output_coolest_file, input_coolest_file=None, **kwargs_serializer):
        if input_coolest_file is None:
            raise NotImplementedError("You must provide an input coolest file (for now)")
            input_coolest_file = output_coolest_file
        if not os.path.isabs(input_coolest_file):
            input_coolest_file = os.path.abspath(input_coolest_file)
        if not os.path.isabs(output_coolest_file):
            output_coolest_file = os.path.abspath(output_coolest_file)
        self._input_coolest_file = input_coolest_file
        self._output_coolest_file = output_coolest_file
        self._json_dir = os.path.dirname(output_coolest_file)
        self._kwargs_serializer = kwargs_serializer
        self._load_coolest_object()

    @property
    def coolest_object(self):
        return self._coolest

    def update_from_model(self, lens_image, lensing_entity_mapping, 
                          parameters=None, samples=None):
        lensing_entities = self.create_lensing_entities(lens_image, lensing_entity_mapping,
                                                        parameters=parameters, samples=samples,
                                                        json_dir=self._json_dir)
        self._coolest.lensing_entities = lensing_entities

    def update_from_loss(self, loss):
        # TODO: update LikelihoodList and RegularizationList from Herculens' loss
        raise NotImplementedError("update_from_loss() not yet implemented.")

    def update_metadata(self, mode='MAP', **meta_kwargs):
        self._coolest.meta['mode'] = mode.upper()
        self._coolest.meta['code_name'] = 'Herculens'
        self._coolest.meta['code_version'] = herculens.__version__
        self._coolest.meta.update(meta_kwargs)

    @staticmethod
    def create_lensing_entities(lens_image, lensing_entity_mapping, 
                                parameters=None, samples=None, json_dir=None):
        """
        lensing_entity_mapping: list of 2-tuples of the following format:
            ('name_of_the_entity', kwargs_mapping)
        where kwargs_mapping is settings for create_extshear_model and create_galaxy_model functions. 
        """
        # TODO: check if multi-plane lensing

        if parameters is not None and isinstance(parameters, dict):
            parameters = unjaxify_kwargs(parameters)
        if samples is not None and isinstance(samples, dict):
            samples = unjaxify_kwargs(samples)

        # initialize list of lensing entities
        entities = []

        # iterate over the lensing entities (galaxies or external shears)
        for entity_name, kwargs_mapping in lensing_entity_mapping:
            entity_type = kwargs_mapping.pop('type')
            if entity_type == 'external_shear':
                entity = util.create_extshear_model(lens_image, entity_name, 
                                                    parameters=parameters,
                                                    samples=samples,
                                                    **kwargs_mapping)
            elif entity_type == 'galaxy':
                entity = util.create_galaxy_model(lens_image, entity_name, 
                                                  parameters=parameters,
                                                  samples=samples,
                                                  file_dir=json_dir,
                                                  **kwargs_mapping)
            else:
                raise ValueError(f"Unknown lensing entity type '{entity_type}'.")

            entities.append(entity)

        return LensingEntityList(*entities)

    def dump_json(self, suffix=None, with_jsonpickle=False):
        output_file = self._output_coolest_file + '-herculens'
        if suffix is not None:
            output_file += '_' + suffix
        serializer = JSONSerializer(output_file, obj=self._coolest, 
                                   **self._kwargs_serializer)
        if with_jsonpickle:
            serializer.dump_jsonpickle()
        else:
            serializer.dump_simple()

    def _load_coolest_object(self):
        serializer = JSONSerializer(self._input_coolest_file, **self._kwargs_serializer)
        std_obj = serializer.load()
        if std_obj.standard.upper() != 'COOLEST':
            raise ValueError("The JSON file is not a COOLEST template file.")
        self._coolest = std_obj
