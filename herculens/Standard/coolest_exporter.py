# Handles coordinate systems
# 
# Copyright (c) 2022, herculens developers and contributors


__author__ =  'aymgal'


from herculens.Standard import coolest_util

from lensmodelapi.io import APISerializer
from lensmodelapi.api.coordinates import CoordinatesOrigin
from lensmodelapi.api.lensing_entity_list import LensingEntityList
from lensmodelapi.api.likelihood_list import LikelihoodList
from lensmodelapi.api.regularization_list import RegularizationList
# from lensmodelapi.api.cosmology import Cosmology


class COOLESTexporter(object):
    """Class that handles conversion from a Herculens model to the COOLEST file system"""

    def __init__(self, template_file_name, **kwargs_serializer):
        self._base_file_name = template_file_name
        self._template_file_name = template_file_name
        self._kwargs_serializer = kwargs_serializer
        self._load_coolest_object()

    @property
    def coolest_object(self):
        return self._coolest
        
    def _load_coolest_object(self):
        serializer = APISerializer(self._template_file_name, **self._kwargs_serializer)
        std_obj = serializer.json_load()
        if std_obj.standard.upper() != 'COOLEST':
            raise ValueError("The JSON file is not a COOLEST template file.")
        self._coolest = std_obj

    def update_from_model(self, lens_image, lensing_entity_mapping, parameters=None):
        lensing_entities = self.create_lensing_entities(lens_image, lensing_entity_mapping,
                                                        parameters=parameters)
        self._coolest.lensing_entities = lensing_entities

    def update_from_loss(self, loss):
        # TODO: update LikelihoodList and RegularizationList from Herculens' loss
        raise NotImplementedError("update_from_loss() not yet implemented.")

    @staticmethod
    def create_lensing_entities(lens_image, lensing_entity_mapping, parameters=None):
        # TODO: check if multi-plane lensing

        # initialize list of lensing entities
        entities = []

        # iterate over the lensing entities (galaxies or external shears)
        for entity_name, kwargs_mapping in lensing_entity_mapping.items():
            entity_type = kwargs_mapping.pop('type')
            if entity_type == 'external_shear':
                entity = coolest_util.create_extshear_model(lens_image, entity_name, 
                                                            parameters=parameters,
                                                            **kwargs_mapping)
            elif entity_type == 'galaxy':
                entity = coolest_util.create_galaxy_model(lens_image, entity_name, 
                                                          parameters=parameters,
                                                          **kwargs_mapping)
            else:
                raise ValueError(f"Unknown lensing entity type '{entity_type}'.")

            entities.append(entity)

        return LensingEntityList(*entities)
