# Describes a collection of mass models used for multi-plane lensing
# 
# Copyright (c) 2024, herculens developers and contributors

__author__ = 'krawczyk'

import numpy as np
import jax
import jax.numpy as jnp

from functools import partial
from herculens.MassModel.mass_model import MassModel


class MPMassModel(object):
    def __init__(self, mp_mass_model_list, **mass_model_kwargs):
        '''
        Create a MPMassModel object.

        Parameters
        ----------
        mp_mass_model_list : list of str
            List of lists containing Lens model profiles for each plane of
            the lens system. One inner list per plane with the outer list
            sorted by distance from observer.
        mass_model_kwargs : dictionary for settings related to PIXELATED
            profiles.
        '''
        if all([isinstance(mm, str) for mm in mp_mass_model_list]):
            self.mp_profile_type_list = mp_mass_model_list
            self.mass_models = []
            for mass_plane in self.mp_profile_type_list:
                self.mass_models.append(MassModel(
                    mass_plane,
                    **mass_model_kwargs
                ))
        elif all([isinstance(mm, MassModel) for mm in mp_mass_model_list]):
            self.mass_models = mp_mass_model_list
            self.mp_profile_type_list = [mm.func_list for mm in self.mass_models]
        else:
            raise ValueError(
                "MPMassModel needs to be initialized either with a list of lists of strings, "
                "or directly with a list of (single plane) MassModel instances.")
        self.number_mass_planes = len(self.mass_models)
        
        # Eta will be passed in flattened to `ray_shooting`, use these
        # index values to un-flatten it back into an array
        self.eta_idx = np.triu_indices(self.number_mass_planes + 1, k=2)
        # The know values for the un-flattened eta array
        self.base_eta = jnp.eye(self.number_mass_planes + 1, k=1)

    @property
    def has_pixels(self):
        # An list of bools indicating if a plane contains an pixelated mass model
        return [mass_model.has_pixels for mass_model in self.mass_models]

    @partial(jax.jit, static_argnums=(0, 5, 6))
    def ray_shooting(self, x, y, eta_flat, kwargs, N=None, k=None):
        '''
        maps image to source position (inverse deflection) on each mass plane
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param eta: upper triangular elements of eta matrix, values defined as
            eta_ij = D_ij D_i+1 / D_j D_ii+1 where D_ij is the angular diameter
            distance between redshifts i and j. Only include values where
            j > i+1. This convention implies that all einstein radii are defined
            with respect to the **next** mass plane back (**not** the last plane in
            the stack).
        :param kwargs: list of list of keyword arguments of lens model
            parameters matching the lens model classes
        :param N: number of planes to ray trace (front to back, starting at 1)
        :param k: only evaluate the k-th lens model (list of list of index values)
        :return: tuple of source plane positions corresponding to (x, y) on each
            mass plane (first index of each corresponds to mass plane)
        '''
        if (N is None) or (N > self.number_mass_planes):
            N = self.number_mass_planes
        if k is None:
            k = [None] * N
        # un-flatten eta_flat into full eta array
        etas_t = self.base_eta.at[self.eta_idx].set(eta_flat)[:-1, :(N + 1)].T

        # create output array
        xs = jnp.stack([x] * (N + 1), axis=0)
        ys = jnp.stack([y] * (N + 1), axis=0)

        # iterate the lensing equation for each mass plane
        for j in range(N):
            dx, dy = self.mass_models[j].alpha(
                xs[j],
                ys[j],
                kwargs=kwargs[j],
                k=k[j]
            )
            etas_j = etas_t[:, j:j + 1]
            xs = xs - etas_j * dx
            ys = ys - etas_j * dy
        return xs, ys

