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
        self.mp_profile_type_list = mp_mass_model_list
        self.mass_models = []
        self.number_mass_planes = len(self.mp_profile_type_list)
        for mass_plane in self.mp_profile_type_list:
            self.mass_models.append(MassModel(
                mass_plane,
                **mass_model_kwargs
            ))
        
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
        '''Maps image to source position (inverse deflection) on each mass plane

        Parameters
        ----------
        x : jax.numpy array
            x-position (preferentially arcsec)
        y : jax.numpy array
            y-position (preferentially arcsec)
        eta_flat : jax.numpy array
            upper triangular elements of eta matrix, values defined as
            eta_ij = D_ij D_i+1 / D_j D_ii+1 where D_ij is the angular diameter
            distance between redshifts i and j. Only include values where
            j > i+1. This convention implies that all einstein radii are defined
            with respect to the **next** mass plane back (**not** the last plane in
            the stack).
        kwargs: list of list
            List of lists of parameter dictionaries of lens mass model parameters
            corresponding to each mass plane.
        N : int, optional
            number of planes to ray trace (front to back, starting at 1), by default None
        k : list of list, optional
            only evaluate the k-th lens model (list of list of index values) for a particular
            plane, by default None

        Returns
        -------
        x_deflected : jax.numpy array
            x source plane positions on each mass plane (first index of each corresponds to
            mass plane)
        y_deflected : jax.numpy array
            y source plane positions on each mass plane (first index of each corresponds to
            mass plane)
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

    def _ray_shooting_slice(self, x, y, eta_flat, kwargs):
        '''Helper function that give *scaler* inputs of x and y give a *vector*
        output for each mass plane. Used for the vectorization of the `A` method'''
        return jnp.stack(
            self.ray_shooting(jnp.array([x]), jnp.array([y]), eta_flat, kwargs)
        ).T.squeeze()

    def _A_stack(self, x, y, eta_flat, kwargs, kind='auto'):
        '''Helper function that takes the jacobian of the ray shooting give *scaler*
        inputs for x and y and returns a 2x2 array.'''
        if kind == 'auto':
            return jnp.stack(
                jax.jacfwd(
                    self._ray_shooting_slice,
                    argnums=(0, 1)
                )(x, y, eta_flat, kwargs)
            )
        elif kind == 'direct':
            N = self.number_mass_planes
            A = jnp.kron(jnp.eye(2), jnp.ones((N + 1, 1))).reshape(2, N + 1, 2)
            xs, ys = self.ray_shooting(jnp.array([x]), jnp.array([y]), eta_flat, kwargs)
            etas_t = self.base_eta.at[self.eta_idx].set(eta_flat)[:-1, :(N + 1)].T
            for j in range(N):
                hxx, hxy, hyx, hyy = self.mass_models[j].hessian(
                    xs[j][0],
                    ys[j][0],
                    kwargs[j]
                )
                etas_j = etas_t[:, j:j + 1]
                A_H = A[:, j, :] @ jnp.array([[hxx, hxy], [hyx, hyy]])
                A = A - jnp.kron(A_H, etas_j).reshape(2, N + 1, 2)
            return A

    @partial(jax.jit, static_argnums=(0, 5))
    def A(self, x, y, eta_flat, kwargs, kind='auto'):
        '''
        Area distortion matrix of the lens mapping.

        Parameters
        ----------
        x : jax.numpy array
            x-position (preferentially arcsec)
        y : jax.numpy array
            y-position (preferentially arcsec)
        eta_flat : jax.numpy array
            upper triangular elements of eta matrix, values defined as
            eta_ij = D_ij D_i+1 / D_j D_ii+1 where D_ij is the angular diameter
            distance between redshifts i and j. Only include values where
            j > i+1. This convention implies that all einstein radii are defined
            with respect to the **next** mass plane back (**not** the last plane in
            the stack).
        kwargs: list of list
            List of lists of parameter dictionaries of lens mass model parameters
            corresponding to each mass plane.
        kind : str, optional
            either "auto" or "direct". Determines how the distortion matrix is
            computed, "auto" will using automatic differentiation using using JAX's
            `jaxfwd` function. "direct" will using the `hessian` method for each
            mass plane. "auto" is typically faster and is the default and recommended
            method.

        Returns
        -------
        jax.numpy array
            The area distortion matrix of the lens for each position
            and each mass plane (including the image plane) with shape
            (N+1, *(x.shape), 2, 2) where N is the number of mass planes.
        '''
        A_stack_part = partial(
            self._A_stack,
            eta_flat=eta_flat,
            kwargs=kwargs,
            kind=kind
        )
        return jnp.moveaxis(jnp.vectorize(
            A_stack_part,
            signature='(),()->(i,j,i)'
        )(x, y), -2, 0)

    def inverse_magnification(self, x, y, eta_flat, kwargs, kind='auto'):
        '''Return the inverse magnification map for each plane of the lens
        system.

        Parameters
        ----------
        x : jax.numpy array
            x-position (preferentially arcsec)
        y : jax.numpy array
            x-position (preferentially arcsec)
        eta_flat : jax.numpy array
            upper triangular elements of eta matrix, values defined as
            eta_ij = D_ij D_i+1 / D_j D_ii+1 where D_ij is the angular diameter
            distance between redshifts i and j. Only include values where
            j > i+1. This convention implies that all einstein radii are defined
            with respect to the **next** mass plane back (**not** the last plane in
            the stack).
        kwargs : list of list
            List of lists of parameter dictionaries of lens mass model parameters
            corresponding to each mass plane.
        kind : str, optional
            either "auto" or "direct". Determines how the distortion matrix is
            computed, "auto" will using automatic differentiation using using JAX's
            `jaxfwd` function. "direct" will using the `hessian` method for each
            mass plane. "auto" is typically faster and is the default and recommended
            method.

        Returns
        -------
        jax.numpy array
            The inverse magnification of the lens for each position
            and each mass plane (including the image plane) with shape
            (N+1, *(x.shape)) where N is the number of mass planes.
        '''
        A = self.A(x, y, eta_flat, kwargs, kind=kind)
        return A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]

    def kappa(self, x, y, eta_flat, kwargs, kind='auto'):
        '''Lensing convergence k = 1/2 laplacian(phi) map for each plane of the lens
        system.

        Parameters
        ----------
        x : jax.numpy array
            x-position (preferentially arcsec)
        y : jax.numpy array
            x-position (preferentially arcsec)
        eta_flat : jax.numpy array
            upper triangular elements of eta matrix, values defined as
            eta_ij = D_ij D_i+1 / D_j D_ii+1 where D_ij is the angular diameter
            distance between redshifts i and j. Only include values where
            j > i+1. This convention implies that all einstein radii are defined
            with respect to the **next** mass plane back (**not** the last plane in
            the stack).
        kwargs : list of list
            List of lists of parameter dictionaries of lens mass model parameters
            corresponding to each mass plane.
        kind : str, optional
            either "auto" or "direct". Determines how the distortion matrix is
            computed, "auto" will using automatic differentiation using using JAX's
            `jaxfwd` function. "direct" will using the `hessian` method for each
            mass plane. "auto" is typically faster and is the default and recommended
            method.

        Returns
        -------
        jax.numpy array
            The lensing convergence for each position and each mass plane (including the
            image plane) with shape (N+1, *(x.shape)) where N is the number of mass planes.
        '''
        A = self.A(x, y, eta_flat, kwargs, kind=kind)
        return 1 - 0.5 * (A[..., 0, 0] + A[..., 1, 1])

    def gamma(self, x, y, eta_flat, kwargs, kind='auto'):
        '''shear computation for each plane of the lens system
        g1 = 1/2(d^2phi/dx^2 - d^2phi/dy^2)
        g2 = d^2phi/dxdy

        Parameters
        ----------
        x : jax.numpy array
            x-position (preferentially arcsec)
        y : jax.numpy array
            x-position (preferentially arcsec)
        eta_flat : jax.numpy array
            upper triangular elements of eta matrix, values defined as
            eta_ij = D_ij D_i+1 / D_j D_ii+1 where D_ij is the angular diameter
            distance between redshifts i and j. Only include values where
            j > i+1. This convention implies that all einstein radii are defined
            with respect to the **next** mass plane back (**not** the last plane in
            the stack).
        kwargs : list of list
            List of lists of parameter dictionaries of lens mass model parameters
            corresponding to each mass plane.
        kind : str, optional
            either "auto" or "direct". Determines how the distortion matrix is
            computed, "auto" will using automatic differentiation using using JAX's
            `jaxfwd` function. "direct" will using the `hessian` method for each
            mass plane. "auto" is typically faster and is the default and recommended
            method.

        Returns
        -------
        gamma1 : jax.numpy array
            The first shear component for each position and each mass plane (including the
            image plane) with shape (N+1, *(x.shape)) where N is the number of mass planes.
        gamma2 : jax.numpy array
            The second shear component for each position and each mass plane (including the
            image plane) with shape (N+1, *(x.shape)) where N is the number of mass planes.
        '''
        A = self.A(x, y, eta_flat, kwargs, kind=kind)
        gamma1 = 0.5 * (A[..., 1, 1] - A[..., 0, 0])
        gamma2 = -A[..., 0, 1]
        return gamma1, gamma2
