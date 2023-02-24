# Utility functions for regularization methods
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal'


import copy
import numpy as np
import time
from scipy import signal
import warnings

import jax
import jax.numpy as jnp
from jax import lax, vmap

from utax.wavelet import WaveletTransform
from herculens.LensImage.lensing_operator import LensingOperator
from herculens.Util import jax_util, vkl_util



def data_noise_to_wavelet_source(lens_image, kwargs_res, 
                                 wavelet_type_list=['starlet', 'battle-lemarie-3'],
                                 num_samples=10000, seed=0,
                                 starlet_second_gen=False, noise_var=None):

    # get the data noise
    nx, ny = lens_image.Grid.num_pixel_axes
    if noise_var is None:
        model_image = lens_image.model(**kwargs_res)
        diag_cov_d = lens_image.Noise.C_D_model(model_image)
    else:
        diag_cov_d = noise_var
    std_d = np.sqrt(diag_cov_d)[jnp.newaxis, :, :]
    
    # construct the lensing operator
    lensing_op = LensingOperator(lens_image.MassModel, 
                                 lens_image.Grid, # TODO: should be the model (from Numerics) grid at some point
                                 lens_image.SourceModel.pixel_grid)
    lensing_op.compute_mapping(kwargs_res['kwargs_lens'])

    # setup the transposed convolution
    kernel = jnp.copy(lens_image.PSF.kernel_point_source)
    nxk, nyk = kernel.shape
    kernel = kernel[:, :, jnp.newaxis, jnp.newaxis] 
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    dn = lax.conv_dimension_numbers((1, nx, ny, 1),  # NHWC
                                    kernel.shape,  # HWIO
                                    dimension_numbers)
    kernel_rot = jnp.rot90(jnp.rot90(kernel, axes=(0, 1)), axes=(0, 1))
    
    def B_T(n):
        res = lax.conv_general_dilated(n[jnp.newaxis, :, :, jnp.newaxis], 
                                       kernel_rot, 
                                       (1,1), #(k//2,k//2),  # window strides
                                       ((nxk//2, nyk//2), (nxk//2, nyk//2)), # padding mode
                                       (1,1), #(k//2,k//2),  # lhs/image dilation
                                       (1,1),  # rhs/kernel dilation
                                       dn)     # dimension_numbers = lhs, rhs, out dimension permutation
        return jnp.squeeze(res)
    
    def F_T(n):
        return lensing_op.image2source_2d(n)

    wavelet_class_list = []
    std_per_scale_list = []

    for wavelet_type in wavelet_type_list:

        # setup the wavelet transform
        nxsrc, nysrc = lens_image.SourceModel.pixel_grid.num_pixel_axes
        if 'battle-lemarie' in wavelet_type:
            nscales = 1  # we only care about the first scale for this one
        else:
            nscales = int(np.log2(min(nxsrc, nysrc)))  # max number of scales allowed
        wavelet = WaveletTransform(nscales, wavelet_type=wavelet_type, second_gen=starlet_second_gen)
        
        def Phi_T(n):
            return wavelet.decompose(n)
        
        @vmap
        def propagate_noise(n):
            # multiply by the inverse of the data (diagonal) covariance matrix
            n /= diag_cov_d
            # transposed convolution
            n = B_T(n)
            # de-lens to source plane
            n = F_T(n)
            # transform to wavelet space
            n = Phi_T(n)
            #debug = noise_i.copy()
            return n
        
        noise_samples = std_d * jax.random.normal(jax.random.PRNGKey(seed), 
                                                  shape=(num_samples, nx, ny))
        noise_samples_prop = propagate_noise(noise_samples)
        std_per_scale = jnp.std(noise_samples_prop, axis=0)

        wavelet_class_list.append(wavelet)
        std_per_scale_list.append(std_per_scale)

    return std_per_scale_list, wavelet_class_list



def data_noise_to_wavelet_potential(lens_image, kwargs_res, k_src=None,
                                    likelihood_type='chi2',
                                    wavelet_type_list=['starlet', 'battle-lemarie-3'], 
                                    starlet_second_gen=False,
                                    method='MC', num_samples=10000, seed=None, 
                                    ignore_poisson_noise=False, ignore_lens_light_flux=False,
                                    model_var_map=None, verbose=False):
    if likelihood_type not in ['l2_norm', 'chi2']:
        raise ValueError("Only 'l2_norm' and 'chi2' are supported options for likelihood_type.")

    mass_model = lens_image.MassModel
    kwargs_lens = kwargs_res['kwargs_lens']
    source_model = lens_image.SourceModel
    kwargs_source = kwargs_res['kwargs_source']

    # get model data variance
    model = lens_image.model(**kwargs_res, lens_light_add=(not ignore_lens_light_flux))
    if ignore_poisson_noise:
        data_var_map = lens_image.Noise.background_rms**2 * np.ones_like(model)
    else:
        data_var_map = lens_image.Noise.C_D_model(model, force_recompute=True)
    if model_var_map is not None:
        var_map = data_var_map + model_var_map  # add variances
    else:
        var_map = data_var_map

    var_map = np.array(var_map)  # cast to numpy array otherwise computations are slowed down
    std_map = np.sqrt(var_map)
    var_d = var_map.flatten()
    std_d = std_map.flatten()

    # extract coordinates grid, in image plane, ray-shot to source plane, and for the pixelated potential
    x_grid_d, y_grid_d = lens_image.Grid.pixel_coordinates
    x_grid_rs, y_grid_rs = mass_model.ray_shooting(x_grid_d, y_grid_d, kwargs_lens)
    x_grid_psi, y_grid_psi = lens_image.MassModel.pixel_grid.pixel_coordinates

    # number of pixels and wavelet scales
    nx_d, ny_d = x_grid_d.shape
    nx_psi, ny_psi = x_grid_psi.shape
    
    # compute derivatives of the source light at ray-shot coordinates
    source_deriv_x, source_deriv_y = source_model.spatial_derivatives(x_grid_rs, 
                                                                      y_grid_rs, 
                                                                      kwargs_source, k=k_src)
    source_deriv_x *= lens_image.Grid.pixel_area  # correct flux units
    source_deriv_y *= lens_image.Grid.pixel_area  # correct flux units

    # reshape all quantities for building the operator
    data_x = x_grid_d.flatten()
    data_y = np.flip(y_grid_d, axis=0).flatten()  # WARNING: y-coords must be flipped vertically!
    dpsi_x = x_grid_psi.flatten()
    dpsi_y = np.flip(y_grid_psi, axis=0).flatten()  # WARNING: y-coords must be flipped vertically!
    dpsi_dx = abs(x_grid_psi[0, 0] - x_grid_psi[0, 1])
    dpsi_dy = abs(y_grid_psi[0, 0] - y_grid_psi[1, 0])
    dpsi_xmin = x_grid_psi[0, :].min() - dpsi_dx/2.
    dpsi_ymax = y_grid_psi[:, 0].max() + dpsi_dy/2.
    dpsi_Nx, dpsi_Ny = nx_psi, ny_psi
    source0_dx = source_deriv_x.flatten()
    source0_dy = source_deriv_y.flatten()

    # get the DsD operator (see Koopmans+05)
    start = time.time()
    DsDpsi_matrix = vkl_util.vkl_operator(np.array(data_x), np.array(data_y),
                                          dpsi_xmin, dpsi_dx, dpsi_Nx, np.array(dpsi_x), 
                                          dpsi_ymax, dpsi_dy, dpsi_Ny, np.array(dpsi_y),
                                          np.array(source0_dx), np.array(source0_dy))
    if verbose: print("compute DsDpsi:", time.time()-start)

    # blurring operator for PSF convolutions
    start = time.time()
    B_matrix = lens_image.PSF.blurring_matrix(model.shape)
    if verbose: print("compute B:", time.time()-start)

    # D operator
    start = time.time()
    D_matrix  = - B_matrix.dot(DsDpsi_matrix)
    DT_matrix = D_matrix.T
    if verbose: print("compute D:", time.time()-start)

    # wavelet transform Phi^T operators
    PhiT_operator_list = []
    num_scales_list = []
    for wavelet_type in wavelet_type_list:
        if wavelet_type in ['battle-lemarie-1', 'battle-lemarie-3']:
            num_scales = 1  # we only care about the first scale for this one
        else:
            num_scales = int(np.log2(min(nx_psi, ny_psi)))
        wavelet = WaveletTransform(num_scales, wavelet_type=wavelet_type, 
                                   second_gen=starlet_second_gen)
        PhiT_operator_list.append(wavelet.decompose)
        num_scales_list.append(num_scales)


    if method == 'MC':

        # initialize random generator seed
        np.random.seed(seed)

        # draw samples from the data covariance matrix
        # cov_d = np.diag(var_d)
        # mu = np.zeros(nx_d*ny_d)  # zero mean
        # noise_reals = np.random.multivariate_normal(mu, cov_d, size=num_samples)
        
        # apply the operators for each realization of the noise
        psi_wt_std_list = []
        for wavelet_type, PhiT_operator in zip(wavelet_type_list, PhiT_operator_list):

            start = time.time()
            psi_wt_reals = []
            for i in range(num_samples):
                # noise_i = noise_reals[i, :]
                noise_i = std_d * np.random.randn(*std_d.shape)  # draw a noise realization

                # if chi2 loss, rescale by the data variance
                if likelihood_type == 'chi2':
                    noise_i /= var_d

                psi_i = DT_matrix.dot(noise_i)
                psi_wt_i = PhiT_operator( psi_i.reshape(nx_psi, ny_psi) )
                psi_wt_reals.append(psi_wt_i)
            psi_wt_reals = np.array(psi_wt_reals) # --> shape = (num_samples, num_scales, nx_psi, ny_psi)
            if verbose: print(f"loop over MC samples for wavelet '{wavelet_type}':", time.time()-start)

            # compute the variance per wavelet scale per potential pixel over all the samples
            psi_wt_var = np.var(psi_wt_reals, axis=0)
            # check
            if np.any(psi_wt_var < 0.):
                raise ValueError("Negative variance terms!")

            # convert to standard deviation
            psi_wt_std = np.sqrt(psi_wt_var)

            psi_wt_std_list.append(psi_wt_std)


    elif method == 'SLIT':

        psi_wt_std_list = []
        for PhiT_operator, num_scales in zip(PhiT_operator_list, num_scales_list):

            # following the same recipe as SLIT(ronomy):

            # first term of
            B_noise = std_map * np.sqrt(np.sum(psf_kernel_2d.T**2))
            B_noise = B_noise.flatten()
            DsDpsiB_noise = (DsDpsi_matrix.T).dot(B_noise)
            DsDpsiB_noise = DsDpsiB_noise.reshape(nx_psi, ny_psi)

            dirac = np.zeros((nx_psi, ny_psi))
            if nx_psi % 2 == 0:
                warnings.warn("Expect the potential noise maps to be shifted by one pixel (psi grid has even size!).")
            dirac[nx_psi//2, ny_psi//2] = 1
            dirac_wt = PhiT_operator(dirac)

            psi_wt_std = []
            for k in range(num_scales+1):
                psi_wt_std2_k = signal.fftconvolve(DsDpsiB_noise**2, dirac_wt[k]**2, mode='same')
                psi_wt_std_k = np.sqrt(psi_wt_std2_k)
                psi_wt_std.append(psi_wt_std_k)
            psi_wt_std = np.array(psi_wt_std) # --> shape = (num_scales, nx_psi, ny_psi)

            psi_wt_std_list.append(psi_wt_std)

    else:
        raise ValueError(f"Method '{method}' for noise propagation is not supported.")
    
    return psi_wt_std_list