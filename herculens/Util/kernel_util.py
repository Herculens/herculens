# Utility functions
#
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the Util module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'aymgal'


import copy
import numpy as np
# import scipy.ndimage.interpolation as interp
import herculens.Util.util as util
from herculens.Util import image_util
from herculens.LightModel.Profiles.gaussian import Gaussian


def kernel_norm(kernel):
    """

    :param kernel:
    :return: normalisation of the psf kernel
    """
    return kernel / np.sum(kernel)

def subgrid_kernel(kernel, subgrid_res, odd=False, num_iter=100):
    """
    creates a higher resolution kernel with subgrid resolution as an interpolation of the original kernel in an
    iterative approach

    :param kernel: initial kernel
    :param subgrid_res: subgrid resolution required
    :return: kernel with higher resolution (larger)
    """
    subgrid_res = int(subgrid_res)
    if subgrid_res == 1:
        return kernel
    nx, ny = np.shape(kernel)
    d_x = 1. / nx
    x_in = np.linspace(d_x/2, 1-d_x/2, nx)
    d_y = 1. / nx
    y_in = np.linspace(d_y/2, 1-d_y/2, ny)
    nx_new = nx * subgrid_res
    ny_new = ny * subgrid_res
    if odd is True:
        if nx_new % 2 == 0:
            nx_new -= 1
        if ny_new % 2 == 0:
            ny_new -= 1

    d_x_new = 1. / nx_new
    d_y_new = 1. / ny_new
    x_out = np.linspace(d_x_new/2., 1-d_x_new/2., nx_new)
    y_out = np.linspace(d_y_new/2., 1-d_y_new/2., ny_new)
    kernel_input = copy.deepcopy(kernel)
    kernel_subgrid = image_util.re_size_array(x_in, y_in, kernel_input, x_out, y_out)
    kernel_subgrid = kernel_norm(kernel_subgrid)
    for i in range(num_iter):
        # given a proposition, re-size it to original pixel size
        if subgrid_res % 2 == 0:
            kernel_pixel = averaging_even_kernel(kernel_subgrid, subgrid_res)
        else:
            kernel_pixel = util.averaging(kernel_subgrid, numGrid=nx_new, numPix=nx)
        delta = kernel - kernel_pixel
        temp_kernel = kernel_input + delta
        kernel_subgrid = image_util.re_size_array(x_in, y_in, temp_kernel, x_out, y_out)#/norm_subgrid
        kernel_subgrid = kernel_norm(kernel_subgrid)
        kernel_input = temp_kernel

    #from scipy.ndimage import zoom

    #ratio = subgrid_res
    #kernel_subgrid = zoom(kernel, ratio, order=4) / ratio ** 2
    #print(np.shape(kernel_subgrid))
    # whatever has not been matched is added to zeroth order (in squares of the undersampled PSF)
    if subgrid_res % 2 == 0:
        return kernel_subgrid
    kernel_pixel = util.averaging(kernel_subgrid, numGrid=nx_new, numPix=nx)
    kernel_pixel = kernel_norm(kernel_pixel)
    delta_kernel = kernel_pixel - kernel_norm(kernel)
    id = np.ones((subgrid_res, subgrid_res))
    delta_kernel_sub = np.kron(delta_kernel, id)/subgrid_res**2
    return kernel_norm(kernel_subgrid - delta_kernel_sub)

def averaging_even_kernel(kernel_high_res, subgrid_res):
    """
    makes a lower resolution kernel based on the kernel_high_res (odd numbers) and the subgrid_res (even number), both
    meant to be centered.

    :param kernel_high_res: high resolution kernel with even subsampling resolution, centered
    :param subgrid_res: subsampling resolution (even number)
    :return: averaged undersampling kernel
    """
    n_kernel_high_res = len(kernel_high_res)
    n_low = int(round(n_kernel_high_res / subgrid_res + 0.5))
    if n_low % 2 == 0:
        n_low += 1
    n_high = int(n_low * subgrid_res - 1)
    assert n_high % 2 == 1
    if n_high == n_kernel_high_res:
        kernel_high_res_edges = kernel_high_res
    else:
        i_start = int((n_high - n_kernel_high_res) / 2)
        kernel_high_res_edges = np.zeros((n_high, n_high))
        kernel_high_res_edges[i_start:-i_start, i_start:-i_start] = kernel_high_res
    kernel_low_res = np.zeros((n_low, n_low))
    # adding pixels that are fully within a single re-binned pixel
    for i in range(subgrid_res-1):
        for j in range(subgrid_res-1):
            kernel_low_res += kernel_high_res_edges[i::subgrid_res, j::subgrid_res]
    # adding half of a pixel that has over-lap with two pixels
    i = subgrid_res - 1
    for j in range(subgrid_res - 1):
        kernel_low_res[1:, :] += kernel_high_res_edges[i::subgrid_res, j::subgrid_res] / 2
        kernel_low_res[:-1, :] += kernel_high_res_edges[i::subgrid_res, j::subgrid_res] / 2
    j = subgrid_res - 1
    for i in range(subgrid_res - 1):
        kernel_low_res[:, 1:] += kernel_high_res_edges[i::subgrid_res, j::subgrid_res] / 2
        kernel_low_res[:, :-1] += kernel_high_res_edges[i::subgrid_res, j::subgrid_res] / 2
    # adding a quarter of a pixel value that is at the boarder of four pixels
    i = subgrid_res - 1
    j = subgrid_res - 1
    kernel_edge = kernel_high_res_edges[i::subgrid_res, j::subgrid_res]
    kernel_low_res[1:, 1:] += kernel_edge / 4
    kernel_low_res[:-1, 1:] += kernel_edge / 4
    kernel_low_res[1:, :-1] += kernel_edge / 4
    kernel_low_res[:-1, :-1] += kernel_edge / 4
    return kernel_low_res

def cut_psf(psf_data, psf_size):
    """
    cut the psf properly
    :param psf_data: image of PSF
    :param psf_size: size of psf
    :return: re-sized and re-normalized PSF
    """
    kernel = image_util.cut_edges(psf_data, psf_size)
    kernel = kernel_norm(kernel)
    return kernel

def kernel_gaussian(kernel_numPix, deltaPix, fwhm):
    sigma = util.fwhm2sigma(fwhm)
    #if kernel_numPix % 2 == 0:
    #    kernel_numPix += 1
    x_grid, y_grid = util.make_grid(kernel_numPix, deltaPix)
    gaussian = Gaussian()
    kernel = gaussian.function(x_grid, y_grid, amp=1., sigma=sigma, center_x=0, center_y=0)
    kernel /= np.sum(kernel)
    return kernel.reshape(kernel_numPix, kernel_numPix)

def split_kernel(kernel_super, supersampling_kernel_size, supersampling_factor, normalized=True):
    """
    pixel kernel and subsampling kernel such that the convolution of both applied on an image can be
    performed, i.e. smaller subsampling PSF and hole in larger PSF

    :param kernel: PSF kernel of the size of the pixel
    :param kernel_super: super-sampled kernel
    :param supersampling_kernel_size: size of super-sampled PSF in units of degraded pixels
    :param normalized: boolean, if True returns a split kernel that is area normalized=1 representing a convolution kernel
    :return: degraded kernel with hole and super-sampled kernel
    """
    if supersampling_factor <= 1:
        raise ValueError('To split a kernel, the supersampling_factor needs to be > 1, givn %s' %supersampling_factor)
    if supersampling_kernel_size % 2 == 0:
        raise ValueError('supersampling_kernel_size needs to be an odd number!')
    n_super = len(kernel_super)
    n_sub = supersampling_kernel_size * supersampling_factor
    if n_sub % 2 == 0:
        n_sub += 1
    if n_sub > n_super:
        n_sub = n_super

    kernel_hole = copy.deepcopy(kernel_super)
    n_min = int((n_super-1) / 2 - (n_sub - 1) / 2)
    n_max = int((n_super-1) / 2 + (n_sub - 1) / 2 + 1)
    kernel_hole[n_min:n_max, n_min:n_max] = 0
    kernel_hole_resized = degrade_kernel(kernel_hole, degrading_factor=supersampling_factor)
    kernel_subgrid_cut = kernel_super[n_min:n_max, n_min:n_max]
    if normalized is True:
        flux_subsampled = np.sum(kernel_subgrid_cut)
        flux_hole = np.sum(kernel_hole_resized)
        if flux_hole > 0:
            kernel_hole_resized *= (1. - flux_subsampled) / np.sum(kernel_hole_resized)
        else:
            kernel_subgrid_cut /= np.sum(kernel_subgrid_cut)
    else:
        kernel_hole_resized /= supersampling_factor ** 2
    return kernel_hole_resized, kernel_subgrid_cut

def degrade_kernel(kernel_super, degrading_factor):
    """

    :param kernel_super: higher resolution kernel (odd number per axis)
    :param degrading_factor: degrading factor (effectively the super-sampling resolution of the kernel given
    :return: degraded kernel with odd axis number with the sum of the flux/values in the kernel being preserved
    """
    if degrading_factor == 1:
        return kernel_super
    if degrading_factor % 2 == 0:
        kernel_low_res = averaging_even_kernel(kernel_super, degrading_factor)
    else:
        n_kernel = len(kernel_super)
        numPix = int(round(n_kernel / degrading_factor + 0.5))
        if numPix % 2 == 0:
            numPix += 1
        n_high = numPix * degrading_factor

        kernel_super_ = np.zeros((n_high, n_high))
        i_start = int((n_high-n_kernel)/2)
        kernel_super_[i_start:i_start+n_kernel, i_start:i_start+n_kernel] = kernel_super
        kernel_low_res = util.averaging(kernel_super_, numGrid=n_high, numPix=numPix) * degrading_factor**2  # multiplicative factor added when providing flux conservation
    return kernel_low_res

def fwhm_kernel(kernel):
    """

    :param kernel:
    :return:
    """
    n = len(kernel)
    center = (n - 1) / 2.
    I_r = image_util.radial_profile(kernel, center=[center, center])
    if n % 2 == 0:
        raise ValueError('only works with odd number of pixels in kernel!')
    max_flux = kernel[int((n-1)/2), int((n-1)/2)]
    I_2 = max_flux / 2.
    r = np.linspace(0, (n - 1) / 2, int((n + 1) / 2)) + 0.33
    for i in range(1, len(I_r)):
        if I_r[i] < I_2:
            fwhm_2 = (I_2 - I_r[i - 1]) / (I_r[i] - I_r[i - 1]) + r[i - 1]
            return fwhm_2 * 2
    raise ValueError('The kernel did not drop to half the max value - fwhm not determined!')
