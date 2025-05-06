# Utility functions
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the Util module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'dangilman', 'aymgal'

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import interpolation as interp
from jax import random
import jax.numpy as jnp

from objax import functional
from objax.constants import ConvPadding
# from herculens.Util.jax_util import BilinearInterpolator


def add_layer2image(grid2d, x_pos, y_pos, kernel, order=1):
    """
    adds a kernel on the grid2d image at position x_pos, y_pos with an interpolated subgrid pixel shift of order=order
    :param grid2d: 2d pixel grid (i.e. image)
    :param x_pos: x-position center (pixel coordinate) of the layer to be added
    :param y_pos: y-position center (pixel coordinate) of the layer to be added
    :param kernel: the layer to be added to the image
    :param order: interpolation order for sub-pixel shift of the kernel to be added
    :return: image with added layer, cut to original size
    """

    x_int = int(round(x_pos))
    y_int = int(round(y_pos))
    shift_x = x_int - x_pos
    shift_y = y_int - y_pos
    kernel_shifted = interp.shift(kernel, [-shift_y, -shift_x], order=order)
    return add_layer2image_int(grid2d, x_int, y_int, kernel_shifted)

def add_layer2image_int(grid2d, x_pos, y_pos, kernel):
    """
    adds a kernel on the grid2d image at position x_pos, y_pos at integer positions of pixel
    :param grid2d: 2d pixel grid (i.e. image)
    :param x_pos: x-position center (pixel coordinate) of the layer to be added
    :param y_pos: y-position center (pixel coordinate) of the layer to be added
    :param kernel: the layer to be added to the image
    :return: image with added layer
    """
    nx, ny = np.shape(kernel)
    if nx % 2 == 0:
        raise ValueError("kernel needs odd numbers of pixels")

    num_x, num_y = np.shape(grid2d)
    x_int = int(round(x_pos))
    y_int = int(round(y_pos))

    k_x, k_y = np.shape(kernel)
    k_l2_x = int((k_x - 1) / 2)
    k_l2_y = int((k_y - 1) / 2)

    min_x = np.maximum(0, x_int-k_l2_x)
    min_y = np.maximum(0, y_int-k_l2_y)
    max_x = np.minimum(num_x, x_int+k_l2_x + 1)
    max_y = np.minimum(num_y, y_int+k_l2_y + 1)

    min_xk = np.maximum(0, -x_int + k_l2_x)
    min_yk = np.maximum(0, -y_int + k_l2_y)
    max_xk = np.minimum(k_x, -x_int + k_l2_x + num_x)
    max_yk = np.minimum(k_y, -y_int + k_l2_y + num_y)
    if min_x >= max_x or min_y >= max_y or min_xk >= max_xk or min_yk >= max_yk or (max_x-min_x != max_xk-min_xk) or (max_y-min_y != max_yk-min_yk):
        return grid2d
    kernel_re_sized = kernel[min_yk:max_yk, min_xk:max_xk]
    new = grid2d.copy()
    new[min_y:max_y, min_x:max_x] += kernel_re_sized
    return new

def add_background(image, sigma_bkd, seed):
    """
    adds background noise to image
    :param image: pixel values of image
    :param sigma_bkd: background noise (sigma)
    :return: a realisation of Gaussian noise of the same size as image
    """
    background = random.normal(seed, shape=np.shape(image)) * sigma_bkd
    # without JAX:
    # nx, ny = 
    # background = np.random.randn(*image.shape) * sigma_bkd
    return background

def add_poisson(image, exp_time, seed):
    """
    adds a poison (or Gaussian) distributed noise with mean given by surface brightness
    :param image: pixel values (photon counts per unit exposure time)
    :param exp_time: exposure time
    :return: Poisson noise realization of input image
    """
    """
    adds a poison (or Gaussian) distributed noise with mean given by surface brightness
    """
    sigma = jnp.sqrt(jnp.abs(image) / exp_time) # Gaussian approximation for Poisson distribution, normalized to exposure time
    poisson = random.normal(seed, shape=jnp.shape(image)) * sigma
    # without JAX:
    # sigma = np.sqrt(np.abs(image) / exp_time) # Gaussian approximation for Poisson distribution, normalized to exposure time
    # poisson = np.random.randn(*image.shape) * sigma
    return poisson

def cut_edges(image, numPix):
    """
    cuts out the edges of a 2d image and returns re-sized image to numPix
    center is well defined for odd pixel sizes.
    :param image: 2d numpy array
    :param numPix: square size of cut out image
    :return: cutout image with size numPix
    """
    nx, ny = image.shape
    if nx < numPix or ny < numPix:
        raise ValueError('image can not be resized, in routine cut_edges with image shape (%s %s) '
                         'and desired new shape (%s %s)' % (nx, ny, numPix, numPix))
    if (nx % 2 == 0 and ny % 2 == 1) or (nx % 2 == 1 and ny % 2 == 0):
        raise ValueError('image with odd and even axis (%s %s) not supported for re-sizeing' % (nx, ny))
    if (nx % 2 == 0 and numPix % 2 == 1) or (nx % 2 == 1 and numPix % 2 == 0):
        raise ValueError('image can only be re-sized from even to even or odd to odd number.')

    x_min = int((nx - numPix) / 2)
    y_min = int((ny - numPix) / 2)
    x_max = nx - x_min
    y_max = ny - y_min
    resized = image[x_min:x_max, y_min:y_max]
    # return copy.deepcopy(resized)
    return resized  # No need to copy, since JAX returns a new DeviceArray

def re_size(image, factor=1):
    """
    re-sizes image with nx x ny to nx/factor x ny/factor
    :param image: 2d image with shape (nx,ny)
    :param factor: integer >=1
    :return:
    """
    if factor < 1:
        raise ValueError('scaling factor in re-sizing %s < 1' %factor)
    elif factor == 1:
        return image
    f = int(factor)
    nx, ny = np.shape(image)
    if int(nx/f) == nx/f and int(ny/f) == ny/f:
        # small = image.reshape([int(nx/f), f, int(ny/f), f]).mean(3).mean(1)
        image_nchw = image[None, None, :, :]
        small_nchw = functional.average_pool_2d(image_nchw, size=f, padding=ConvPadding.SAME)
        small = jnp.squeeze(small_nchw)
        return small
    else:
        raise ValueError("scaling with factor %s is not possible with grid size %s, %s" %(f, nx, ny))

def re_size_array(x_in, y_in, input_values, x_out, y_out):
    """
    resizes 2d array (i.e. image) to new coordinates. So far only works with square output aligned with coordinate axis.
    :param x_in:
    :param y_in:
    :param input_values:
    :param x_out:
    :param y_out:
    :return:
    """
    interp_2d = RectBivariateSpline(x_in, y_in, z=input_values, kx=1, ky=1, s=0)
    out_values = interp_2d(x_out, y_out)
    return out_values
