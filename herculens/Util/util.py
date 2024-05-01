# Utility functions
#
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the Util module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp
import json
import re


def rotate(xcoords, ycoords, angle):
    """Rotate points about the origin by an angle."""
    cos = jnp.cos(angle)
    sin = jnp.sin(angle)
    new_x =   xcoords * cos + ycoords * sin
    new_y = - xcoords * sin + ycoords * cos
    return new_x, new_y

def map_coord2pix(ra, dec, x_0, y_0, M):
    """Perform a linear transformation between two coordinate systems.

    Mainly used to transform angular into pixel coordinates in an image.
    :param ra: ra coordinates
    :param dec: dec coordinates
    :param x_0: pixel value in x-axis of ra,dec = 0,0
    :param y_0: pixel value in y-axis of ra,dec = 0,0
    :param M: 2x2 matrix to transform angular to pixel coordinates
    :return: transformed coordinate systems of input ra and dec
    """
    #x, y = M.dot(np.array([ra, dec]))
    x, y = jnp.array(M).dot(jnp.array([ra, dec]))
    return x + x_0, y + y_0

def array2image(array, nx=0, ny=0):
    """Convert a 1d array into a 2d array.

    Note: this only works when length of array is a perfect square, or else if
    nx and ny are provided

    :param array: 1d array of image values
    :type array: array of size n**2
    :returns:  2d array
    :raises: ValueError
    """
    if nx == 0 or ny == 0:
        # Avoid turning n into a JAX-traced object with jax.numpy.sqrt
        n = int(len(array)**0.5)
        if n**2 != len(array):
            err_msg = f"Input array size {len(array)} is not a perfect square."
            raise ValueError(err_msg)
        nx, ny = n, n
    image = array.reshape(int(nx), int(ny))
    return image

def image2array(image):
    """Convert a 2d array into a 1d array.

    :param array: image values
    :type array: array of size (n,n)
    :returns:  1d array
    :raises: AttributeError, KeyError
    """
    # nx, ny = image.shape  # find the size of the array
    # imgh = np.reshape(image, nx * ny)  # change the shape to be 1d
    # return imgh
    return image.ravel()

def make_grid(numPix, deltapix, subgrid_res=1, left_lower=False):
    """
    creates pixel grid (in 1d arrays of x- and y- positions)
    default coordinate frame is such that (0,0) is in the center of the coordinate grid

    :param numPix: number of pixels per axis
        Give an integers for a square grid, or a 2-length sequence
        (first, second axis length) for a non-square grid.
    :param deltapix: pixel size
    :param subgrid_res: sub-pixel resolution (default=1)
    :return: x, y position information in two 1d arrays
    """

    # Check numPix is an integer, or 2-sequence of integers
    if isinstance(numPix, (tuple, list, np.ndarray)):
        assert len(numPix) == 2
        if any(x != round(x) for x in numPix):
            raise ValueError("numPix contains non-integers: %s" % numPix)
        # numPix = np.asarray(numPix, dtype=int)
        numPix = list(numPix)
    else:
        if numPix != round(numPix):
            raise ValueError("Attempt to specify non-int numPix: %s" % numPix)
        # numPix = np.array([numPix, numPix], dtype=int)
        numPix = [numPix, numPix]

    # Super-resolution sampling
    # numPix_eff = int(numPix * subgrid_res)
    numPix_eff = [int(n * subgrid_res) for n in numPix]
    deltapix_eff = deltapix / float(subgrid_res)

    # Compute unshifted grids.
    # X values change quickly, Y values are repeated many times
    # NOTE jax.numpy.tile checks if `reps` is of type int, but numpy.int64
    #      is not in fact this type. Simply casting as int(numPix_eff[1])
    #      causes problems elsewhere with jax tracing, so we use another approach
    # x_grid = np.tile(np.arange(numPix_eff[0]), numPix_eff[1]) * deltapix_eff
    # y_grid = np.repeat(np.arange(numPix_eff[1]), numPix_eff[0]) * deltapix_eff
    x_space = np.arange(numPix_eff[0]) * deltapix_eff
    y_space = np.arange(numPix_eff[1]) * deltapix_eff
    x_grid, y_grid = np.meshgrid(x_space, y_space)
    x_grid, y_grid = x_grid.flatten(), y_grid.flatten()

    if left_lower is True:
        # Shift so (0, 0) is in the "lower left"
        # Note this does not shift when subgrid_res = 1
        shift = -1. / 2 + 1. / (2 * subgrid_res) * np.array([1, 1])
    else:
        # Shift so (0, 0) is centered
        # shift = deltapix_eff * (numPix_eff - 1) / 2
        shift = np.array([deltapix_eff * (n - 1) / 2 for n in numPix_eff])

    return x_grid - shift[0], y_grid - shift[1]

def make_grid_transformed(numPix, Mpix2Angle):
    """
    returns grid with linear transformation (deltaPix and rotation)
    :param numPix: number of Pixels
    :param Mpix2Angle: 2-by-2 matrix to mat a pixel to a coordinate
    :return: coordinate grid
    """
    x_grid, y_grid = make_grid(numPix, deltapix=1)
    ra_grid, dec_grid = map_coord2pix(x_grid, y_grid, 0, 0, Mpix2Angle)
    return ra_grid, dec_grid

def grid_from_coordinate_transform(nx, ny, Mpix2coord, ra_at_xy_0, dec_at_xy_0):
    """Return x and y coordinate grids that satisfy the coordinate system.

    :param nx: number of pixels in x-axis
    :param ny: number of pixels in y-axis
    :param Mpix2coord: transformation matrix (2x2) of pixels into coordinate displacements
    :param ra_at_xy_0: RA coordinate at (x,y) = (0,0)
    :param dec_at_xy_0: DEC coordinate at (x,y) = (0,0)
    :return: RA coordinate grid, DEC coordinate grid
    """
    a = np.arange(nx)
    b = np.arange(ny)
    matrix = np.dstack(np.meshgrid(a, b)).reshape(-1, 2)
    x_grid = matrix[:, 0]
    y_grid = matrix[:, 1]
    ra_grid = x_grid * Mpix2coord[0, 0] + y_grid * Mpix2coord[0, 1] + ra_at_xy_0
    dec_grid = x_grid * Mpix2coord[1, 0] + y_grid * Mpix2coord[1, 1] + dec_at_xy_0
    return ra_grid, dec_grid

def subgrid_from_coordinate_transform(nx, ny, Mpix2coord, ra_at_xy_0, dec_at_xy_0, subgrid_res=2):
    """

    :param nx: number of pixels in x-axis
    :param ny: number of pixels in y-axis
    :param Mpix2coord: transformation matrix (2x2) of pixels into coordinate displacements
    :param ra_at_xy_0: RA coordinate at (x,y) = (0,0)
    :param dec_at_xy_0: DEC coordinate at (x,y) = (0,0)
    :return: RA coordinate grid, DEC coordinate grid
    """
    if subgrid_res == 1:
        return grid_from_coordinate_transform(nx, ny, Mpix2coord, ra_at_xy_0, dec_at_xy_0)

    nx_sub, ny_sub = int(nx * subgrid_res), int(ny * subgrid_res)
    subgrid_res = float(subgrid_res)
    Mcoord2pix = np.linalg.inv(Mpix2coord)
    x_at_radec_0, y_at_radec_0 = map_coord2pix(ra_at_xy_0, dec_at_xy_0, x_0=0, y_0=0, M=Mcoord2pix)
    Mpix2coord_sub = Mpix2coord / subgrid_res
    x_at_radec_0_sub = x_at_radec_0 * subgrid_res - 0.5*(subgrid_res-1)
    y_at_radec_0_sub = y_at_radec_0 * subgrid_res - 0.5*(subgrid_res-1)
    ra_at_xy_0_sub, dec_at_xy_0_sub = map_coord2pix(x_at_radec_0_sub, y_at_radec_0_sub, x_0=0, y_0=0, M=Mpix2coord_sub)
    return grid_from_coordinate_transform(nx_sub, ny_sub, Mpix2coord_sub, ra_at_xy_0_sub, dec_at_xy_0_sub)

def averaging(grid, numGrid, numPix):
    """
    resize 2d pixel grid with numGrid to numPix and averages over the pixels
    :param grid: higher resolution pixel grid
    :param numGrid: number of pixels per axis in the high resolution input image
    :param numPix: lower number of pixels per axis in the output image (numGrid/numPix is integer number)
    :return:
    """

    Nbig = numGrid
    Nsmall = numPix
    small = grid.reshape([int(Nsmall), int(Nbig / Nsmall), int(Nsmall), int(Nbig / Nsmall)]).mean(3).mean(1)
    return small

def fwhm2sigma(fwhm):
    """

    :param fwhm: full-widt-half-max value
    :return: gaussian sigma (sqrt(var))
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma

def sigma2fwhm(sigma):
    """

    :param sigma:
    :return:
    """
    fwhm = sigma * (2 * np.sqrt(2 * np.log(2)))
    return fwhm

def make_subgrid(ra_coord, dec_coord, subgrid_res=2):
    """
    return a grid with subgrid resolution
    :param ra_coord:
    :param dec_coord:
    :param subgrid_res:
    :return:
    """
    # With JAX we need to avoid array item assignment through indexing

    ra = array2image(ra_coord)[0]  # First row containing all ra values
    n_ra, ra_min, ra_max = len(ra), ra[0], ra[-1]
    d_ra = (np.diff(ra)[0] / 2.) * (1. - 1. / subgrid_res)  # Desired spacing
    ra_new = np.linspace(ra_min - d_ra, ra_max + d_ra, subgrid_res * n_ra)

    dec = array2image(dec_coord)[:, 0]  # First column containing all dec values
    n_dec, dec_min, dec_max = len(dec), dec[0], dec[-1]
    d_dec = (np.diff(dec)[0] / 2.) * (1. - 1. / subgrid_res)
    dec_new = np.linspace(dec_min - d_dec, dec_max + d_dec, subgrid_res * n_dec)

    ra_subgrid, dec_subgrid = np.meshgrid(ra_new, dec_new)
    return image2array(ra_subgrid), image2array(dec_subgrid)

def convert_bool_list(n, k=None):
    """
    returns a bool list of the length of the lens models
    if k = None: returns bool list with True's
    if k is int, returns bool list with False's but k'th is True
    if k is a list of int, e.g. [0, 3, 5], returns a bool list with True's in the integers listed and False elsewhere
    if k is a boolean list, checks for size to match the numbers of models and returns it

    :param n: integer, total lenght of output boolean list
    :param k: None, int, or list of ints
    :return: bool list
    """
    if k is None:
        bool_list = [True] * n
    elif isinstance(k, (int, np.integer)):  # single integer
        bool_list = [False] * n
        bool_list[k] = True
    elif len(k) == 0:  # empty list
        bool_list = [False] * n
    elif isinstance(k[0], bool):
        if n != len(k):
            raise ValueError('length of selected lens models in format of boolean list is %s '
                             'and does not match the models of this class instance %s.' % (len(k), n))
        bool_list = k
    elif isinstance(k[0], (int, np.integer)):  # list of integers
        bool_list = [False] * n
        for i, k_i in enumerate(k):
            if k_i is not False:
                # if k_i is True:
                #    bool_list[i] = True
                if k_i < n:
                    bool_list[k_i] = True
                else:
                    raise ValueError("k as set by %s is not convertable in a bool string!" % k)
    else:
        raise ValueError('input list k as %s not compatible' % k)
    return bool_list

def check_psd_force_symmetry(X_in):
    """ensure that the matrix X is symmetric positive-semidefinite"""
    X = np.copy(X_in)
    # check if positive-semidefinite
    if not np.all(np.linalg.eigvals(X) > 0):
        min_eig = np.min(np.real(np.linalg.eigvals(X)))
        print(f"correcting for negative eigenvalues with {10*min_eig}")
        X -= 10*min_eig * np.eye(*X.shape)

    # check symmetry (sometimes condition for being positive-semidefinite)
    if not np.all(X - X.T == 0):
        print("forcing symmetry")
        lower_triangle = np.tril(X)
        X = lower_triangle + lower_triangle.T - np.diag(lower_triangle.diagonal())

    # final check
    assert (np.all(np.linalg.eigvals(X) > 0) and np.all(X - X.T == 0))
    return X

def read_json(input_path):
    with open(input_path,'r') as f:
        input_str = f.read()
        input_str = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "", input_str)
        input_str = re.sub(re.compile("//.*?\n" ), "", input_str)
        json_in   = json.loads(input_str)
    return json_in
