import numpy as np
import pytest
import numpy.testing as npt
import unittest

from jaxtronomy.Util import util


def test_map_coord2pix():
    ra = 0
    dec = 0
    x_0 = 1
    y_0 = -1
    M = np.array([[1, 0], [0, 1]])
    x, y = util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x == 1
    assert y == -1

    ra = [0, 1, 2]
    dec = [0, 2, 1]
    x, y = util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x[0] == 1
    assert y[0] == -1
    assert x[1] == 2

    M = np.array([[0, 1], [1, 0]])
    x, y = util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x[1] == 3
    assert y[1] == 0

def test_make_grid():
    numPix = 11
    deltapix = 1.
    grid = util.make_grid(numPix, deltapix)
    assert grid[0][0] == -5
    assert np.sum(grid[0]) == 0
    x_grid, y_grid = util.make_grid(numPix, deltapix, subgrid_res=2.)
    print(np.sum(x_grid))
    assert np.sum(x_grid) == 0
    assert x_grid[0] == -5.25

    x_grid, y_grid = util.make_grid(numPix, deltapix, subgrid_res=1, left_lower=True)
    assert x_grid[0] == 0
    assert y_grid[0] == 0

def test_array2image():
    array = np.linspace(1, 100, 100)
    image = util.array2image(array)
    assert image[9][9] == 100
    assert image[0][9] == 10

def test_image2array():
    image = np.zeros((10,10))
    image[1,2] = 1
    array = util.image2array(image)
    assert array[12] == 1

def test_image2array2image():
    image = np.zeros((20, 10))
    nx, ny = np.shape(image)
    image[1, 2] = 1
    array = util.image2array(image)
    image_new = util.array2image(array, nx, ny)
    assert image_new[1, 2] == image[1, 2]

def test_make_grid_transform():
    numPix = 11
    theta = np.pi / 2
    deltaPix = 0.05
    Mpix2coord = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * deltaPix
    ra_coord, dec_coord = util.make_grid_transformed(numPix, Mpix2coord)
    ra2d = util.array2image(ra_coord)
    assert ra2d[5, 5] == 0
    assert ra2d[4, 5] == deltaPix
    npt.assert_almost_equal(ra2d[5, 4], 0, decimal=10)

def test_make_subgrid():
    numPix = 101
    deltapix = 1
    x_grid, y_grid = util.make_grid(numPix, deltapix, subgrid_res=1)
    x_sub_grid, y_sub_grid = util.make_subgrid(x_grid, y_grid, subgrid_res=2)
    assert np.sum(x_grid) == 0
    assert x_sub_grid[0] == -50.25
    assert y_sub_grid[17] == -50.25

    x_sub_grid_new, y_sub_grid_new = util.make_subgrid(x_grid, y_grid, subgrid_res=4)
    assert x_sub_grid_new[0] == -50.375

def test_fwhm2sigma():
    fwhm = 0.5
    sigma = util.fwhm2sigma(fwhm)
    assert sigma == fwhm/ (2 * np.sqrt(2 * np.log(2)))

def test_convert_bool_list():
    bool_list = util.convert_bool_list(n=10, k=None)
    assert len(bool_list) == 10
    assert bool_list[0] == True

    bool_list = util.convert_bool_list(n=10, k=3)
    assert len(bool_list) == 10
    assert bool_list[3] is True
    assert bool_list[2] is False

    bool_list = util.convert_bool_list(n=10, k=[3, 7])
    assert len(bool_list) == 10
    assert bool_list[3] is True
    assert bool_list[7] is True
    assert bool_list[2] is False

    bool_list = util.convert_bool_list(n=3, k=[False, False, True])
    assert len(bool_list) == 3
    assert bool_list[0] is False
    assert bool_list[1] is False
    assert bool_list[2] is True

    bool_list = util.convert_bool_list(n=3, k=[])
    assert len(bool_list) == 3
    assert bool_list[0] is False


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            array = np.ones(5)
            util.array2image(array)
        with self.assertRaises(ValueError):
            util.convert_bool_list(n=2, k=[3, 7])
        with self.assertRaises(ValueError):
            util.convert_bool_list(n=3, k=[True, True])
        with self.assertRaises(ValueError):
            util.convert_bool_list(n=2, k=[0.1, True])


if __name__ == '__main__':
    pytest.main()
