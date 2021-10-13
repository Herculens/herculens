import pytest
import unittest
import numpy as np
import numpy.testing as npt
from herculens.Util import util, image_util


def test_add_layer2image_odd_odd():
    grid2d = np.zeros((101, 101))
    kernel = np.zeros((21, 21))
    kernel[10, 10] = 1
    x_pos = 50
    y_pos = 50
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[50, 50] == 1
    assert added[49, 49] == 0

    x_pos = 70
    y_pos = 95
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)

    assert added[95, 70] == 1

    x_pos = 20
    y_pos = 45
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[45, 20] == 1

    x_pos = 45
    y_pos = 20
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[20, 45] == 1

    x_pos = 20
    y_pos = 55
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[55, 20] == 1

    x_pos = 20
    y_pos = 100
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[100, 20] == 1

    x_pos = 20.5
    y_pos = 100
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=1)
    assert added[100, 20] == 0.5
    assert added[100, 21] == 0.5

def test_add_layer2image_int():
    grid2d = np.zeros((7, 7))
    x_pos, y_pos = 4, 1
    kernel = np.ones((3, 3))
    added = image_util.add_layer2image_int(grid2d, x_pos, y_pos, kernel)
    print(added)
    assert added[0, 0] == 0
    assert added[0, 3] == 1

    added = image_util.add_layer2image_int(grid2d, x_pos + 10, y_pos, kernel)
    print(added)
    npt.assert_almost_equal(grid2d, added, decimal=9)

def test_add_background():
    image = np.ones((10, 10))
    sigma_bkgd = 1.
    image_noisy = image_util.add_background(image, sigma_bkgd)
    assert abs(np.sum(image_noisy)) < np.sqrt(np.sum(image)*sigma_bkgd)*3

def test_add_poisson():
    image = np.ones((100, 100))
    exp_time = 100.
    poisson = image_util.add_poisson(image, exp_time)
    assert abs(np.sum(poisson)) < np.sqrt(np.sum(image)/exp_time)*10

def test_re_size_array():
    numPix = 9
    kernel = np.zeros((numPix, numPix))
    kernel[int((numPix-1)/2), int((numPix-1)/2)] = 1
    subgrid_res = 2
    input_values = kernel
    x_in = np.linspace(0, 1, numPix)
    x_out = np.linspace(0, 1, numPix*subgrid_res)
    out_values = image_util.re_size_array(x_in, x_in, input_values, x_out, x_out)
    kernel_out = out_values
    assert kernel_out[int((numPix*subgrid_res-1)/2), int((numPix*subgrid_res-1)/2)] == 0.58477508650519028

def test_cut_edges():
    image = np.zeros((51,51))
    image[25][25] = 1
    numPix = 21
    resized = image_util.cut_edges(image, numPix)
    nx, ny = resized.shape
    assert nx == numPix
    assert ny == numPix
    assert resized[10][10] == 1

    image = np.zeros((5, 5))
    image[2, 2] = 1
    numPix = 3
    image_cut = image_util.cut_edges(image, numPix)
    assert len(image_cut) == numPix
    assert image_cut[1, 1] == 1

    image = np.zeros((6, 6))
    image[3, 2] = 1
    numPix = 4
    image_cut = image_util.cut_edges(image, numPix)
    assert len(image_cut) == numPix
    assert image_cut[2, 1] == 1

    image = np.zeros((6, 8))
    image[3, 2] = 1
    numPix = 4
    image_cut = image_util.cut_edges(image, numPix)
    assert len(image_cut) == numPix
    assert image_cut[2, 0] == 1

def test_re_size():
    grid = np.zeros((200, 100))
    grid[100, 50] = 4
    grid_small = image_util.re_size(grid, factor=2)
    assert grid_small[50][25] == 1
    grid_same = image_util.re_size(grid, factor=1)
    npt.assert_equal(grid_same, grid)


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            grid2d = np.zeros((7, 7))
            x_pos, y_pos = 4, 1
            kernel = np.ones((2, 2))
            added = image_util.add_layer2image_int(grid2d, x_pos, y_pos, kernel)
        with self.assertRaises(ValueError):
            image = np.ones((5, 5))
            image_util.re_size(image, factor=2)
        with self.assertRaises(ValueError):
            image = np.ones((5, 5))
            image_util.re_size(image, factor=0.5)
        with self.assertRaises(ValueError):
            image = np.ones((5, 5))
            image_util.cut_edges(image, numPix=7)
        with self.assertRaises(ValueError):
            image = np.ones((5, 6))
            image_util.cut_edges(image, numPix=3)
        with self.assertRaises(ValueError):
            image = np.ones((5, 5))
            image_util.cut_edges(image, numPix=2)


if __name__ == '__main__':
    pytest.main()
