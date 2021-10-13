import pytest
import unittest
import numpy as np
import numpy.testing as npt
import scipy.ndimage.interpolation as interp

from herculens.Util import util, image_util, kernel_util
from herculens.LightModel.Profiles.gaussian import Gaussian
gaussian = Gaussian()


def test_fwhm_kernel():
    x_grid, y_gird = util.make_grid(101, 1)
    sigma = 20

    flux = gaussian.function(x_grid, y_gird, amp=1, sigma=sigma)
    kernel = util.array2image(flux)
    kernel = kernel_util.kernel_norm(kernel)
    fwhm_kernel = kernel_util.fwhm_kernel(kernel)
    fwhm = util.sigma2fwhm(sigma)
    npt.assert_almost_equal(fwhm/fwhm_kernel, 1, 2)

def test_cut_psf():
    image = np.ones((7, 7))
    psf_cut = kernel_util.cut_psf(image, 5)
    assert len(psf_cut) == 5

def test_pixel_kernel():
    # point source kernel
    kernel_size = 9
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[4, 4] = 1.
    pixel_kernel = kernel_util.pixel_kernel(point_source_kernel=kernel, subgrid_res=1)
    assert pixel_kernel[4, 4] == kernel[4, 4]

    pixel_kernel = kernel_util.pixel_kernel(point_source_kernel=kernel, subgrid_res=11)
    npt.assert_almost_equal(pixel_kernel[4, 4], 0.65841, decimal=3)

def test_split_kernel():
    kernel = np.zeros((9, 9))
    kernel[4, 4] = 1
    subgrid_res = 3
    subgrid_kernel = kernel_util.subgrid_kernel(kernel, subgrid_res=subgrid_res, odd=True)
    subsampling_size = 3
    kernel_hole, kernel_cutout = kernel_util.split_kernel(subgrid_kernel, supersampling_kernel_size=subsampling_size,
                                                          supersampling_factor=subgrid_res)

    assert kernel_hole[4, 4] == 0
    assert len(kernel_cutout) == subgrid_res*subsampling_size
    npt.assert_almost_equal(np.sum(kernel_hole) + np.sum(kernel_cutout), 1, decimal=4)

    subgrid_res = 2
    subgrid_kernel = kernel_util.subgrid_kernel(kernel, subgrid_res=subgrid_res, odd=True)
    subsampling_size = 3
    kernel_hole, kernel_cutout = kernel_util.split_kernel(subgrid_kernel, supersampling_kernel_size=subsampling_size,
                                                          supersampling_factor=subgrid_res)

    assert kernel_hole[4, 4] == 0
    assert len(kernel_cutout) == subgrid_res * subsampling_size + 1
    npt.assert_almost_equal(np.sum(kernel_hole) + np.sum(kernel_cutout), 1, decimal=4)

def test_subgrid_kernel():

    kernel = np.zeros((9, 9))
    kernel[4, 4] = 1
    subgrid_res = 3
    subgrid_kernel = kernel_util.subgrid_kernel(kernel, subgrid_res=subgrid_res, odd=True)
    kernel_re_sized = image_util.re_size(subgrid_kernel, factor=subgrid_res) *subgrid_res**2
    #import matplotlib.pyplot as plt
    #plt.matshow(kernel); plt.show()
    #plt.matshow(subgrid_kernel); plt.show()
    #plt.matshow(kernel_re_sized);plt.show()
    #plt.matshow(kernel_re_sized- kernel);plt.show()
    npt.assert_almost_equal(kernel_re_sized[4, 4], 1, decimal=2)
    assert np.max(subgrid_kernel) == subgrid_kernel[13, 13]
    #assert kernel_re_sized[4, 4] == 1

def test_averaging_even_kernel():
    subgrid_res = 4

    x_grid, y_gird = util.make_grid(19, 1., 1)
    sigma = 1.5
    flux = gaussian.function(x_grid, y_gird, amp=1, sigma=sigma)
    kernel_super = util.array2image(flux)

    kernel_pixel = kernel_util.averaging_even_kernel(kernel_super, subgrid_res)
    npt.assert_almost_equal(np.sum(kernel_pixel), 1, decimal=5)
    assert len(kernel_pixel) == 5

    x_grid, y_gird = util.make_grid(17, 1., 1)
    sigma = 1.5
    flux = gaussian.function(x_grid, y_gird, amp=1, sigma=sigma)
    kernel_super = util.array2image(flux)

    kernel_pixel = kernel_util.averaging_even_kernel(kernel_super, subgrid_res)
    npt.assert_almost_equal(np.sum(kernel_pixel), 1, decimal=5)
    assert len(kernel_pixel) == 5

def test_degrade_kernel():
    subgrid_res = 2

    x_grid, y_gird = util.make_grid(19, 1., 1)
    sigma = 1.5
    flux = gaussian.function(x_grid, y_gird, amp=1, sigma=sigma)
    kernel_super = util.array2image(flux)/np.sum(flux)

    kernel_degraded = kernel_util.degrade_kernel(kernel_super, degrading_factor=subgrid_res)
    npt.assert_almost_equal(np.sum(kernel_degraded), 1, decimal=8)

    kernel_degraded = kernel_util.degrade_kernel(kernel_super, degrading_factor=3)
    npt.assert_almost_equal(np.sum(kernel_degraded), 1, decimal=8)

    kernel_degraded = kernel_util.degrade_kernel(kernel_super, degrading_factor=1)
    npt.assert_almost_equal(np.sum(kernel_degraded), 1, decimal=8)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            kernel_super = np.ones((9, 9))
            kernel_util.split_kernel(kernel_super, supersampling_kernel_size=2, supersampling_factor=3)
        with self.assertRaises(ValueError):
            kernel_util.split_kernel(kernel_super, supersampling_kernel_size=3, supersampling_factor=0)
        with self.assertRaises(ValueError):
            kernel_util.fwhm_kernel(kernel=np.ones((4, 4)))
        with self.assertRaises(ValueError):
            kernel_util.fwhm_kernel(kernel=np.ones((5, 5)))


if __name__ == '__main__':
    pytest.main()
