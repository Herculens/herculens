import unittest

from GRF_perturbations.Modules.GRF_inhomogeneities_class import GRF_inhomogeneities_class
from GRF_perturbations.Modules.Surface_Brightness_class import *
import scipy as sc


class test_Surface_Brightness(unittest.TestCase):

    def setUp(self):
        self.GRF_class=GRF_inhomogeneities_class(100,0.08,100)
        self.GRF_class = Surface_brightness_class(100,0.08,0.1,200,2028)

    def test_noise_unperturbed_image(self):
        #Generate a lot of images and measure std of Poisson and bkg noise explicitly
        self.assertTrue(False)

    def test_noise_perturbed_image(self):
        # Generate a lot of images and measure std of Poisson and bkg noise explicitly
        self.assertTrue(False)

    def test_anomalies_from_perturbations(self):
        # In noiseless setup generate perturbed-unperturbed
        # Consider that power spectrum changes and amplitude changes
        self.assertTrue(False)

    def test_check_model(self):
        #Throw a lot of weird non-default models to test where it breaks
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
