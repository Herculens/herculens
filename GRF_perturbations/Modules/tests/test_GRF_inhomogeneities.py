import unittest

import numpy as np

from GRF_perturbations.Modules.GRF_inhomogeneities_class import *
import scipy
import os


class test_GRF_inhomogeneities(unittest.TestCase):
    """GRF_inhomogeneities class handles generation of isotropic GRF field that is a model of
     gravitational potential inhomogeneities introduced by galaxy satellites.
     It also generates the corresponding deflection and convergence fields
     as well as estimates the average variances of the fields using Parseval's theorem"""

    @classmethod
    def setUpClass(self):
        self.GRF_class = GRF_inhomogeneities_class(100, 0.08, 1000)

    def test_box_muller(self):
        """An algorithm to sample to independent standard normal random variates
        to be used as unit spectrum realisations of Real and Imaginary parts of GRF"""
        BM_variates = np.zeros((1000, 2))
        for i in range(1000):
            np.random.seed(i)
            BM_variates[i] = Box_Muller_transform()

        np.random.seed(42)

        # Random variates should be independent
        p_value_dependence = scipy.stats.pearsonr(BM_variates[:, 0], BM_variates[:, 1])[1]
        self.assertGreater(p_value_dependence, 0.5)

        # Test that the variates are normally distributed
        p_values_normality=scipy.stats.normaltest(BM_variates,axis=0).pvalue
        self.assertTrue((p_values_normality>0.5).all())

        # Standard normal means mean=0, std=1
        self.assertTrue(np.allclose(BM_variates.std(axis=0),1,atol=0.1))
        self.assertTrue(np.allclose(BM_variates.mean(axis=0), 0, atol=0.1))


    def test_sample_unit_fourier_image(self):
        """sample_unit_Fourier_image(random_seed) simulates a Fourier image for GRF Power_spectrum(k)=1
        based on Box-Muller transform with RNG defined by the 'random_seed'."""

        # Don't consider the points of Fourier image that should be real to keep the Configuration image real
        mask_full_variates = np.ones((100, 100), dtype=bool)
        mask_full_variates[0, 0] = False
        mask_full_variates[0, 50] = False
        mask_full_variates[50, 0] = False
        mask_full_variates[50, 50] = False


        # Real and Imaginary parts should be standard normal
        Fourier_values = (np.sqrt(2)) * self.GRF_class.tensor_unit_Fourier_images[:, mask_full_variates]

        # Total power is 1
        Total_powers=np.sum(np.abs(Fourier_values)**2,axis=1)/(2*100**2)
        self.assertTrue(np.allclose(Total_powers,1,atol=0.2))
        os.system('echo total power max= {:.5f}, total power min= {:.5f}'.format(Total_powers.max(),Total_powers.min()))
        p_values_normality_real = scipy.stats.normaltest(Fourier_values.real, axis=0).pvalue
        p_values_normality_imag = scipy.stats.normaltest(Fourier_values.imag, axis=0).pvalue
        os.system('echo pval max r= {:.5f}, pval min r= {:.5f}'.format(p_values_normality_real.max(), p_values_normality_real.min()))
        os.system('echo pval max i= {:.5f}, pval min i= {:.5f}'.format(p_values_normality_imag.max(), p_values_normality_imag.min()))


        self.assertTrue(np.allclose(Fourier_values.real.std(axis=1), 1., rtol=0.05), msg='Std is not 1')
        self.assertTrue(np.allclose(Fourier_values.imag.std(axis=1), 1., rtol=0.05), msg='Std is not 1')

    # TODO: test that mean is zero and that the image is actually real
    def test_potential(self):

        logA_array=[-9.,-8.,-7.]
        Beta_array=[0,2,4]

        theoretical_Variances=np.zeros((len(logA_array),len(Beta_array),100))
        generated_Variances=np.zeros((len(logA_array),len(Beta_array),100))

        for i,logA in enumerate(logA_array):
            for j,Beta in enumerate(Beta_array):
                #100 variances for each logA,Beta from Parseval's theorem
                theoretical_Variances[i,j]=np.repeat(self.GRF_class.field_variance([logA,Beta],field='potential'),100)
                #100 variances for each logA,Beta from actually sampled field
                Potentials = np.array([self.GRF_class.potential([logA,Beta],self.GRF_class.tensor_unit_Fourier_images[seed])\
                                       for seed in range(100)])
                generated_Variances[i,j]=Potentials.var(axis=(-1,-2))


        #Test that logVariances match
        self.assertTrue(np.allclose(np.log(generated_Variances),np.log(theoretical_Variances),rtol=0.25))

    def test_alpha(self):

        logA_array = [-9., -8., -7.]
        Beta_array = [0, 2, 4]

        theoretical_Variances = np.zeros((len(logA_array), len(Beta_array), 100, 2))
        generated_Variances = np.zeros((len(logA_array), len(Beta_array), 100, 2))

        for i, logA in enumerate(logA_array):
            for j, Beta in enumerate(Beta_array):
                # 100 variances for each logA,Beta from Parseval's theorem
                theoretical_Variances[i, j,:,0] = np.repeat(self.GRF_class.field_variance([logA, Beta], field='alpha_y'),100)
                theoretical_Variances[i, j,:,1] = np.repeat(self.GRF_class.field_variance([logA, Beta], field='alpha_x'),100)

                # 100 variances for each logA,Beta from actually sampled field
                alphas_y = np.array([self.GRF_class.alpha([logA, Beta], self.GRF_class.tensor_unit_Fourier_images[seed],\
                                                        direction='y') for seed in range(100)])
                alphas_x = np.array([self.GRF_class.alpha([logA, Beta], self.GRF_class.tensor_unit_Fourier_images[seed], \
                                                          direction='x') for seed in range(100)])

                generated_Variances[i, j,:,0] = alphas_y.var(axis=(-1, -2))
                generated_Variances[i, j, :, 1] = alphas_x.var(axis=(-1, -2))

        # Test that logVariances match
        self.assertTrue(np.allclose(np.log(generated_Variances), np.log(theoretical_Variances), rtol=0.25))

    def test_kappa(self):

        logA_array=[-9.,-8.,-7.]
        Beta_array=[0,2,4]

        theoretical_Variances=np.zeros((len(logA_array),len(Beta_array),100))
        generated_Variances=np.zeros((len(logA_array),len(Beta_array),100))

        for i,logA in enumerate(logA_array):
            for j,Beta in enumerate(Beta_array):
                #100 variances for each logA,Beta from Parseval's theorem
                theoretical_Variances[i,j]=np.repeat(self.GRF_class.field_variance([logA,Beta],field='kappa'),100)
                #100 variances for each logA,Beta from actually sampled field
                kappas = np.array([self.GRF_class.kappa([logA,Beta],self.GRF_class.tensor_unit_Fourier_images[seed])\
                                       for seed in range(100)])
                generated_Variances[i,j]=kappas.var(axis=(-1,-2))

        # Test that logVariances match
        self.assertTrue(np.allclose(np.log(generated_Variances), np.log(theoretical_Variances), rtol=0.25))

if __name__ == '__main__':
    unittest.main()
