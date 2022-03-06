import unittest

from GRF_perturbations.Modules.GRF_inhomogeneities_class import GRF_inhomogeneities_class
from GRF_perturbations.Modules.Surface_Brightness_class import *
from GRF_perturbations.Modules.Inference_class import Inference_class
import jax
import scipy
from tqdm import tqdm

class test_Surface_Brightness(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.GRF_class=GRF_inhomogeneities_class(100,0.08,100)
        self.Surface_brightness=Surface_brightness_class(100,0.08,0.1,200,2028)
        self.Inference = Inference_class(self.GRF_class, self.Surface_brightness, Grad_descent_max_iter=0)

    def test_check_model(self):
        #Empty model
        self.assertTrue(check_model([],[{}]))

        # single model, int/float parameters
        self.assertTrue(check_model([''], [{'':0}]))
        self.assertTrue(check_model([''], [{'':0.}]))

        # several models
        self.assertTrue(check_model(['','',''], [{},{'':0.},{'':0,'1':0}]))

        # different number of models and parameter dicts
        self.assertRaises(ValueError,check_model,['',''],[{'':0}])

        # model is not list of strings
        self.assertRaises(ValueError,check_model,1,[{'',0.}])
        self.assertRaises(ValueError, check_model, [1], [{'', 0.}])

        # kwargs is not list of dicts with string keys and numeric values
        self.assertRaises(ValueError, check_model, [''], {'': 0.})
        self.assertRaises(ValueError, check_model, [''], [{1: 0}])
        self.assertRaises(ValueError, check_model, [''], [{'': ''}])


    def test_noise_unperturbed_image(self):
        simulate_unperturbed_image = self.Surface_brightness.unperturbed_image_getter
        Images_unperturbed=np.array([simulate_unperturbed_image(self.Surface_brightness.kwargs_unperturbed_model,noise_seed=i)
                         for i in range(1000)])

        #Compare generated noise to requested noise map
        self.assertTrue(np.allclose(Images_unperturbed.var(axis=0),self.Surface_brightness.noise_var, rtol=0.25))

        #Peak SNR
        i, j = np.unravel_index(np.argmax(Images_unperturbed[0]), shape=Images_unperturbed[0].shape)
        bkg_noise_std=np.sqrt(self.Surface_brightness.noise_var[0,0])
        self.assertTrue(np.allclose(Images_unperturbed[:,i,j]/bkg_noise_std, 200, rtol=0.25))

    def test_anomalies_from_perturbations(self):
        # Unperturbed surface brightness
        simulate_unperturbed_image = self.Surface_brightness.unperturbed_image_getter
        unperturbed_Image = simulate_unperturbed_image(self.Surface_brightness.kwargs_unperturbed_model, Noise_flag=False)

        simulate_perturbed_image = self.Surface_brightness.perturbed_image_getter

        # Parameters of perturbations to be tested
        logA_array = [-8.5, -8.2, -7.8, -7.5, -7.3]
        Beta_array = [0, 1, 2, 3, 4]

        # Perturbed surface brightness mocks for tests
        Images = np.zeros((5, 100, 100, 100))
        #We will generate surface brighntess anomalies and fit their power spectra with power law
        #Since potential's logA and Beta grew, the ones of anomalies should also grow (simplistic but true)
        SB_anomalies_spectrum_logAs = np.zeros((5, 100))
        SB_anomalies_spectrum_Betas = np.zeros((5, 100))
        power_law_function=lambda k,logA,Beta: np.power(10,logA)*np.power(k,-Beta)

        for i in tqdm(range(5)):
            # 100 realisations for tests
            for seed in range(100):
                GRF_potential = self.GRF_class.potential([logA_array[i], Beta_array[i]],self.GRF_class.tensor_unit_Fourier_images[seed])
                Images[i, seed] = np.array([simulate_perturbed_image(GRF_potential, Noise_flag=False)])
                Surface_brightness_Anomalies_spectrum=self.Inference.compute_radial_spectrum(Images[i,seed]-unperturbed_Image)
                fit_results,_=scipy.optimize.curve_fit(power_law_function,self.Surface_brightness.frequencies,Surface_brightness_Anomalies_spectrum)
                SB_anomalies_spectrum_logAs[i,seed]=fit_results[0]
                SB_anomalies_spectrum_Betas[i,seed]=fit_results[1]

        #Conservation of total flux
        Total_flux=unperturbed_Image.sum()
        self.assertTrue(np.allclose(Images.sum(axis=(-1,-2,)).flatten(),np.repeat(Total_flux, 5 * 100), rtol=0.1))

        # Amplitude of anomalies should grow with amplitude of potential perturbations
        self.assertLess(scipy.stats.page_trend_test(SB_anomalies_spectrum_logAs.T).pvalue,0.05)
        # Spatial scale of anomalies should grow with spatial scale of potential perturbations
        self.assertLess(scipy.stats.page_trend_test(SB_anomalies_spectrum_Betas.T).pvalue, 0.05)
        # Quite a weird test, I know. But it will tell you if your generation of perturbations doesn't work or if you have mistaken with signs

if __name__ == '__main__':
    unittest.main()
