import unittest

from GRF_perturbations.Modules.Utils import *
from GRF_perturbations.Modules.GRF_inhomogeneities_class import GRF_inhomogeneities_class
from GRF_perturbations.Modules.Surface_Brightness_class import Surface_brightness_class
import scipy

class test_Utils(unittest.TestCase):

    def setUp(self):
        self.GRF_class = GRF_inhomogeneities_class(100, 0.08, 100)
        self.Surface_brightness = Surface_brightness_class(100, 0.08, 0.1, 200, 2028)

    def test_gradient_descent(self):
        x=np.array([1,2,3]).astype(float)
        initial_guess = np.array([0,0]).astype(float)
        #MSE loss
        Loss_function = lambda args,y: ((args[0] * x + args[1] - y) ** 2).sum()

        def get_fit(y):
            Loss = lambda args: Loss_function(args,y)
            gradient_function = jax.grad(Loss)

            args_fit = gradient_descent(gradient_function, initial_guess, 10, 0.05)
            return args_fit

        y = 10 * x + 1

        self.assertLess(Loss_function(get_fit(y),y),Loss_function(initial_guess,y),msg='Likelihood did not improve')

        # Gradients of maximum likelihood estimators with respect to the fitted data
        gradient_of_k_wrp_y = jax.grad(lambda y: get_fit(y)[0])
        gradient_of_b_wrp_y = jax.grad(lambda y: get_fit(y)[1])

        y1=np.array([1., -2., 3.])
        y2=np.array([3., 2, 1.])

        # Test that gradients are not zero
        self.assertTrue(all(np.abs(gradient_of_k_wrp_y(y1)) > 0),msg='Zero gradients of k')
        self.assertTrue(all(np.abs(gradient_of_b_wrp_y(y1)) > 0), msg='Zero gradients of b')

        # The loss gradient is linear wrp data, so grad of maximum likelihood estimators should not depend on the data
        self.assertTrue(np.allclose(gradient_of_k_wrp_y(y1),gradient_of_k_wrp_y(y2)),msg='dL/dk are linear wrp y, so grads should be equal')
        self.assertTrue(np.allclose(gradient_of_b_wrp_y(y1), gradient_of_b_wrp_y(y2)),msg='dL/dk are linear wrp b, so grads should be equal')

    def test_spectrum_radial_averaging(self):
        logA_array = [-9., -8., -7.]
        Beta_array = [0, 2, 4]

        independent_spectrum_index = self.Surface_brightness.pixel_number // 2
        k_grid_half = self.GRF_class.k_grid[:, :independent_spectrum_index]
        spectrum_logAs = np.zeros((len(logA_array),len(Beta_array), 10))
        spectrum_Betas = np.zeros((len(logA_array),len(Beta_array), 10))
        power_law_function = lambda k, logA, Beta: np.power(10, logA) * np.power(k, -Beta)

        for i,logA in enumerate(logA_array):
            for j,Beta in enumerate(Beta_array):
                for seed in range(10):
                    #100 variances for each logA,Beta from Parseval's theorem
                    GRF_field = self.GRF_class.potential([logA,Beta],self.GRF_class.tensor_unit_Fourier_images[seed])
                    Fourier_image_half = jnp.fft.fft2(GRF_field)[:, :independent_spectrum_index]
                    power_spectrum_half = jnp.abs(Fourier_image_half) ** 2

                    normalized_spectrum_half = power_spectrum_half / self.Surface_brightness.annulus_mask.sum()
                    Radial_power_spectrum = Spectrum_radial_averaging(normalized_spectrum_half, k_grid_half,
                                                                      self.Surface_brightness.frequencies)
                    fit_results, _ = scipy.optimize.curve_fit(power_law_function, self.Surface_brightness.frequencies,
                                                              Radial_power_spectrum)
                    spectrum_logAs[i, j, seed] = fit_results[0]
                    spectrum_Betas[i, j, seed] = fit_results[1]

        self.assertTrue(np.allclose(spectrum_logAs[2]-spectrum_logAs[1],1))
        self.assertTrue(np.allclose(spectrum_logAs[1] - spectrum_logAs[0], 1))

        # TODO: test for Beta



if __name__ == '__main__':
    unittest.main()
