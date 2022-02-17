import unittest

from GRF_perturbations.Modules.Utils import *
from GRF_perturbations.Modules.GRF_inhomogeneities_class import GRF_inhomogeneities_class
from GRF_perturbations.Modules.Surface_Brightness_class import Surface_brightness_class
import scipy

class test_Utils(unittest.TestCase):

    @classmethod
    def setUpClass(self):
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
        logA_array = [-9.,-8.5, -8.,-7.5, -7.]
        Beta_array = [0,1, 2,3, 4]

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

        # fitted logA change like mock logA (different normalisations though)
        for i in range(len(logA_array)-1):
            self.assertTrue(np.allclose(spectrum_logAs[i+1]-spectrum_logAs[i],0.5))

        # Check that fitted Beta is independent of logA
        for i in range(len(logA_array)-1):
            self.assertTrue(np.allclose(spectrum_Betas[i+1].flatten() - spectrum_Betas[i].flatten(), 0,atol=5e-5))

        # rearranged tensor (logA*phi,Beta)
        Spec_subjects_treatments=np.transpose(spectrum_Betas, axes=[0, 2, 1]).flatten().reshape((len(logA_array)*10, len(Beta_array)))
        # Check the trend that fitted Beta grows with mock Beta
        self.assertLess(scipy.stats.page_trend_test(Spec_subjects_treatments).pvalue, 0.05)

    def test_jax_map(self):
        np.random.seed(42)
        tensor = np.random.normal(size=(10, 2, 2))
        #test that maps over first dimension
        self.assertTrue(np.allclose(jax_map(lambda x: x.sum(),tensor),
                                    tensor.sum(axis=(1,2))))

        def func(tensor):
            reduced_tensor = jax_map(lambda x: x.sum(), tensor)
            power_sum = jnp.power(reduced_tensor, 2).sum()
            return power_sum

        gradients=jax.grad(func)(tensor)
        analytic_gradients=tensor.sum(axis=(1,2))*2
        analytic_gradients_tensor=np.repeat(analytic_gradients,4).reshape((10,2,2))

        #Test that gradients are correct
        self.assertTrue(np.allclose(gradients,analytic_gradients_tensor))

    def test_scipy_fit_surface_brightness(self):

        # Functions for image generation
        simulate_unperturbed_image = self.Surface_brightness.unperturbed_image_getter
        simulate_unperturbed_image_noiseless = lambda model_kwargs: simulate_unperturbed_image(model_kwargs,Noise_flag=False)
        simulate_perturbed_image = self.Surface_brightness.perturbed_image_getter

        # Images for fitting to be tested on
        Image_unperturbed_noiseless=simulate_unperturbed_image_noiseless(self.Surface_brightness.kwargs_unperturbed_model)
        Image_unperturbed_noisy=simulate_unperturbed_image(self.Surface_brightness.kwargs_unperturbed_model,Noise_flag=True,noise_seed=18)
        GRF_potential = self.GRF_class.potential([-7.8, 2], self.GRF_class.tensor_unit_Fourier_images[0])
        Image_perturbed_noisy=simulate_perturbed_image(GRF_potential,Noise_flag=True,noise_seed=42)

        # args<->kwargs transformation
        SL_parameters = self.Surface_brightness.parameters()
        #Chi^2 loss function
        def Loss_function(args, data):
            kwargs = SL_parameters.args2kwargs(args)
            model = simulate_unperturbed_image_noiseless(kwargs)
            # Chi^2 loss
            return jnp.mean((data - model) ** 2 / self.Surface_brightness.noise_var)

        losses = np.zeros(3)
        for i, data in enumerate([Image_unperturbed_noiseless, Image_unperturbed_noisy, Image_perturbed_noisy]):
            fit = scipy_fit_Surface_Brightness(data, self.Surface_brightness,method='Newton-CG')
            losses[i] = Loss_function(SL_parameters.kwargs2args(fit), data)

        self.assertAlmostEqual(losses[0],0,msg='Noiseless unperturbed image should be fitted perfectly')
        self.assertTrue(np.isclose(losses[1], 1,rtol=0.2), msg='Noisy unperturbed image should result in chi^2 close to 1')
        self.assertGreater(losses[2],1,msg='Perturbed image should result in chi^2>1, since the model has no power to describe perturbations')
        self.assertLess(losses[2], 3, msg='In the perturbation limit the model should still result in relevant description of the image')

if __name__ == '__main__':
    unittest.main()
