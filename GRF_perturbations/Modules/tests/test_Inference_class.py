import unittest

from GRF_perturbations.Modules.Inference_class import *
from GRF_perturbations.Modules.Utils import jax_map
import scipy
import os
import math

#CPUs parallelization
import multiprocessing
max_thread_numbers=multiprocessing.cpu_count()
#os.system('echo CPU count %d'%)

#Parallelize on several CPUs
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%d"%(max_thread_numbers)

# TODO: comment general ideas for all the tests
class test_Inference(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.GRF_class = GRF_inhomogeneities_class(100, 0.08, 100)
        self.Surface_brightness = Surface_brightness_class(100, 0.08, 0.1, 200, 2028)
        self.Inference= Inference_class(self.GRF_class,self.Surface_brightness,Grad_descent_max_iter=400)

        # Spectrum to be fitted
        self.data_spectrum=self.Inference.Anomalies_Radial_Power_Spectrum([-7.8, 2],self.GRF_class.tensor_unit_Fourier_images[0])

        # Functions for image generation
        simulate_unperturbed_image = self.Surface_brightness.unperturbed_image_getter
        simulate_unperturbed_image_noiseless = lambda model_kwargs: simulate_unperturbed_image(model_kwargs,Noise_flag=False)
        simulate_perturbed_image = self.Surface_brightness.perturbed_image_getter

        # Images for fitting to be tested on
        self.Image_unperturbed_noiseless=simulate_unperturbed_image_noiseless(self.Surface_brightness.kwargs_unperturbed_model)
        self.Image_unperturbed_noisy=simulate_unperturbed_image(self.Surface_brightness.kwargs_unperturbed_model,Noise_flag=True,noise_seed=18)
        GRF_potential = self.GRF_class.potential([-7.8, 2], self.GRF_class.tensor_unit_Fourier_images[0])
        self.Image_perturbed_noisy=simulate_perturbed_image(GRF_potential,Noise_flag=True,noise_seed=42)
        # Precompute 32 spectra of Surface brightness anomalies
        compute_spectrum_pure = lambda unit_Fourier_image: self.Inference.Anomalies_Radial_Power_Spectrum(jnp.array([-7.8, 2.]), unit_Fourier_image)
        self.Perturbations_spectra=np.zeros((32,len(self.Surface_brightness.frequencies)))
        if (max_thread_numbers>=32):
            self.Perturbations_spectra=jax.pmap(compute_spectrum_pure)(self.GRF_class.tensor_unit_Fourier_images[:32])
        else:
            iterations_num=math.ceil(32/max_thread_numbers)
            for i in range(iterations_num):
                start_index=i*max_thread_numbers
                finish_index=np.minimum(32,(i+1)*max_thread_numbers)
                self.Perturbations_spectra[start_index:finish_index]=jax.pmap(compute_spectrum_pure)(self.GRF_class.tensor_unit_Fourier_images[start_index:finish_index])



    def test_Loss_unperturbed_model(self):

        # True arguments used to generate unperturbed Source-Lens model
        args_data=self.Inference.SL_parameters.kwargs2args(self.Surface_brightness.kwargs_unperturbed_model)

        Loss_perturbed_noisy = self.Inference.Loss_unperturbed_model(args_data,self.Image_perturbed_noisy)
        Loss_unperturbed_noisy = self.Inference.Loss_unperturbed_model(args_data,self.Image_unperturbed_noisy)
        Loss_unperturbed_noiseless = self.Inference.Loss_unperturbed_model(args_data, self.Image_unperturbed_noiseless)

        self.assertGreater(Loss_perturbed_noisy,Loss_unperturbed_noisy)
        self.assertGreater(Loss_unperturbed_noisy, Loss_unperturbed_noiseless)
        self.assertAlmostEqual(Loss_unperturbed_noiseless,0)


    def test_differentiable_fit_Surface_Brightness(self):

        args_data = self.Inference.SL_parameters.kwargs2args(self.Surface_brightness.kwargs_unperturbed_model)
        Loss_initial=self.Inference.Loss_unperturbed_model(args_data, self.Image_perturbed_noisy)

        args_max_likelihood=self.Inference.differentiable_fit_Surface_Brightness(self.Image_perturbed_noisy)
        Loss_final=self.Inference.Loss_unperturbed_model(args_max_likelihood, self.Image_perturbed_noisy)

        # Check that fitting improves model quality
        self.assertLess(Loss_final,Loss_initial)

        # Check that we can take gradients wrp fitted data
        Inference_few_steps = Inference_class(self.GRF_class, self.Surface_brightness, Grad_descent_max_iter=2)
        test_func= jax.jit(lambda data:  self.Inference.Loss_unperturbed_model(
                            Inference_few_steps.differentiable_fit_Surface_Brightness(data),data))
        Grads_of_Loss_final=jax.grad(test_func)(self.Image_perturbed_noisy)
        self.assertTrue((np.abs(Grads_of_Loss_final)>0).all())

    def test_compute_radial_spectrum(self):

        # Image of noise only
        PS_Noise=self.Inference.compute_radial_spectrum(self.Image_unperturbed_noisy-self.Image_unperturbed_noiseless)

        # Image of perturbations+noise
        PS_Perturbations =self.Inference.compute_radial_spectrum(self.Image_perturbed_noisy - self.Image_unperturbed_noisy)

        power_law_function = lambda k, logA, Beta: np.power(10, logA) * np.power(k, -Beta)

        fit_results, _ = scipy.optimize.curve_fit(power_law_function, self.Surface_brightness.frequencies,PS_Noise)
        logA_noise,Beta_noise = fit_results

        fit_results, _ = scipy.optimize.curve_fit(power_law_function, self.Surface_brightness.frequencies, PS_Perturbations)
        logA_Perturbations, Beta_Perturbations = fit_results

        # Perturbations+Noise should have more power than just noise
        self.assertGreater(PS_Perturbations.sum(),PS_Noise.sum())
        # Noise should have flat spectrum
        self.assertLess(Beta_noise, 0.1)
        # Spectrum of perturbations should not be flat
        self.assertGreater(Beta_Perturbations, 0.1)

    def test_Anomalies_Radial_Power_Spectrum(self):

        PS_noise=self.Inference.Anomalies_Radial_Power_Spectrum([-10,0],self.GRF_class.tensor_unit_Fourier_images[0])
        PS_perturbations_only=self.Inference.Anomalies_Radial_Power_Spectrum([-7.8,2],self.GRF_class.tensor_unit_Fourier_images[0],Noise_flag=False)

        power_law_function = lambda k, logA, Beta: np.power(10, logA) * np.power(k, -Beta)

        fit_results, _ = scipy.optimize.curve_fit(power_law_function, self.Surface_brightness.frequencies, PS_noise)
        _, Beta_noise = fit_results

        fit_results, _ = scipy.optimize.curve_fit(power_law_function, self.Surface_brightness.frequencies, PS_perturbations_only)
        _, Beta_pert_only = fit_results

        # Powerlaw fits of noisy perturbations
        Betas_pert_noise = np.zeros(32)
        for i in range(32):
            fit_results, _ = scipy.optimize.curve_fit(power_law_function, self.Surface_brightness.frequencies,
                                                      self.Perturbations_spectra[i])
            Betas_pert_noise[i] = fit_results[1]


        # Power with perturbations is greater
        self.assertTrue((self.Perturbations_spectra.sum(axis=1)>PS_noise.sum()).all())
        # Power with noise is greater
        self.assertTrue((self.Perturbations_spectra.sum(axis=1)>PS_perturbations_only.sum()).all())
        # Noise should have flat spectrum
        self.assertLess(Beta_noise, 0.1)
        # Spectrum of perturbations should not be flat
        self.assertGreater(Beta_pert_only, 0.1)

        # Noise should flatten the spectrum
        p_val_greater=scipy.stats.ttest_1samp(Betas_pert_noise,Beta_pert_only,alternative='greater').pvalue
        self.assertGreater(p_val_greater,0.95)

        Inference_few_steps = Inference_class(self.GRF_class, self.Surface_brightness, Grad_descent_max_iter=2)
        power_sum= jax.jit(lambda GRF_params: jnp.sum(Inference_few_steps.Anomalies_Radial_Power_Spectrum(
                                                        GRF_params,self.GRF_class.tensor_unit_Fourier_images[0])))       # Check differentiability
        Power_gradient=jax.grad(power_sum)([-7.8,2.])
        # Power of anomalies is correlated with amplitude of potential perturbations
        self.assertGreater(Power_gradient[0],0)
        # Power of anomalies is anticorrelated with power slope of potential perturbations
        self.assertLess(Power_gradient[1], 0)


    def test_GRF_Power_Spectrum_Loss(self):
        Mean_spectra = np.log(self.Perturbations_spectra).mean(axis=0)
        Std_spectra = np.sqrt(np.power(np.log(self.Perturbations_spectra) - Mean_spectra, 2).sum(axis=0) / (len(self.Perturbations_spectra) - 1))

        def Spectra_Loss(model_spectra, data_spectrum, Uncertainty):
            data_log_spectrum = jnp.log(data_spectrum)
            models_log_spectra = jnp.log(model_spectra)

            Mean_logN = models_log_spectra.mean(axis=-2)
            # Chi^2 loss for Normal likelihood of log(Power_spectrum)
            return jnp.mean(jnp.power((data_log_spectrum - Mean_logN)/Uncertainty, 2), axis=-1)

        Spectra_Loss_function=jax.jit(lambda model_spectra: Spectra_Loss(model_spectra, self.data_spectrum, Std_spectra))

        Inference_few_steps = Inference_class(self.GRF_class, self.Surface_brightness, Grad_descent_max_iter=2)
        GRF_Loss_function= jax.jit(lambda GRF_params: Inference_few_steps.GRF_Power_Spectrum_Loss(GRF_params,10,Spectra_Loss_function))

        Loss_in_true_point=GRF_Loss_function([-7.8,2.])
        Loss_in_false_point=GRF_Loss_function([-7.7,2.1])
        Grad_in_false_point=jax.grad(GRF_Loss_function)([-7.7,2.1])

        # Loss in the correct point is less that the loss in the wrong one
        self.assertLess(Loss_in_true_point,Loss_in_false_point)
        # Gradients in the wrong point lead to direction of the correct point
        self.assertGreater(Grad_in_false_point[0],0)
        self.assertLess(Grad_in_false_point[1], 0)


if __name__ == '__main__':
    unittest.main()
