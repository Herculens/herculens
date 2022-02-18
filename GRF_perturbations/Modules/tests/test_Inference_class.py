import unittest

from GRF_perturbations.Modules.Inference_class import *
import scipy
import os
import math

#CPUs parallelization
import multiprocessing
max_thread_numbers=multiprocessing.cpu_count()
#os.system('echo CPU count %d'%)

#Parallelize on several CPUs
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%d"%(max_thread_numbers)

class test_Inference(unittest.TestCase):
    """ Inference class handles inference of unperturbed source-lens model from gravitational lens surface brightness,
     as well as GRF power spectrum from a given radial power spectrum of surface brightness anomalies on the Einstein ring"""

    @classmethod
    def setUpClass(self):
        """ Predefine some quantities to be used throughout testing"""

        # Classes handling GRF generation, Lens SB generation and Inference of GRF and SB
        self.GRF_class = GRF_inhomogeneities_class(100, 0.08, 100)
        self.Surface_brightness = Surface_brightness_class(100, 0.08, 0.1, 200, 2028)
        self.Inference= Inference_class(self.GRF_class,self.Surface_brightness,Grad_descent_max_iter=400)

        # SB anomalies radial power spectrum. Will be used to test inference of GRF
        self.data_spectrum=self.Inference.Anomalies_Radial_Power_Spectrum([-7.8, 2],self.GRF_class.tensor_unit_Fourier_images[0])

        # Functions for image generation
        simulate_unperturbed_image = self.Surface_brightness.unperturbed_image_getter
        simulate_unperturbed_image_noiseless = lambda model_kwargs: simulate_unperturbed_image(model_kwargs,Noise_flag=False)
        simulate_perturbed_image = self.Surface_brightness.perturbed_image_getter

        # Images for SB fitting to be tested on
        self.Image_unperturbed_noiseless=simulate_unperturbed_image_noiseless(self.Surface_brightness.kwargs_unperturbed_model)
        self.Image_unperturbed_noisy=simulate_unperturbed_image(self.Surface_brightness.kwargs_unperturbed_model,Noise_flag=True,noise_seed=18)
        GRF_potential = self.GRF_class.potential([-7.8, 2], self.GRF_class.tensor_unit_Fourier_images[0])
        self.Image_perturbed_noisy=simulate_perturbed_image(GRF_potential,Noise_flag=True,noise_seed=42)

        # We need a sample of 32 surface brightness anomalies power spectra
        # to estimate GRF random seed-based and noise-based uncertainty
        compute_spectrum_pure = lambda unit_Fourier_image: self.Inference.Anomalies_Radial_Power_Spectrum(jnp.array([-7.8, 2.]), unit_Fourier_image)
        self.Perturbations_spectra=np.zeros((32,len(self.Surface_brightness.frequencies)))
        # Run spectra simulation in parallel
        if (max_thread_numbers>=32):
            self.Perturbations_spectra=jax.pmap(compute_spectrum_pure)(self.GRF_class.tensor_unit_Fourier_images[:32])
        else:
            iterations_num=math.ceil(32/max_thread_numbers)
            for i in range(iterations_num):
                start_index=i*max_thread_numbers
                finish_index=np.minimum(32,(i+1)*max_thread_numbers)
                self.Perturbations_spectra[start_index:finish_index]=jax.pmap(compute_spectrum_pure)(self.GRF_class.tensor_unit_Fourier_images[start_index:finish_index])

    def test_Loss_unperturbed_model(self):
        """Loss_unperturbed_model(args,data) is a chi^2 negative log-likelihood for a hypothesis that
         the SB image 'data' was generated from unperturbed Source-Lens model with parameters 'args'.
         SB image 'data' can be Source-Lens,Source-Lens + noise,Source-Lens + noise+perturbations,
         The Loss for fitted 'args' should increase consequently."""


        # True arguments used to generate unperturbed Source-Lens model
        args_data=self.Inference.SL_parameters.kwargs2args(self.Surface_brightness.kwargs_unperturbed_model)

        Loss_perturbed_noisy = self.Inference.Loss_unperturbed_model(args_data,self.Image_perturbed_noisy)
        Loss_unperturbed_noisy = self.Inference.Loss_unperturbed_model(args_data,self.Image_unperturbed_noisy)
        Loss_unperturbed_noiseless = self.Inference.Loss_unperturbed_model(args_data, self.Image_unperturbed_noiseless)

        # Perfect fit if fit model and data model are the same
        self.assertAlmostEqual(Loss_unperturbed_noiseless, 0)
        # Adding noise decreases the likelihood in the fit point
        self.assertGreater(Loss_unperturbed_noisy, Loss_unperturbed_noiseless)
        # Adding perturbation to noisy data decreases likelihood even more
        self.assertGreater(Loss_perturbed_noisy,Loss_unperturbed_noisy)



    def test_differentiable_fit_Surface_Brightness(self):
        """args=differentiable_fit_Surface_Brightness(data) should fit SB image 'data' with unperturbed Source-Lens model
        using simplistic gradient descent. However the function should be differentiable in a sense d(args)/d(data).
        So we are able to know how the best fit model 'args' would change if we change the fitted SB 'data'."""

        # 'data' is Source+Lens+perturbations+noise. These are true args of Source+Lens
        args_data = self.Inference.SL_parameters.kwargs2args(self.Surface_brightness.kwargs_unperturbed_model)
        # Loss in the true 'args'. Not the best fit, cause it doesn't account for perturbations+noise
        Loss_initial=self.Inference.Loss_unperturbed_model(args_data, self.Image_perturbed_noisy)

        # Fit of 'data' which is Source+Lens+perturbations+noise using Source+Lens model
        args_max_likelihood=self.Inference.differentiable_fit_Surface_Brightness(self.Image_perturbed_noisy)
        # Loss in the fit 'args'. Should be better than initial
        Loss_final=self.Inference.Loss_unperturbed_model(args_max_likelihood, self.Image_perturbed_noisy)

        # Check that fitting improves model quality
        self.assertLess(Loss_final,Loss_initial)

        # Check that we can take gradients wrp fitted data d(args)/d(data) using simplified inference
        Inference_few_steps = Inference_class(self.GRF_class, self.Surface_brightness, Grad_descent_max_iter=2)
        best_fit_Loss= jax.jit(lambda data:  self.Inference.Loss_unperturbed_model(
                            Inference_few_steps.differentiable_fit_Surface_Brightness(data),data))
        Grads_of_best_fit_Loss=jax.grad(best_fit_Loss)(self.Image_perturbed_noisy)
        # Check that gradients are propagated and they are not zeros
        self.assertTrue((np.abs(Grads_of_best_fit_Loss)>0).all())

    def test_compute_radial_spectrum(self):
        """compute_radial_spectrum(SB_anomalies) should mask the image 'SB_anomalies' and compute their Radial power spectrum.
        Noise is not spatially correlated, so it has flat spectrum  (Beta=0), actual SB anomalies have non flat one (Beta>0),
        so we test that compute_radial_spectrum(SB_anomalies) will result in the same."""

        # Image of noise only
        PS_Noise=self.Inference.compute_radial_spectrum(self.Image_unperturbed_noisy-self.Image_unperturbed_noiseless)

        # Image of perturbations+noise
        PS_Perturbations =self.Inference.compute_radial_spectrum(self.Image_perturbed_noisy - self.Image_unperturbed_noisy)

        power_law_function = lambda k, logA, Beta: np.power(10, logA) * np.power(k, -Beta)

        # Infer power slope of noise
        fit_results, _ = scipy.optimize.curve_fit(power_law_function, self.Surface_brightness.frequencies,PS_Noise)
        _,Beta_noise = fit_results

        # Infer power slope of the perturbation anomalies+noise
        fit_results, _ = scipy.optimize.curve_fit(power_law_function, self.Surface_brightness.frequencies, PS_Perturbations)
        _, Beta_Perturbations = fit_results

        # Perturbations+Noise should have more power than just noise
        self.assertGreater(PS_Perturbations.sum(),PS_Noise.sum())
        # Noise should have flat spectrum
        self.assertLess(Beta_noise, 0.1)
        # Spectrum of perturbations should not be flat
        self.assertGreater(Beta_Perturbations, 0.1)

    def test_Anomalies_Radial_Power_Spectrum(self):
        """Anomalies_Radial_Power_Spectrum(GRF_params,unit_Fourier_image,Noise_flag) generates GRF potential for GRF_params=[logA,Beta] and
        unit_Fourier_image (random realisation of the GRF). It simulates mock perturbed gravitational lens (with/without observation noise for Noise_flag).
        Then it fits the mock with unperturbed Source-Lens model, extracts SB anomalies as fit residuals and returns the Radial power spectrum of those.
        We need to test that the noise introduce flat spectrum anomalies, whereas anomalies for perturbations have non-zero power slope.
        Also we need to test that the function is differentiable and has sensible gradients."""

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
        # Power of anomalies increases with increasing amplitude of potential perturbations
        self.assertGreater(Power_gradient[0],0)
        # Increasing power slope of potential perturbations decreases total power of anomalies
        self.assertLess(Power_gradient[1], 0)


    def test_GRF_Power_Spectrum_Loss(self):
        """GRF_Power_Spectrum_Loss(GRF_params,GRF_seeds_number,Spectra_Loss_function,Noise_flag) computes GRF_seeds_number of SB anomalies radial power spectra 'model_spectra'
        for GRF_params-defined potential perturbations, using Anomalies_Radial_Power_Spectrum(GRF_params,unit_Fourier_image,Noise_flag).
        Then it estimates negative log-likelihood of 'data_spectrum' being generated with 'GRF_params' using Spectra_Loss_function(model_spectra) precomputed for a 'data_spectrum'.
        The function should give lower Loss in the wrong GRF_params point than in the true one. Also we should be differentiable in the sense d(Loss)/d(GRF_params)
        and gradient should point to the correct direction."""

        # Estimate Anomelies spectrum uncertainties for logNormal-based Chi^2
        Mean_spectra = np.log(self.Perturbations_spectra).mean(axis=0)
        Std_spectra = np.sqrt(np.power(np.log(self.Perturbations_spectra) - Mean_spectra, 2).sum(axis=0) / (len(self.Perturbations_spectra) - 1))

        # Chi^2 for logNormal likelihood of spectrum with predefine uncertainties
        def Spectra_Loss(model_spectra, data_spectrum, Uncertainty):
            data_log_spectrum = jnp.log(data_spectrum)
            models_log_spectra = jnp.log(model_spectra)

            Mean_logN = models_log_spectra.mean(axis=-2)
            # Chi^2 loss for Normal likelihood of log(Power_spectrum)
            return jnp.mean(jnp.power((data_log_spectrum - Mean_logN)/Uncertainty, 2), axis=-1)

        # Pure loss function precompiled for 'data' and 'uncertainties'
        Spectra_Loss_function=jax.jit(lambda model_spectra: Spectra_Loss(model_spectra, self.data_spectrum, Std_spectra))

        # Simplified inference procedure
        Inference_few_steps = Inference_class(self.GRF_class, self.Surface_brightness, Grad_descent_max_iter=2)
        # Pure precompiled function Loss(GRF_params)
        GRF_Loss_function= jax.jit(lambda GRF_params: Inference_few_steps.GRF_Power_Spectrum_Loss(GRF_params,10,Spectra_Loss_function))

        Loss_in_true_point=GRF_Loss_function([-7.8,2.])
        Loss_in_false_point=GRF_Loss_function([-7.7,2.1])
        Grad_in_false_point=jax.grad(GRF_Loss_function)([-7.7,2.1])

        # Loss in the correct point is less that loss in the wrong one
        self.assertLess(Loss_in_true_point,Loss_in_false_point)
        # Gradients in the wrong point lead to direction of the correct point
        self.assertGreater(Grad_in_false_point[0],0)
        self.assertLess(Grad_in_false_point[1], 0)


if __name__ == '__main__':
    unittest.main()
