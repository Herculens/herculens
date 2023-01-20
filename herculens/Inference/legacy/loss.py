# Defines the full loss function, from likelihood, prior and regularization terms
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal', 'austinpeel'


import numpy as np
import jax.numpy as jnp
from jax import jit

from herculens.Inference.legacy.base_differentiable import Differentiable
from herculens.Util.jax_util import WaveletTransform
from herculens.Util import model_util


__all__ = ['Loss']


class Loss(Differentiable):

    # TODO: creates subclasses Likelihood, Regularization and Prior to abstract out some of the methods here

    """
    Class that manages the (auto-differentiable) loss function, defined as:
    L = - log(likelihood) - log(prior) - log(regularization)

    Note that gradient, hessian, etc. are computed in the InferenceBase class.

    Supported options are:
    - likelihood_type [default: 'chi2']: single choice among
        'chi2', 'reduced_chi2', 'l2_norm'
    - regularization_terms [default: None]: a list containing choices among
        - for a 'PIXELATED' source or lens light: 'l1_starlet_source', 'l1_battle_source', 'positivity_source'
        - for a 'PIXELATED' lens potential: 'l1_starlet_potential', 'l1_battle_potential', 'positivity_potential'
    - prior_terms [default: None]: a list containing choices among
        'uniform', 'gaussian'
    """

    _supported_ll = ('chi2', 'reduced_chi2', 'l2_norm')
    _supported_regul_source = (
                               'l1_starlet_source', 
                               'l1_battle_source', 
                               'positivity_source'
                               )
    _supported_regul_lens_mass = (
                                  'l1_starlet_potential', 
                                  'l1_battle_potential', 
                                  'positivity_potential', 
                                  'negativity_potential', 
                                  'positivity_convergence',

                                  'analytical_potential',  # TEST: by default, regularize the penultimate (index -2) lens profile with the last one (index -1)
                                  )
    _supported_regul_lens_light = ('l1_starlet_lens_light', 'l1_battle_lens_light', 'positivity_lens_light')
    _supported_prior = ('uniform', 'gaussian')

    def __init__(self, data, image_class, param_class, 
                 likelihood_type='chi2', likelihood_mask=None, 
                 regularization_terms=None, regularization_strengths=None, 
                 regularization_weights=None, regularization_masks=None,
                 prior_terms=None, starlet_second_gen=False, index_analytical_potential=None):
        self._data  = data
        self._image = image_class
        self._param = param_class
        
        self._check_choices(likelihood_type, prior_terms, 
                            regularization_terms, regularization_strengths, 
                            regularization_weights, regularization_masks)
        self._init_likelihood(likelihood_type, likelihood_mask)
        self._init_regularizations(regularization_terms, 
                                   regularization_strengths, 
                                   regularization_weights, 
                                   regularization_masks, 
                                   starlet_second_gen, index_analytical_potential)
        self._init_priors(prior_terms)

    def _func(self, args):
        """negative log(likelihood*prior*regularization)"""
        kwargs = self._param.args2kwargs(args)
        model = self._image.model(**kwargs, k_lens=self._regul_k_lens)
        neg_log_ll  = - self.log_likelihood(model)
        neg_log_reg = - self.log_regularization(kwargs)
        neg_log_p   = - self.log_prior(args)
        neg_log = neg_log_ll + neg_log_reg + neg_log_p
        neg_log /= self._global_norm  # to keep loss magnitude in acceptable range
        return jnp.nan_to_num(neg_log, nan=1e15, posinf=1e15, neginf=1e15)

    @property
    def likelihood_num_data_points(self):
        return self._ll_num_data_points

    @property
    def likelihood_mask(self):
        return self._ll_mask

    @property
    def data(self):
        return self._data

    def _check_choices(self, likelihood_type, prior_terms, 
                       regularization_terms, regularization_strengths, 
                       regularization_weights, regularization_masks):
        if likelihood_type not in self._supported_ll:
            raise ValueError(f"Likelihood term '{likelihood_type}' is not supported")
        if prior_terms is not None:
            for term in prior_terms:
                if term not in self._supported_prior:
                    raise ValueError(f"Prior term '{term}' is not supported")
        if regularization_terms is not None:
            if len(regularization_terms) != len(regularization_strengths):
                raise ValueError(f"There should be at least one choice of "
                                 "regularization strength per regularization term.")
            if likelihood_type in ['chi2', 'reduced_chi2']:
                UserWarning(f"Likelihood type is '{likelihood_type}', which might "
                            "cause issues with some regularization choices")
            for term in regularization_terms:
                if term not in (self._supported_regul_source + 
                                self._supported_regul_lens_mass +
                                self._supported_regul_lens_light):
                    raise ValueError(f"Regularization term '{term}' is not supported")
                # TODO: if any regularization terms are not dependent on PIXELATED profiles
                # need to update these checks below
                if (term in self._supported_regul_source and 
                    'PIXELATED' not in self._image.SourceModel.profile_type_list):
                    raise ValueError(f"Regularization term '{term}' is only "
                                     "compatible with a 'PIXELATED' source light profile")
                
                if (term in self._supported_regul_lens_mass and 
                    'PIXELATED' not in self._image.MassModel.profile_type_list):
                    raise ValueError(f"Regularization term '{term}' is only "
                                     "compatible with a 'PIXELATED' lens profile")

                if (term in self._supported_regul_lens_light and 
                    'PIXELATED' not in self._image.LensLightModel.profile_type_list):
                    raise ValueError(f"Regularization term '{term}' is only "
                                     "compatible with a 'PIXELATED' lens profile")

    def _init_likelihood(self, likelihood_type, likelihood_mask):
        if likelihood_mask is None:
            self._ll_mask = np.ones_like(self._data)
        else:
            self._ll_mask = likelihood_mask.astype(float)
        self._ll_num_data_points = np.count_nonzero(self._ll_mask)
        if likelihood_type == 'chi2':
            self.log_likelihood = self.log_likelihood_chi2
            self._global_norm = 1.
        elif likelihood_type == 'reduced_chi2':
            self.log_likelihood = self.log_likelihood_chi2
            self._global_norm = 0.5 * self.likelihood_num_data_points
        elif likelihood_type == 'l2_norm':
            self.log_likelihood = self.log_likelihood_l2
            # here the global norm is such that l2_norm has same order of magnitude as a chi2
            self._global_norm = 1.0 # 0.5 * self._image.Grid.num_pixel * np.mean(self._image.Noise.C_D)

    def _init_regularizations(self, regularization_terms, regularization_strengths, 
                              regularization_weights, regularization_masks,
                              starlet_second_gen, index_analytical_potential):

        self._regul_k_lens = None  # TEMPORARY

        if regularization_terms is None:
            self.log_regularization = lambda kwargs: 0.  # no regularization
            return

        if regularization_masks is None:
            regularization_masks = [None]*len(regularization_terms)

        # TODO: implement regularization_weights for source regularization as well (for now it's only potential)
        i = 0
        regularization_weights_fix = []
        for term in regularization_terms:
            if 'potential' in term:
                regularization_weights_fix.append(regularization_weights[i])
                i += 1
            else:
                # TEMPORARY: just to populate weights for regularization terms other than potential
                # waiting for the source and lens light weights to be handled as well.
                regularization_weights_fix.append(None)

        self._idx_pix_src = self._image.SourceModel.pixelated_index
        self._idx_pix_pot = self._image.MassModel.pixelated_index
        self._idx_pix_ll  = self._image.LensLightModel.pixelated_index

        regul_func_list = []
        for term, strength, weights, mask in zip(regularization_terms, 
                                           regularization_strengths, 
                                           regularization_weights_fix,
                                           regularization_masks):
            # add the log-regularization function to the list
            regul_func_list.append(getattr(self, '_log_regul_'+term))

            if term == 'l1_starlet_source':
                n_pix_src = min(*self._image.SourceModel.pixelated_shape)
                n_scales = int(np.log2(n_pix_src))  # maximum allowed number of scales
                self._starlet_src = WaveletTransform(n_scales, wavelet_type='starlet',
                                                     second_gen=starlet_second_gen)
                wavelet_norms = self._starlet_src.scale_norms[:-1]  # ignore coarsest scale
                self._st_src_norms = jnp.expand_dims(wavelet_norms, (1, 2))
                if isinstance(strength, (int, float)):
                    self._st_src_lambda = self._st_src_lambda_hf = float(strength)
                elif isinstance(strength, (tuple, list)):
                    if len(strength) > 2:
                        raise ValueError("You can only specify two starlet regularization "
                                         "strength values at maximum")
                    self._st_src_lambda_hf = float(strength[0])
                    self._st_src_lambda = float(strength[1])

            elif term == 'l1_starlet_lens_light':
                n_pix_ll = min(*self._image.LensLightModel.pixelated_shape)
                n_scales = int(np.log2(n_pix_ll))  # maximum allowed number of scales
                self._starlet_ll = WaveletTransform(n_scales, wavelet_type='starlet',
                                                    second_gen=starlet_second_gen)
                wavelet_norms = self._starlet_ll.scale_norms[:-1]  # ignore coarsest scale
                self._st_ll_norms = jnp.expand_dims(wavelet_norms, (1, 2))
                if isinstance(strength, (int, float)):
                    self._st_ll_lambda = float(strength)
                    self._st_ll_lambda_hf = float(strength)
                elif isinstance(strength, (tuple, list)):
                    if len(strength) > 2:
                        raise ValueError("You can only specify two starlet regularization "
                                         "strength values at maximum")
                    self._st_ll_lambda_hf = float(strength[0])
                    self._st_ll_lambda = float(strength[1])

            elif term == 'l1_battle_source':
                n_scales = 1  # maximum allowed number of scales
                self._battle_src = WaveletTransform(n_scales, wavelet_type='battle-lemarie-3')
                self._bl_src_norm = self._battle_src.scale_norms[0]  # consider only first scale
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for Battle-Lemarie regularization")
                self._bl_src_lambda = float(strength)

            elif term == 'l1_battle_lens_light':
                n_scales = 1  # maximum allowed number of scales
                self._battle_ll = WaveletTransform(n_scales, wavelet_type='battle-lemarie-3')
                self._bl_ll_norm = self._battle_ll.scale_norms[0]  # consider only first scale
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for Battle-Lemarie regularization")
                self._bl_ll_lambda = float(strength)

            elif term == 'positivity_source':
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for positivity constraint")
                self._pos_src_lambda = float(strength)

            elif term == 'positivity_lens_light':
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for positivity constraint")
                self._pos_ll_lambda = float(strength)

            elif term == 'l1_starlet_potential':
                n_pix_pot = min(*self._image.MassModel.pixelated_shape)
                n_scales = int(np.log2(n_pix_pot))  # maximum allowed number of scales
                self._starlet_pot = WaveletTransform(n_scales, wavelet_type='starlet',
                                                     second_gen=starlet_second_gen)
                wavelet_norms = self._starlet_pot.scale_norms[:-1]  # ignore coarsest scale
                # self._st_pot_norms = jnp.expand_dims(wavelet_norms, (1, 2))
                if weights.shape[0] != n_scales+1:
                    raise ValueError(f"The weights do not contain enough wavelet scales"
                                     f" (should be {n_scales+1} inc. coarsest).")
                self._st_pot_weigths = weights
                if isinstance(strength, (int, float)):
                    self._st_pot_lambda = float(strength)
                    self._st_pot_lambda_hf = float(strength)
                elif isinstance(strength, (tuple, list)):
                    if len(strength) > 2:
                        raise ValueError("You can only specify two starlet regularization "
                                         "strength values at maximum")
                    self._st_pot_lambda_hf = float(strength[0])
                    self._st_pot_lambda = float(strength[1])

            elif term == 'l1_battle_potential':
                n_scales = 1  # maximum allowed number of scales
                self._battle_pot = WaveletTransform(n_scales, wavelet_type='battle-lemarie-3')
                # self._bl_pot_norm = self._battle_pot.scale_norms[0]  # consider only first scale
                if weights.shape[0] != n_scales+1:
                    raise ValueError(f"The weights do not contain enogh wavelet scales"
                                     f" (should be {n_scales+1} inc. coarsest).")
                self._bl_pot_weigths = weights
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for Battle-Lemarie regularization")
                self._bl_pot_lambda = float(strength)

            elif term == 'positivity_potential':
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for positivity constraint")
                self._pos_pot_lambda = float(strength)

            elif term == 'negativity_potential':
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for positivity constraint")
                self._neg_pot_lambda = float(strength)

            elif term == 'positivity_convergence':
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for positivity constraint")
                self._pos_conv_lambda = float(strength)
                self._x_lens, self._y_lens = self._image.Grid.model_pixel_coordinates('lens')


            elif term == 'analytical_potential':
                if index_analytical_potential is None:
                    raise ValueError("For analytical potential regularization, a `index_analytical_potential` is required.")
                self._idx_ana_pot = index_analytical_potential
                self._regul_k_lens = tuple([True if i != self._idx_ana_pot else False for i in range(len(self._image.MassModel.profile_type_list))])
                self._weigths = weights
                self._lambda  = float(strength)
                self._mask    = mask
                self._x_lens, self._y_lens = self._image.Grid.model_pixel_coordinates('lens')


        # build the composite function (sum of regularization terms)
        self.log_regularization = lambda kw: sum([func(kw) for func in regul_func_list])

    def _init_priors(self, prior_terms):
        if prior_terms is None:
            self.log_prior = lambda args: 0.
            return
        if prior_terms == ['uniform']:
            self.log_prior = self._param.log_prior_uniform
        elif prior_terms == ['gaussian']:
            self.log_prior = self._param.log_prior_gaussian
        elif 'gaussian' in prior_terms and 'uniform' in prior_terms:
            self.log_prior = self._param.log_prior

    # def log_likelihood_gaussian(self, model):
    #     C_D = self._image.Noise.C_D_model(model)
    #     det_C_D = jnp.prod(noise_var)  # knowing that C_D is diagonal
    #     #print("det_C_D", det_C_D)
    #     Z_D = np.sqrt( (2*np.pi)**self.likelihood_num_data_points * det_C_D )  # Eq. 24 from Vegetti & Koopmans 2009
    #     chi2 = - 0.5 * jnp.sum( (self._data - model)**2 * self.likelihood_mask / C_D )
    #     return jnp.log(Z_D) + chi2

    def log_likelihood_chi2(self, model):
        noise_var = self._image.Noise.C_D_model(model)
        # noise_var = self._image.Noise.C_D
        residuals = (self._data - model) * self.likelihood_mask
        return - 0.5 * jnp.sum(residuals**2 / noise_var)

    def log_likelihood_l2(self, model):
        # TODO: check that mask here does not mess up with the balance between l2-norm and wavelet regularization
        residuals = (self._data - model) * self.likelihood_mask
        return - 0.5 * jnp.sum(residuals**2)

    def _log_regul_l1_starlet_source(self, kwargs):
        model = self._image.model(**kwargs)
        noise_map = jnp.sqrt(self._image.Noise.C_D_model(model))  # TODO: do not take into account shot noise from lens light
        noise_level = jnp.mean(noise_map[self.likelihood_mask == 1])

        source_model = kwargs['kwargs_source'][self._idx_pix_src]['pixels']
        st = self._starlet_src.decompose(source_model)[:-1]  # ignore coarsest scale
        st_weighted_l1_hf = jnp.sum(self._st_src_norms[0] * noise_level * jnp.abs(st[0]))  # first scale (i.e. high frequencies)
        st_weighted_l1 = jnp.sum(self._st_src_norms[1:] * noise_level * jnp.abs(st[1:]))  # other scales
        return - (self._st_src_lambda_hf * st_weighted_l1_hf + self._st_src_lambda * st_weighted_l1)

    def _log_regul_l1_starlet_lens_light(self, kwargs):
        # TODO: generalise this for Poisson noise! but then the noise needs to be properly propagated to source plane
        noise_map = np.sqrt(self._image.Noise.C_D)

        # TEST reweight the noise map based on lensed source model
        #lensed_source_model = self._image.source_surface_brightness(kwargs['kwargs_source'], 
        #                                                            kwargs_lens=kwargs['kwargs_lens'],
        #                                                            de_lensed=False, unconvolved=True)
        #noise_level = noise_map # + lensed_source_model**3
        noise_level = np.mean(noise_map[self.likelihood_mask == 1])
        # end TEST

        model = kwargs['kwargs_lens_light'][self._idx_pix_ll]['pixels']
        st = self._starlet_ll.decompose(model)[:-1]  # ignore coarsest scale
        st_weighted_l1_hf = jnp.sum(self._st_ll_norms[0] * noise_level * jnp.abs(st[0]))  # first scale (i.e. high frequencies)
        st_weighted_l1 = jnp.sum(self._st_ll_norms[1:] * noise_level * jnp.abs(st[1:]))  # other scales
        return - (self._st_ll_lambda_hf * st_weighted_l1_hf + self._st_ll_lambda * st_weighted_l1)

    def _log_regul_l1_starlet_potential(self, kwargs):
        weights = self._st_pot_weigths
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['pixels']
        st = self._starlet_pot.decompose(psi_model)
        st_weighted_l1_hf = jnp.sum(jnp.abs(weights[0] * st[0]))  # first scale (i.e. high frequencies)
        st_weighted_l1 = jnp.sum(jnp.abs(weights[1:-1] * st[1:-1]))  # other scales (except coarsest)
        return - (self._st_pot_lambda_hf * st_weighted_l1_hf + self._st_pot_lambda * st_weighted_l1)

    def _log_regul_l1_battle_source(self, kwargs):
        model = self._image.model(**kwargs)
        noise_map = jnp.sqrt(self._image.Noise.C_D_model(model))  # TODO: do not take into account shot noise from lens light
        noise_level = jnp.mean(noise_map[self.likelihood_mask == 1])
        source_model = kwargs['kwargs_source'][self._idx_pix_src]['pixels']
        bl = self._battle_src.decompose(source_model)[0]  # consider only first scale
        bl_weighted_l1 = jnp.sum(self._bl_src_norm * noise_level * jnp.abs(bl))
        return - self._bl_src_lambda * bl_weighted_l1

    def _log_regul_l1_battle_lens_light(self, kwargs):
        # TODO: generalise this for Poisson noise! but then the noise needs to be properly propagated to source plane
        noise_map = np.sqrt(self._image.Noise.C_D)
        noise_level = np.mean(noise_map[self.likelihood_mask == 1])
        #noise_level = noise_map
        model = kwargs['kwargs_lens_light'][self._idx_pix_ll]['pixels']
        bl = self._battle_ll.decompose(model)[0]  # consider only first scale
        bl_weighted_l1 = jnp.sum(self._bl_ll_norm * noise_level * jnp.abs(bl))
        return - self._bl_ll_lambda * bl_weighted_l1

    def _log_regul_l1_battle_potential(self, kwargs):
        weights = self._bl_pot_weigths
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['pixels']
        bl = self._battle_pot.decompose(psi_model)
        bl_weighted_l1 = jnp.sum(jnp.abs(weights[0] * bl[0]))  # only first BL scale
        return - self._bl_pot_lambda * bl_weighted_l1

    def _log_regul_positivity_source(self, kwargs):
        source_model = kwargs['kwargs_source'][self._idx_pix_src]['pixels']
        return - self._pos_src_lambda * jnp.abs(jnp.sum(jnp.minimum(0., source_model)))

    def _log_regul_positivity_lens_light(self, kwargs):
        model = kwargs['kwargs_lens_light'][self._idx_pix_ll]['pixels']
        return - self._pos_ll_lambda * jnp.abs(jnp.sum(jnp.minimum(0., model)))

    def _log_regul_positivity_potential(self, kwargs):
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['pixels']
        return - self._pos_pot_lambda * jnp.abs(jnp.sum(jnp.minimum(0., psi_model)))

    def _log_regul_negativity_potential(self, kwargs):
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['pixels']
        return - self._neg_pot_lambda * jnp.abs(jnp.sum(jnp.maximum(0., psi_model)))

    def _log_regul_positivity_convergence(self, kwargs):
        kappa_model = self._image.MassModel.kappa(self._x_lens, 
                                                  self._y_lens,
                                                  kwargs['kwargs_lens'], 
                                                  k=self._idx_pix_pot)
        return - self._pos_conv_lambda * jnp.abs(jnp.sum(jnp.minimum(0., kappa_model)))

    def _log_regul_analytical_potential(self, kwargs):
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['pixels']
        target_model = self._image.MassModel.potential(self._x_lens, self._y_lens, 
                                                       kwargs['kwargs_lens'],
                                                       k=self._idx_ana_pot)
        return - self._lambda * jnp.sum(self._mask * self._weigths * (psi_model - target_model)**2)
        # or similar to Tagore & Keeton 2014 (does not seem to work tho)
        #return - self._lambda * (jnp.sum(self._mask * self._weigths * psi_model / target_model) - jnp.sum(self._mask) * jnp.mean(psi_model / target_model))
