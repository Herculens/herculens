import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import jit

from herculens.Util.jax_util import WaveletTransform
from herculens.Util import model_util


__all__ = ['Loss']


class Loss(object):

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
    _supported_regul_source = ('l1_starlet_source', 'l1_battle_source', 'positivity_source')
    _supported_regul_lens_mass = ('l1_starlet_potential', 'l1_battle_potential', 'positivity_potential', 'positivity_convergence')
    _supported_regul_lens_light = ('l1_starlet_lens_light', 'l1_battle_lens_light', 'positivity_lens_light')
    _supported_prior = ('uniform', 'gaussian')

    def __init__(self, data, image_class, param_class, 
                 likelihood_type='chi2', likelihood_mask=None, mask_from_source_plane=False,
                 regularization_terms=None, regularization_strengths=None,
                 potential_noise_map=None, prior_terms=None):
        self._data  = data
        self._image = image_class
        self._param = param_class
        
        self._check_choices(likelihood_type, prior_terms, regularization_terms, regularization_strengths)
        self._init_likelihood(likelihood_type, likelihood_mask, mask_from_source_plane)
        self._init_regularizations(regularization_terms, regularization_strengths, potential_noise_map)
        self._init_priors(prior_terms)

    @partial(jit, static_argnums=(0,))
    def __call__(self, args):
        return self.loss(args)

    def loss(self, args):
        """defined as the negative log(likelihood*prior*regularization)"""
        kwargs = self._param.args2kwargs(args)
        model = self._image.model(**kwargs)
        neg_log = - self.log_likelihood(model) - self.log_regularization(kwargs) - self.log_prior(args)
        neg_log /= self._global_norm  # to keep loss magnitude in acceptable range
        return neg_log

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
                       regularization_terms, regularization_strengths):
        if likelihood_type not in self._supported_ll:
            raise ValueError(f"Likelihood term '{likelihood_type}' is not supported")
        if prior_terms is not None:
            for term in prior_terms:
                if term not in self._supported_prior:
                    raise ValueError(f"Prior term '{term}' is not supported")
        if regularization_terms is not None:
            if regularization_strengths is None:
                # default regularization strength is 3, typically suitable for sparsity+wavelets priors 
                regularization_strengths = [3.]*len(regularization_terms)
            elif len(regularization_terms) != len(regularization_strengths):
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
                    'PIXELATED' not in self._image.LensModel.lens_model_list):
                    raise ValueError(f"Regularization term '{term}' is only "
                                     "compatible with a 'PIXELATED' lens profile")

                if (term in self._supported_regul_lens_light and 
                    'PIXELATED' not in self._image.LensLightModel.profile_type_list):
                    raise ValueError(f"Regularization term '{term}' is only "
                                     "compatible with a 'PIXELATED' lens profile")

    def _init_likelihood(self, likelihood_type, likelihood_mask, mask_from_source_plane):
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
        if mask_from_source_plane is True and self._image.SourceModel.has_pixels:
            self._ll_mask = model_util.mask_from_pixelated_source(self._image, self._param)
        elif likelihood_mask is None:
            self._ll_mask = np.ones_like(self._data)
        else:
            self._ll_mask = likelihood_mask.astype(float)
        self._ll_num_data_points = np.count_nonzero(self._ll_mask)

    def _init_regularizations(self, regularization_terms, regularization_strengths, 
                              potential_noise_map):
        if regularization_terms is None:
            self.log_regularization = lambda kwargs: 0.  # no regularization
            return

        self._idx_pix_src = self._image.SourceModel.pixelated_index
        self._idx_pix_pot = self._image.LensModel.pixelated_index
        self._idx_pix_ll  = self._image.LensLightModel.pixelated_index
        self._noise_map_pot = potential_noise_map

        regul_func_list = []
        for term, strength in zip(regularization_terms, regularization_strengths):
            # add the log-regularization function to the list
            regul_func_list.append(getattr(self, '_log_regul_'+term))

            if term == 'l1_starlet_source':
                n_pix_src = min(*self._image.SourceModel.pixelated_shape)
                n_scales = int(np.log2(n_pix_src))  # maximum allowed number of scales
                self._starlet_src = WaveletTransform(n_scales, wavelet_type='starlet')
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

            if term == 'l1_starlet_lens_light':
                n_pix_ll = min(*self._image.LensLightModel.pixelated_shape)
                n_scales = int(np.log2(n_pix_ll))  # maximum allowed number of scales
                self._starlet_ll = WaveletTransform(n_scales, wavelet_type='starlet')
                wavelet_norms = self._starlet_ll.scale_norms[:-1]  # ignore coarsest scale
                self._st_ll_norms = jnp.expand_dims(wavelet_norms, (1, 2))
                if isinstance(strength, (int, float)):
                    self._st_ll_lambda = self._st_ll_lambda_hf = float(strength)
                elif isinstance(strength, (tuple, list)):
                    if len(strength) > 2:
                        raise ValueError("You can only specify two starlet regularization "
                                         "strength values at maximum")
                    self._st_ll_lambda_hf = float(strength[0])
                    self._st_ll_lambda = float(strength[1])

            elif term == 'l1_battle_source':
                n_scales = 1  # maximum allowed number of scales
                self._battle_src = WaveletTransform(n_scales, wavelet_type='battle-lemarie-3')
                self._bt_src_norm = self._battle_src.scale_norms[0]  # consider only first scale
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for Battle-Lemarie regularization")
                self._bt_src_lambda = float(strength)

            elif term == 'l1_battle_lens_light':
                n_scales = 1  # maximum allowed number of scales
                self._battle_ll = WaveletTransform(n_scales, wavelet_type='battle-lemarie-3')
                self._bt_ll_norm = self._battle_ll.scale_norms[0]  # consider only first scale
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for Battle-Lemarie regularization")
                self._bt_ll_lambda = float(strength)

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
                n_pix_pot = min(*self._image.LensModel.pixelated_shape)
                n_scales = int(np.log2(n_pix_pot))  # maximum allowed number of scales
                self._starlet_pot = WaveletTransform(n_scales, wavelet_type='starlet')
                wavelet_norms = self._starlet_pot.scale_norms[:-1]  # ignore coarsest scale
                self._st_pot_norms = jnp.expand_dims(wavelet_norms, (1, 2))
                if isinstance(strength, (int, float)):
                    self._st_pot_lambda = self._st_pot_lambda_hf = float(strength)
                elif isinstance(strength, (tuple, list)):
                    if len(strength) > 2:
                        raise ValueError("You can only specify two starlet regularization "
                                         "strength values at maximum")
                    self._st_pot_lambda_hf = float(strength[0])
                    self._st_pot_lambda = float(strength[1])

            elif term == 'l1_battle_potential':
                n_scales = 1  # maximum allowed number of scales
                self._battle_pot = WaveletTransform(n_scales, wavelet_type='battle-lemarie-3')
                self._bt_pot_norm = self._battle_pot.scale_norms[0]  # consider only first scale
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for Battle-Lemarie regularization")
                self._bt_pot_lambda = float(strength)

            elif term == 'positivity_potential':
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for positivity constraint")
                self._pos_pot_lambda = float(strength)

            elif term == 'positivity_convergence':
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for positivity constraint")
                self._pos_conv_lambda = float(strength)
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
        # TODO: generalise this for Poisson noise! but then the noise needs to be properly propagated to source plane
        noise_map = np.sqrt(self._image.Noise.C_D)
        noise_level = np.mean(noise_map[self.likelihood_mask == 1])
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
        noise_map = self._noise_map_pot
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['pixels']
        st = self._starlet_pot.decompose(psi_model)[:-1]  # ignore coarsest scale
        st_weighted_l1_hf = jnp.sum(self._st_pot_norms[0] * noise_map * jnp.abs(st[0]))  # first scale (i.e. high frequencies)
        st_weighted_l1 = jnp.sum(self._st_pot_norms[1:] * noise_map * jnp.abs(st[1:]))  # other scales
        return - (self._st_pot_lambda_hf * st_weighted_l1_hf + self._st_pot_lambda * st_weighted_l1)

    def _log_regul_l1_battle_source(self, kwargs):
        # TODO: generalise this for Poisson noise! but then the noise needs to be properly propagated to source plane
        noise_map = np.sqrt(self._image.Noise.C_D)
        noise_level = np.mean(noise_map[self.likelihood_mask == 1])
        source_model = kwargs['kwargs_source'][self._idx_pix_src]['pixels']
        bt = self._battle_src.decompose(source_model)[0]  # consider only first scale
        bt_weighted_l1 = jnp.sum(self._bt_src_norm * noise_level * jnp.abs(bt))
        return - self._bt_src_lambda * bt_weighted_l1

    def _log_regul_l1_battle_lens_light(self, kwargs):
        # TODO: generalise this for Poisson noise! but then the noise needs to be properly propagated to source plane
        noise_map = np.sqrt(self._image.Noise.C_D)
        noise_level = np.mean(noise_map[self.likelihood_mask == 1])
        #noise_level = noise_map
        model = kwargs['kwargs_lens_light'][self._idx_pix_ll]['pixels']
        bt = self._battle_ll.decompose(model)[0]  # consider only first scale
        bt_weighted_l1 = jnp.sum(self._bt_ll_norm * noise_level * jnp.abs(bt))
        return - self._bt_ll_lambda * bt_weighted_l1

    def _log_regul_l1_battle_potential(self, kwargs):
        noise_map = self._noise_map_pot
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['pixels']
        bt = self._battle_pot.decompose(psi_model)[0]  # consider only first scale
        bt_weighted_l1 = jnp.sum(self._bt_pot_norm * noise_map * jnp.abs(bt))
        return - self._bt_pot_lambda * bt_weighted_l1

    def _log_regul_positivity_source(self, kwargs):
        source_model = kwargs['kwargs_source'][self._idx_pix_src]['pixels']
        return - self._pos_src_lambda * jnp.abs(jnp.sum(jnp.minimum(0., source_model)))

    def _log_regul_positivity_lens_light(self, kwargs):
        model = kwargs['kwargs_lens_light'][self._idx_pix_ll]['pixels']
        return - self._pos_ll_lambda * jnp.abs(jnp.sum(jnp.minimum(0., model)))

    def _log_regul_positivity_potential(self, kwargs):
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['pixels']
        return - self._pos_pot_lambda * jnp.abs(jnp.sum(jnp.minimum(0., psi_model)))

    def _log_regul_positivity_convergence(self, kwargs):
        kappa_model = self._image.LensModel.kappa(self._x_lens, 
                                                  self._y_lens,
                                                  kwargs['kwargs_lens'], 
                                                  k=self._idx_pix_pot)
        return - self._pos_conv_lambda * jnp.abs(jnp.sum(jnp.minimum(0., kappa_model)))
