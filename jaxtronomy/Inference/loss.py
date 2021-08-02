import warnings
import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import jit
from jaxtronomy.Util.jax_util import WaveletTransform


class Loss(object):
    """
    Class that manages the (auto-differentiable) loss function, defined as:
    L = - log(likelihood) - log(prior) - log(regularization)

    Note that gradient, hessian, etc. are computed in the InferenceBase class.

    Supported options are:
    - likelihood_type [default: 'chi2']: single choice among
        'chi2', 'l2_norm'
    - regularization_terms [default: None]: a list containing choices among
        - for a 'PIXELATED' source: 'l1_starlet_source', 'l1_battle_source'
        - for a 'PIXELATED' lens potential: 'l1_starlet_potential', 'l1_battle_potential'
    - prior_terms [default: None]: a list containing choices among
        'uniform', 'gaussian'
    """

    _supported_ll = ('chi2', 'l2_norm')
    _supported_regul_source = ('l1_starlet_source', 'l1_battle_source', 'positivity_source')
    _supported_regul_lens = ('l1_starlet_potential', 'l1_battle_potential')
    _supported_prior = ('uniform', 'gaussian')

    def __init__(self, data, image_class, param_class, 
                 likelihood_type='chi2', prior_terms=None,
                 regularization_terms=None, regularization_strengths=None,
                 potential_noise_map=None):
        self._data  = data
        self._image = image_class
        self._param = param_class
        
        self._check_choices(likelihood_type, prior_terms, regularization_terms, regularization_strengths)
        self._initialize_likelihood(likelihood_type)
        self._initialize_regularizations(regularization_terms, regularization_strengths,
                                         potential_noise_map)
        self._initialize_priors(prior_terms)

    @partial(jit, static_argnums=(0,))
    def __call__(self, args):
        return self._loss(args)

    def _loss(self, args):
        """defined as the negative log(likelihood*prior*regularization)"""
        kwargs = self._param.args2kwargs(args)
        model = self._image.model(**kwargs)
        neg_log = - self._log_likelihood(model) - self._log_regul(kwargs) - self._log_prior(args)
        neg_log /= self._global_norm  # to keep loss magnitude in acceptable range
        return neg_log

    def _check_choices(self, likelihood_type, prior_terms, regularization_terms, regularization_strengths):
        if likelihood_type not in self._supported_ll:
            raise ValueError(f"Likelihood term '{likelihood_type}' is not supported")
        if prior_terms is not None:
            for term in prior_terms:
                if term not in self._supported_prior:
                    raise ValueError(f"Prior term '{term}' is not supported")
        if regularization_terms is not None:
            if likelihood_type == 'chi2':
                warnings.warn(f"WARNING: likelihood type is '{likelihood_type}', which might "
                              "cause issues with some regularization choices")
            for term in regularization_terms:
                if term not in self._supported_regul_source + self._supported_regul_lens:
                    raise ValueError(f"Regularization term '{term}' is not supported")
                # TODO: if any regularization terms are not dependent on PIXELATED profiles, update this check
                source_profile_list = self._image.SourceModel.profile_type_list
                lens_profile_list = self._image.LensModel.lens_model_list
                if (term in self._supported_regul_source 
                    and 'PIXELATED' not in source_profile_list
                    and 'PIXELATED_BICUBIC' not in source_profile_list):
                    raise ValueError(f"Regularization term '{term}' is only "
                                     "compatible with a 'PIXELATED' source light profile")
                
                elif (term in self._supported_regul_lens
                      and 'PIXELATED' not in lens_profile_list):
                    raise ValueError(f"Regularization term '{term}' is only "
                                     "compatible with a 'PIXELATED' lens profile")
            if len(regularization_terms) != len(regularization_strengths):
                raise ValueError(f"There should be one choice of regularization strength per regularization term")

    def _initialize_likelihood(self, likelihood_type):
        if likelihood_type == 'chi2':
            self._log_likelihood = self._log_likelihood_chi2
            self._global_norm = 1.
        elif likelihood_type == 'l2_norm':
            self._log_likelihood = self._log_likelihood_l2
            self._global_norm = 0.5 * self._image.Grid.num_pixel * np.mean(self._image.Noise.C_D)

    def _initialize_regularizations(self, regularization_terms, regularization_strengths, 
                                    potential_noise_map):
        if regularization_terms is None:
            self._log_regul = lambda kwargs: 0.  # no regularization
            return

        # pre-compute some quantities required for the chosen regularizations
        # TODO: generalise this for Poisson noise!
        data_noise_map = self._image.Noise.background_rms

        self._idx_pix_src = self._param.pixelated_source_index
        self._idx_pix_pot = self._param.pixelated_potential_index

        regul_func_list = []
        for term, strength in zip(regularization_terms, regularization_strengths):
            # add the log-regularization function to the list
            regul_func_list.append(getattr(self, '_log_regul_'+term))

            if term == 'l1_starlet_source':
                n_pix_src = min(*self._param.pixelated_source_shape)
                n_scales = int(np.log2(n_pix_src))  # maximum allowed number of scales
                self._starlet_src = WaveletTransform(n_scales, wavelet_type='starlet')
                wavelet_norms = self._starlet_src.scale_norms[:-1]  # ignore coarsest scale
                self._st_src_weights = data_noise_map * jnp.expand_dims(wavelet_norms, (1, 2))   # <<-- not full noise sigma !
                if isinstance(strength, (int, float)):
                    self._st_src_lambda = self._st_src_lambda_hf = float(strength)
                elif isinstance(strength, (tuple, list)):
                    if len(strength) > 2:
                        raise ValueError("You can only specify two starlet regularization "
                                         "strength values at maximum")
                    self._st_src_lambda_hf = float(strength[0])
                    self._st_src_lambda = float(strength[1])

            elif term == 'l1_battle_source':
                n_scales = 1  # maximum allowed number of scales
                self._battle_src = WaveletTransform(n_scales, wavelet_type='battle-lemarie-3')
                wavelet_norm = self._battle_src.scale_norms[0]  # consider only first scale
                self._bt_src_weights = data_noise_map * wavelet_norm   # <<-- not full noise sigma !
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for Battle-Lemarie regularization")
                self._bt_src_lambda = float(strength)

            elif term == 'positivity_source':
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for positivity constraint")
                self._pos_src_lambda = float(strength)

            elif term == 'l1_starlet_potential':
                n_pix_pot = min(*self._param.pixelated_potential_shape)
                n_scales = int(np.log2(n_pix_pot))  # maximum allowed number of scales
                self._starlet_pot = WaveletTransform(n_scales, wavelet_type='starlet')
                wavelet_norms = self._starlet_pot.scale_norms[:-1]  # ignore coarsest scale
                self._st_pot_weights = potential_noise_map * jnp.expand_dims(wavelet_norms, (1, 2))   # <<-- not full noise sigma !
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
                wavelet_norm = self._battle_pot.scale_norms[0]  # consider only first scale
                self._bt_pot_weights = potential_noise_map * wavelet_norm   # <<-- not full noise sigma !
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for Battle-Lemarie regularization")
                self._bt_pot_lambda = float(strength)

            elif term == 'positivity_potential':
                if isinstance(strength, (tuple, list)):
                    raise ValueError("You can only specify one regularization "
                                     "strength for positivity constraint")
                self._pos_pot_lambda = float(strength)

        # build the composite function (sum of regularization terms)
        self._log_regul = lambda kw: sum([func(kw) for func in regul_func_list])

    def _initialize_priors(self, prior_terms):
        if prior_terms is None:
            self._log_prior = lambda args: 0.
            return
        if prior_terms == ['uniform']:
            self._log_prior = self._param.log_prior_uniform
        elif prior_terms == ['gaussian']:
            self._log_prior = self._param.log_prior_gaussian
        elif 'gaussian' in prior_terms and 'uniform' in prior_terms:
            self._log_prior = self._param.log_prior

    def _log_likelihood_chi2(self, model):
        #noise_var = self._image.Noise.C_D_model(model)
        noise_var = self._image.Noise.C_D
        return - jnp.mean((self._data - model)**2 / noise_var)

    def _log_likelihood_l2(self, model):
        return - 0.5 * jnp.sum((self._data - model)**2)

    def _log_regul_l1_starlet_source(self, kwargs):
        source_model = kwargs['kwargs_source'][self._idx_pix_src]['image']
        st = self._starlet_src.decompose(source_model)[:-1]  # ignore coarsest scale
        st_weighted_l1_hf = jnp.sum(self._st_src_weights[0] * jnp.abs(st[0]))  # first scale (i.e. high frequencies)
        st_weighted_l1 = jnp.sum(self._st_src_weights[1:] * jnp.abs(st[1:]))  # other scales
        return - (self._st_src_lambda_hf * st_weighted_l1_hf + self._st_src_lambda * st_weighted_l1)

    def _log_regul_l1_starlet_potential(self, kwargs):
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['psi_grid']
        st = self._starlet_pot.decompose(psi_model)[:-1]  # ignore coarsest scale
        st_weighted_l1_hf = jnp.sum(self._st_pot_weights[0] * jnp.abs(st[0]))  # first scale (i.e. high frequencies)
        st_weighted_l1 = jnp.sum(self._st_pot_weights[1:] * jnp.abs(st[1:]))  # other scales
        return - (self._st_pot_lambda_hf * st_weighted_l1_hf + self._st_pot_lambda * st_weighted_l1)

    def _log_regul_l1_battle_source(self, kwargs):
        source_model = kwargs['kwargs_source'][self._idx_pix_src]['image']
        bt = self._battle_src.decompose(source_model)[0]  # consider only first scale
        bt_weighted_l1 = jnp.sum(self._bt_src_weights * jnp.abs(bt))
        return - self._bt_src_lambda * bt_weighted_l1

    def _log_regul_l1_battle_potential(self, kwargs):
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['psi_grid']
        bt = self._battle_pot.decompose(psi_model)[0]  # consider only first scale
        bt_weighted_l1 = jnp.sum(self._bt_pot_weights * jnp.abs(bt))
        return - self._bt_pot_lambda * bt_weighted_l1

    def _log_regul_positivity_source(self, kwargs):
        source_model = kwargs['kwargs_source'][self._idx_pix_src]['image']
        return - self._pos_src_lambda * jnp.abs(jnp.sum(jnp.minimum(0., source_model)))

    def _log_regul_positivity_potential(self, kwargs):
        psi_model = kwargs['kwargs_lens'][self._idx_pix_pot]['psi_grid']
        return - self._pos_pot_lambda * jnp.abs(jnp.sum(jnp.minimum(0., psi_model)))
