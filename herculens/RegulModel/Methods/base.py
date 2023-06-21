__all__ = ['BaseRegularization']


class BaseRegularization(object):
    """Base class for all regularization methods."""

    param_names = []
    lower_limit_default = {}
    upper_limit_default = {}
    fixed_default = {}

    def __init__(self, model_type, profile_index, mass_form=None):
        if model_type not in ['source', 'lens_light', 'mass_light']:
            raise ValueError("Unsupported model type for regularization")
        self.model_type = model_type
        self.profile_idx = profile_index
        if self.model_type == 'source':
            self._param_key = 'kwargs_source'
        elif self.model_type == 'lens_light':
            self._param_key = 'kwargs_lens_light'
        elif self.model_type == 'lens_mass':
            self._param_key = 'kwargs_lens'
        self._mass_form = mass_form

    def log_prob(self, kwargs_hyperparams, kwargs_params):
        raise NotImplementedError

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_pixel_params(self, kwargs_params):
        return kwargs_params[self._param_key][self.profile_idx]['pixels']

    @property
    def has_weights(self):
        # some regularization methods require weights
        return hasattr(self, 'weights')
