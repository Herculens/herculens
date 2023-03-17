__all__ = ['BaseRegulization']


class BaseRegulization(object):
    """Base class for all regularization methods."""

    def __init__(self, model_type, profile_index, mass_form=None):
        if model_type not in ['source', 'lens_light', 'mass_light']:
            raise ValueError("Unsupported model type for regularization")
        self.model_type = model_type
        self.profile_idx = profile_index
        self._mass_form = mass_form

    def log_prob(self, kwargs_hyperparams, kwargs_params):
        raise NotImplementedError

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_pixel_params(self, kwargs_params):
        if self.model_type == 'source':
            key = 'kwargs_source'
        elif self.model_type == 'lens_light':
            key = 'kwargs_lens_light'
        elif self.model_type == 'lens_mass':
            key = 'kwargs_lens'
        idx = self.profile_idx
        return kwargs_params[key][idx]['pixels']

    @property
    def has_weights(self):
        # some regularization methods require weights
        return hasattr(self, 'weights')
