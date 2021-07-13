import numpy as np
from jaxtronomy.LightModel.light_model_base import LightModelBase

__all__ = ['LinearBasis']


class LinearBasis(LightModelBase):
    """Base class for light models inheriting from LightModelBase."""
    def __init__(self, light_model_list, smoothing=0.0000001):
        """Create a LinearBasis object.

        Parameters
        ----------
        light_model_list : list of str
            Light model types.
        smoothing : float
            Smoothing factor for some models (deprecated).

        """
        super(LinearBasis, self).__init__(light_model_list, smoothing)

    @property
    def param_name_list(self):
        """Get parameter names as a list of strings for each light model."""
        return [func.param_names for func in self.func_list]

    def num_param_linear(self, kwargs_list, list_return=False):
        """Get the total number of linear parameters of all light models.

        Parameters
        ----------
        kwargs_list : list of dicts
            Parameters for each light model.
        list_return : bool, optional
            If True, return a list of the number of linear parameters of each
            model separately. Otherwise, get the total for all models combined.

        Returns
        -------
        count(s) : int or list
            Number of linear basis set coefficients.

        """
        n_list = self.num_param_linear_list(kwargs_list)
        if not list_return:
            return np.sum(n_list)
        return n_list

    def num_param_linear_list(self, kwargs_list):
        """Get the number of linear parameters of each light model.

        Parameters
        ----------
        kwargs_list : list of dicts
            Parameters for each light model.

        Returns
        -------
        counts : list
            Number of linear basis set coefficients.

        Notes
        -----
        This is trivial for the current light profiles provided, but more
        complex profiles might require some calculation, e.g. shapelets.

        """
        n_list = []
        for kwargs, model in zip(kwargs_list, self.profile_type_list):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'UNIFORM']:
                n_list += [1]
            elif model == 'PIXELATED':
                n_list += [kwargs['image'].size]
            else:
                raise ValueError(f"Model type {model} is not valid!")
        return n_list

    def check_positive_flux_profile(self, kwargs_list):
        """Verify non-negativity of flux amplitudes for appropriate profiles.

        Parameters
        ----------
        kwargs_list : list of dicts
            Parameters for each light model.

        Returns
        -------
        validity : bool
            True if no model has negative flux amplitude.

        """
        pos_bool = True
        for kwargs, model in zip(kwargs_list, self.profile_type_list):
            if 'amp' in kwargs:
                if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'UNIFORM']:
                    if kwargs['amp'] < 0:
                        pos_bool = False
                        break
        return pos_bool
