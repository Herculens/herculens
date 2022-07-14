__all__ = ['RegulBase']


class RegulBase(object):
    """Base class for all regularization methods."""

    def __call__(self, *args, **kwargs):
        """Returns the logarithm of the regularization term in the loss function.

        Raises
        ------
        ValueError if this method has not been defined in inheriting classes.

        """
        raise NotImplementedError('Method `__call__` has not been defined for this class.')
    
    def initialize_with_lens_image(self, lens_image):
        pass  # not required for all children classes, depending on the method
