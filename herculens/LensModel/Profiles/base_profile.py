__all__ = ['LensProfileBase']


class LensProfileBase(object):
    """Base class for all lens model profiles."""

    def __init__(self, *args, **kwargs):
        self._static = False

    def function(self, *args, **kwargs):
        """Lensing potential.

        Parameters
        ----------
        x, y : float or array_like
            Coordinate(s) at which to evaluate the potential.
        params : dict
            Keyword arguments of the profile.

        Raises
        ------
        ValueError if this method is not defined in inheriting classes.

        """
        raise ValueError('Method `function` is not defined for this profile.')

    def derivatives(self, *args, **kwargs):
        """Deflection angles (alpha).

        Parameters
        ----------
        x, y : float or array_like
            Coordinate(s) at which to evaluate deflection angles.
        params : dict
            Keyword arguments of the profile.

        Raises
        ------
        ValueError if this method is not defined in inheriting classes.

        """
        raise ValueError('Method `derivatives` is not defined for this profile.')

    def hessian(self, *args, **kwargs):
        """Hessian matrix, i.e. d^2f/dx^2, d^f/dy^2, d^2/dxdy.

        Parameters
        ----------
        x, y : float or array_like
            Coordinate(s) at which to evaluate second derivatives.
        params : dict
            Keyword arguments of the profile.

        Raises
        ------
        ValueError if this method is not defined in inheriting classes.

        """
        raise ValueError('Method `hessian` is not defined for this profile.')

    def density_lens(self, *args, **kwargs):
        """Density evaluated at a 3D radial position.

        The integral along the line-of-sight of this quantity is convergence.

        Parameters
        ----------
        r : array_like
            Coordinate(s) at which to evaluate the density.
        params : dict
            Keyword arguments of the profile.

        Raises
        ------
        ValueError if this method is not defined in inheriting classes.

        """
        raise ValueError('Method `density_lens` is not defined for this profile.')

    def mass_3d_lens(self, *args, **kwargs):
        """Mass enclosed in a 3D sphere of radius r (in angular units).

        Parameters
        ----------
        r : array_like
            Coordinate(s) at which to evaluate the mass.
        params : dict
            Keyword arguments of the profile.

        Raises
        ------
        ValueError if this method is not defined in inheriting classes.

        """
        raise ValueError('Method `mass_3d_lens` is not defined for this profile.')

    def set_static(self, **kwargs):
        """Pre-compute certain position-independent properties of the lens profile.

        For certain lens models, some private attributes are initialized.

        Parameters
        ----------
        params : dict
            Keyword arguments of the profile.

        """
        pass

    def set_dynamic(self):
        """Delete pre-computed variables for certain lens models."""
        pass
