from scipy.optimize import minimize


class LensEquationSolver(object):
    """Find the image plane positions of a point source given a mass model."""
    def __init__(self, mass_model):
        self.mass_model = mass_model

    def solve(self, source_x, source_y):
        """
        Compute the (multiple) image positions corresponding to a source plane
        point.

        Parameters
        ----------
        source_x : float
            X value.
        source_y : float
            Y value.

        """
