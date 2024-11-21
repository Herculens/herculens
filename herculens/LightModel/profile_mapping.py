# NOTE: this file is useful for backward-compatibility only, 
# as the preferred way is now to pass the profile class directly 
# to the LightModel() constructor.

from herculens.LightModel.Profiles.sersic import (Sersic, SersicElliptic)
from herculens.LightModel.Profiles.multipole import Multipole
from herculens.LightModel.Profiles.gaussian import (Gaussian, GaussianEllipse)
from herculens.LightModel.Profiles.pixelated import Pixelated
from herculens.LightModel.Profiles.uniform import Uniform
from herculens.LightModel.Profiles.shapelets import Shapelets


# mapping between the string name to the mass profile class.
STRING_MAPPING = {
    'SERSIC': Sersic,
    'SERSIC_ELLIPSE': SersicElliptic,
    'SERSIC_SUPERELLIPSE': SersicElliptic,
    'GAUSSIAN': Gaussian,
    'GAUSSIAN_ELLIPSE': GaussianEllipse,
    'MULTIPOLE': Multipole,
    'PIXELATED': Pixelated,
    'UNIFORM': Uniform,
    'SHAPELETS': Shapelets
}

SUPPORTED_MODELS = list(STRING_MAPPING.keys())
