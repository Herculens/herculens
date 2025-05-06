from .info import version_info, __version__, __author__, __license__

from .Coordinates.pixel_grid import PixelGrid
from .Instrument.psf import PSF
from .Instrument.noise import Noise

# NOTE: non-elliptical profiles will be suppressed in the future
from .LightModel.light_model import LightModel
from .LightModel.light_model_multiplane import MPLightModel
from .LightModel.Profiles.gaussian import Gaussian as GaussianLight, GaussianEllipse as GaussianEllipseLight
from .LightModel.Profiles.multipole import Multipole
from .LightModel.Profiles.sersic import Sersic, SersicElliptic
from .LightModel.Profiles.shapelets import Shapelets
from .LightModel.Profiles.uniform import Uniform
from .LightModel.Profiles.pixelated import Pixelated as PixelatedLight

from .MassModel.mass_model import MassModel
from .MassModel.mass_model_multiplane import MPMassModel
from .MassModel.Profiles.sis import SIS  # NOTE: this will be suppressed in the future
from .MassModel.Profiles.sie import SIE
from .MassModel.Profiles.nie import NIE
from .MassModel.Profiles.epl import EPL
from .MassModel.Profiles.shear import Shear, ShearGammaPsi
from .MassModel.Profiles.gaussian_potential import Gaussian as GaussianPotential
from .MassModel.Profiles.point_mass import PointMass
from .MassModel.Profiles.multipole import Multipole
from .MassModel.Profiles.pixelated import (
    PixelatedPotential, PixelatedPotentialDirac, PixelatedFixed
)
from .MassModel.Profiles.piemd import PIEMD
from .MassModel.Profiles.dpie import DPIE

from .PointSourceModel.point_source_model import PointSourceModel

try:
    from .GenericModel.correlated_field import CorrelatedField
except ImportError:
    # an error will be raised via the from herculens.Util.jifty_util submodule, 
    # typically when instantiating the CorrelatedField class
    pass

from .LensImage.lens_image import LensImage, LensImage3D
from .LensImage.lens_image_multiplane import MPLensImage
from .Inference.loss import Loss
from .Inference.ProbModel.numpyro import NumpyroModel
from .Inference.Optimization.jaxopt import JaxoptOptimizer
from .Inference.Optimization.optax import OptaxOptimizer
from .Analysis.plot import Plotter

from .Util import param_util as prmu
from .Util import plot_util as pltu
