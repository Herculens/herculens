# NOTE: this file is useful for backward-compatibility only, 
# as the preferred way is now to pass the profile class directly 
# to the MassModel() constructor.

from herculens.MassModel.Profiles.gaussian_potential import Gaussian
from herculens.MassModel.Profiles.point_mass import PointMass
from herculens.MassModel.Profiles.multipole import Multipole
from herculens.MassModel.Profiles.shear import Shear, ShearGammaPsi
from herculens.MassModel.Profiles.sis import SIS
from herculens.MassModel.Profiles.sie import SIE
from herculens.MassModel.Profiles.nie import NIE
from herculens.MassModel.Profiles.epl import EPL
from herculens.MassModel.Profiles.dpie import DPIE_GLEE
from herculens.MassModel.Profiles.pixelated import (
    PixelatedPotential,
    PixelatedFixed,
    PixelatedPotentialDirac,
)

# mapping between the string name to the mass profile class.
STRING_MAPPING = {
    'EPL': EPL,
    'NIE': NIE,
    'SIE': SIE,
    'SIS': SIS,
    'DPIE_GLEE': DPIE_GLEE,
    'GAUSSIAN': Gaussian,
    'POINT_MASS': PointMass,
    'SHEAR': Shear,
    'SHEAR_GAMMA_PSI': ShearGammaPsi,
    'MULTIPOLE': Multipole,
    'PIXELATED': PixelatedPotential,
    'PIXELATED_DIRAC': PixelatedPotentialDirac,
    'PIXELATED_FIXED': PixelatedFixed,
}

SUPPORTED_MODELS = list(STRING_MAPPING.keys())
