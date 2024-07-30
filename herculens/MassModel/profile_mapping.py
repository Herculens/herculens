# NOTE: this file is useful for backward-compatibility only, 
# as the preferred way is now to pass the profile class directly 
# to the MassModel() constructor.

from herculens.MassModel.Profiles import (
    gaussian_potential, 
    point_mass, 
    multipole,
    shear, 
    sie, 
    sis, 
    nie, 
    epl, 
    pixelated
)

SUPPORTED_MODELS = [
    'EPL', 'NIE', 'SIE', 'SIS', 'GAUSSIAN', 'POINT_MASS', 
    'SHEAR', 'SHEAR_GAMMA_PSI', 'MULTIPOLE',
    'PIXELATED', 'PIXELATED_DIRAC', 'PIXELATED_FIXED',
]

# mapping between the string name to the mass profile class.
STRING_MAPPING = {
    'EPL': epl.EPL,
    'NIE': nie.NIE,
    'SIE': sie.SIE,
    'SIS': sis.SIS,
    'GAUSSIAN': gaussian_potential.Gaussian,
    'POINT_MASS': point_mass.PointMass,
    'SHEAR': shear.Shear,
    'SHEAR_GAMMA_PSI': shear.ShearGammaPsi,
    'MULTIPOLE': multipole.Multipole,
    'PIXELATED': pixelated.PixelatedPotential,
    'PIXELATED_DIRAC': pixelated.PixelatedPotentialDirac,
    'PIXELATED_FIXED': pixelated.PixelatedFixed,
}
