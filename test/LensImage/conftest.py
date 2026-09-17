# Ensure pkg_resources (needed by the local jax_cosmo development install) is
# available even on Python environments where setuptools ≥ 72 no longer ships
# it as a top-level package.  We provide a minimal mock so that the import
# chain herculens → MPMassModel → jax_cosmo can complete without error.
# This has no effect in environments where pkg_resources is already installed.

import sys
from unittest.mock import MagicMock

if 'pkg_resources' not in sys.modules:
    _mock = MagicMock()
    _mock.DistributionNotFound = Exception
    _mock.get_distribution.return_value = MagicMock(version='0.0.0')
    sys.modules['pkg_resources'] = _mock
