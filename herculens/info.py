"""PACKAGE INFO

This module provides some basic information about the package.

"""

# Set the package release version
version_info = (0, 0, 1)
__version__ = '.'.join(str(c) for c in version_info)

# Set the package details
__author__ = 'Austin Peel, Aymeric Galan'
__email__ = 'austin.peel@epfl.ch'
__year__ = '2021'
__url__ = 'https://github.com/austinpeel/herculens'
__description__ = 'JAX-enabled autodifferentiable strong lens modelling'
__python__ = '>=3.7'
__requires__ = [
    'numpy>=1.20.3', 
    'scipy>=1.6.3', 
    'jax>=0.3.14', 
    'jaxlib>=0.3.14', 
    'optax>=0.1.0',
    'chex>=0.1.4',
    'matplotlib>=3.0.0',
    'findiff==0.8.9',
]  # Package dependencies

# Default package properties
__license__ = 'MIT'
__about__ = ('{} Author: {}, Email: {}, Year: {}, {}'
             ''.format(__name__, __author__, __email__, __year__,
                       __description__))
__setup_requires__ = ['pytest-runner', ]
__tests_require__ = ['pytest', 'pytest-cov', 'pytest-pep8']
