import setuptools
import os

name = 'herculens'

release_info = {}
infopath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                           name, 'info.py'))
with open(infopath) as open_file:
    exec(open_file.read(), release_info)

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

# Use the README as long description but remove the second version 
# of the logo (readable only by GitHub's dark mode interface)
long_description = readme.replace('<img src="https://raw.githubusercontent.com/Herculens/herculens/main/images/horizontal_dark_bg.png#gh-dark-mode-only" width="600" alt="Herculens logo" />', '')

# Python version
python_requires = '>=3.10'

# Minimal required packages (see also requirements.txt)
install_requires = [
    'numpy>=1.20.3',
    'scipy>=1.6.3',
    'jax>=0.7.0',
    'jaxlib>=0.7.0',
    'objax>=1.8.0',
    'scikit-image>=0.20.0',
    'utax @ git+https://github.com/aymgal/utax.git@main',
]

# Minimal optional packages (see also requirements.txt for even more optional packages)
install_optional = [
    'helens @ git+https://github.com/Herculens/helens.git@main', # for a JAX-based lens equation solver
    'jaxinterp2d @ git+https://github.com/adam-coogan/jaxinterp2d@master',  # for faster bilinear interpolation for pixelated profiles
    'matplotlib>=3.7.0',    # for plotting
    'optax>=0.2.0',         # for optimizers
    'numpyro>=0.20.0',      # for probabilistic modelling and sampling algorithms
    'nifty>=9.1.0',         # for correlated fields and VI sampling
]

version = release_info['__version__']

setuptools.setup(
    name=name,
    author=release_info['__author__'],
    author_email=release_info['__email__'],
    version=version,
    url=release_info['__url__'],
    download_url=f"https://github.com/Herculens/herculens/archive/refs/tags/v{version}.tar.gz",
    packages=setuptools.find_packages(),
    license=release_info['__license__'],
    description=release_info['__description__'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=["herculens", "lensing", "gravitation", "astrophysics"],

    python_requires=python_requires,
    install_requires=install_requires,
    extras_require={
        "opt": install_optional,  # installable via `pip install herculens[opt]`
    },
)
