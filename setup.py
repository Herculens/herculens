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
python_requires = '>=3.9'

# Minimal required packages (see requirements.txt for versions)
install_requires = [
    'numpy',
    'scipy',
    'jax',
    'jaxlib',
    'objax',
    'scikit-image',
    # 'https://github.com/aymgal/utax.git',  # use requirements.txt
]

# Minimal optional packages (see requirements.txt for versions and complete list)
install_optional = [
    'matplotlib',    # for plotting
    'optax',         # for optimizers
    'jaxopt',        # for optimizers
    'findiff',       # for building matrix with finite difference coefficients
    'numpyro',       # for SVI and HMC sampling
    'blackjax',      # for JAX-based HMC sampling
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
    setup_requires=['pytest-runner',],
    tests_require=['pytest', 'pytest-cov', 'pytest-pep8'],
)
