import setuptools
import os

__name__ = 'jaxtronomy'

release_info = {}
infopath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                           __name__, 'info.py'))
with open(infopath) as open_file:
    exec(open_file.read(), release_info)

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name=__name__,
    author=release_info['__author__'],
    author_email=release_info['__email__'],
    version=release_info['__version__'],
    url=release_info['__url__'],
    packages=setuptools.find_packages(),
    python_requires=release_info['__python__'],
    install_requires=release_info['__requires__'],
    license=release_info['__license__'],
    description=release_info['__about__'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    # setup_requires=release_info['__setup_requires__'],
    # tests_require=release_info['__tests_require__']
)
