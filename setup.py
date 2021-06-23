import sys
import os

from sklearn._build_utils import cythonize_extensions


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('rfgv', parent_package, top_path)

    # config.add_subpackage("src/Cython")
    config.add_subpackage("src/Python")
    config.add_subpackage("test")

    if 'sdist' not in sys.argv:
        cythonize_extensions(top_path, config)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())