from setuptools import  setup, find_packages

#with open("requirements.txt") as req_file:
#	requirements = [req.strip() for req in req_file.read().splitlines()]

setup(
	name="rfgi",
	version="0.0.1",
	description="A Decision tree and random forest package for grouped variables",
	url='https://github.com/apoterie/Group_variable_selection_with_RFGV',
	author="Le Gal Maël",
	author_email="mael.legal@live.fr",
	license="LMBA-IRISA",
	package=find_packages(),
	install_requires=requirements,
	zip_safe=False)

#import sys
#import os

#from sklearn._build_utils import cythonize_extensions


#def configuration(parent_package='', top_path=None):
#    from numpy.distutils.misc_util import Configuration
#    import numpy
#
#    libraries = []
#    if os.name == 'posix':
#        libraries.append('m')

#    config = Configuration('rfgv', parent_package, top_path)
#
#    # config.add_subpackage("src")
#    # config.add_subpackage("src/Cython")
#    config.add_subpackage("src/Python")
#    config.add_subpackage("test")

#    if 'sdist' not in sys.argv:
#        cythonize_extensions(top_path, config)
#
#    return config

#if __name__ == '__main__':
#    from numpy.distutils.core import setup
#    setup(**configuration(top_path='').todict())