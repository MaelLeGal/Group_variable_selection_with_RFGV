from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
  Extension("CARTGVTree",["CARTGVTree.pyx"],include_dirs=[numpy.get_include()]),
  Extension("CARTGVSplitter",["CARTGVSplitter.pyx"],include_dirs=[numpy.get_include()]),
  Extension("CARTGVCriterion",["CARTGVCriterion.pyx"],include_dirs=[numpy.get_include()]),
  Extension("CARTGVutils", ["CARTGVutils.pyx"], include_dirs=[numpy.get_include()])
]

setup(
  name="cartgv",
  ext_modules=cythonize(ext_modules,gdb_debug=True))

# import os
#
# import numpy
# from numpy.distutils.misc_util import Configuration
#
# def configuration(parent_package="", top_path=None):
#     config = Configuration("tree", parent_package, top_path)
#     libraries = []
#     if os.name == 'posix':
#         libraries.append('m')
#     config.add_extension("CARTGVTree",
#                          sources=["CARTGVTree.pyx"],
#                          include_dirs=[numpy.get_include()],
#                          libraries=libraries,
#                          extra_compile_args=["-O3"])
#     config.add_extension("CARTGVSplitter",
#                          sources=["CARTGVSplitter.pyx"],
#                          include_dirs=[numpy.get_include()],
#                          libraries=libraries,
#                          extra_compile_args=["-O3"])
#     config.add_extension("CARTGVCriterion",
#                          sources=["CARTGVCriterion.pyx"],
#                          include_dirs=[numpy.get_include()],
#                          libraries=libraries,
#                          extra_compile_args=["-O3"])
#     config.add_extension("CARTGVutils",
#                          sources=["CARTGVutils.pyx"],
#                          include_dirs=[numpy.get_include()],
#                          libraries=libraries,
#                          extra_compile_args=["-O3"])
#
#     return config
#
# if __name__ == "__main__":
#   from numpy.distutils.core import setup
#   setup(**configuration().todict())