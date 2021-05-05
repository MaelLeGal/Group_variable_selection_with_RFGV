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