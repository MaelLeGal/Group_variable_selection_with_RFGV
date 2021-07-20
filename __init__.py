from src import *

# import sys
# import logging
# import os
# import random
#
# logger = logging.getLogger(__name__)
#
#
# # PEP0440 compatible formatted version, see:
# # https://www.python.org/dev/peps/pep-0440/
# #
# # Generic release markers:
# #   X.Y
# #   X.Y.Z   # For bugfix releases
# #
# # Admissible pre-release markers:
# #   X.YaN   # Alpha release
# #   X.YbN   # Beta release
# #   X.YrcN  # Release Candidate
# #   X.Y     # Final release
# #
# # Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# # 'X.Y.dev0' is the canonical version of 'X.Y.dev'
# #
# __version__ = '0.0.1'
#
#
# # On OSX, we can get a runtime error due to multiple OpenMP libraries loaded
# # simultaneously. This can happen for instance when calling BLAS inside a
# # prange. Setting the following environment variable allows multiple OpenMP
# # libraries to be loaded. It should not degrade performances since we manually
# # take care of potential over-subcription performance issues, in sections of
# # the code where nested OpenMP loops can happen, by dynamically reconfiguring
# # the inner OpenMP runtime to temporarily disable it while under the scope of
# # the outer OpenMP parallel section.
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
#
# # Workaround issue discovered in intel-openmp 2019.5:
# # https://github.com/ContinuumIO/anaconda-issues/issues/11294
# os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
#
# from . import _distributor_init  # noqa: F401
# from . import __check_build  # noqa: F401
#
# __all__ = ['src']
#
#
# def setup_module(module):
#     """Fixture for the tests to assure globally controllable seeding of RNGs"""
#
#     import numpy as np
#
#     # Check if a random seed exists in the environment, if not create one.
#     _random_seed = os.environ.get('SKLEARN_SEED', None)
#     if _random_seed is None:
#         _random_seed = np.random.uniform() * np.iinfo(np.int32).max
#     _random_seed = int(_random_seed)
#     print("I: Seeding RNGs with %r" % _random_seed)
#     np.random.seed(_random_seed)
#     random.seed(_random_seed)