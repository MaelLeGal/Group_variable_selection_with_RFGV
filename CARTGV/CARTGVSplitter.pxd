# -*- coding: utf-8 -*-
"""
Created on Tue Mar 2 13:34:12 2021

@author: Alphonse
"""

# _tree = __import__('scikit-learn.sklearn.tree._tree', globals(), locals(), ['Tree', 'TreeBuilder', "Node"], 0)
# _splitter = __import__('scikit-learn.sklearn.tree._splitter', globals(), locals(), ['Splitter'], 0)

# Tree = _tree.Tree
# TreeBuilder = _tree.TreeBuilder
# Node = _tree.Node

# Splitter = _splitter.Splitter

import numpy as np
cimport numpy as np

from cpython cimport Py_INCREF, PyObject, PyTypeObject

from CARTGVCriterion cimport CARTGVCriterion

from sklearn.tree._tree cimport DTYPE_t
from sklearn.tree._tree cimport DOUBLE_t
from sklearn.tree._tree cimport SIZE_t
from sklearn.tree._tree cimport INT32_t
from sklearn.tree._tree cimport UINT32_t

from sklearn.tree._tree cimport Tree, TreeBuilder, Node
from sklearn.tree._splitter cimport Splitter


cdef struct CARTGVSplitRecord:
    # Data to track sample split
    double improvement      # Impurity improvement given parent node.
    double* impurity_childs # Impurity of the split childs.
    SIZE_t* starts          # Array containing the index at which each childs start in the samples array.
    SIZE_t* ends            # Array containing the index of the end of each childs in the samples array.
    char* splitting_tree
    int n_childs

cdef class CARTGVSplitter():
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef public CARTGVCriterion criterion      # Impurity criterion
    cdef public SIZE_t max_grouped_features      # Number of features to test
    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
    cdef public double min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t n_samples                # X.shape[0]
    cdef double weighted_n_samples       # Weighted number of samples
    # cdef SIZE_t** groups
    cdef SIZE_t n_groups
    cdef int[:] len_groups
    cdef SIZE_t* features                # Feature indices in X
    cdef SIZE_t n_features               # X.shape[1]
    cdef object feature_values         # temp. array holding feature values

    cdef SIZE_t start                    # Start position for the current node
    cdef SIZE_t end                      # End position for the current node

    cdef const DOUBLE_t[:, ::1] y
    cdef DOUBLE_t* sample_weight
    
    cdef TreeBuilder splitting_tree_builder
    cdef Tree splitting_tree
    
    cdef int[:,:] groups
    
    cdef object X

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int init(self, object X, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, object groups) except -1

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1

    cdef int node_split(self,
                        double impurity,   # Impurity of the node
                        CARTGVSplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1

    cdef void node_value(self, double* dest) nogil

    cdef double node_impurity(self) nogil