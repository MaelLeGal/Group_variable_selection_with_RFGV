# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:05:19 2021

@author: Alphonse
"""
import numpy as np
cimport numpy as np

from sklearn.tree._tree cimport Tree, TreeBuilder, Node
from sklearn.tree._splitter cimport Splitter, SplitRecord

from CARTGVSplitter cimport CARTGVSplitter, CARTGVSplitRecord

from sklearn.tree._tree cimport DTYPE_t
from sklearn.tree._tree cimport DOUBLE_t
from sklearn.tree._tree cimport SIZE_t
from sklearn.tree._tree cimport INT32_t
from sklearn.tree._tree cimport UINT32_t

cdef struct test:
  SIZE_t parent                         # The parent of the node
  DOUBLE_t impurity                     # The impurity of the node
  SIZE_t n_node_samples                 # The number of samples in the node
  DOUBLE_t weighted_n_node_samples      # The number of weighted samples in the node
  int group                             # The group selected for the split of the node
  int n_childs                          # The number of childs of the node

cdef struct CARTGVNode:
  SIZE_t* childs                        # The childs of the node
  SIZE_t parent                         # The parent of the node
  unsigned char* splitting_tree         # The serialized splitting tree of the node
  DOUBLE_t impurity                     # The impurity of the node
  SIZE_t n_node_samples                 # The number of samples in the node
  DOUBLE_t weighted_n_node_samples      # The number of weighted samples in the node
  int group                             # The group selected for the split of the node
  int n_childs                          # The number of childs of the node
  int current_child
  
cdef class CARTGVTree():
  
  # Input/Output layout
  cdef public SIZE_t n_grouped_features # Number of feature's group
  cdef SIZE_t* n_classes                # Number of classes un y[:, k]
  cdef public SIZE_t n_outputs          # Number of outputs in k
  cdef public SIZE_t max_n_classes      # max(n_classes)
  
  # Inner structures : values are stored separately from node structure,
  # since size is determined at runtime.
  cdef public SIZE_t max_depth          # Max depth of the tree
  cdef public SIZE_t node_count         # Counter for node IDs
  cdef public SIZE_t capacity           # Capacity of tree, in terms of nodes
  cdef CARTGVNode* nodes                #Array of nodes
  cdef double* value                    #(capacity, n_outputs, max_n_classes) array of values
  cdef SIZE_t value_stride              # = n_outputs * max_n_classes
  cdef int** groups                     # The different group
  
  #Methods
  cdef SIZE_t _add_node(self, SIZE_t parent, bint is_leaf,
                        unsigned char* splitting_tree, double impurity,
                        SIZE_t n_node_samples, int n_childs,
                        double weighted_n_samples, int group) nogil except -1
  cdef int _resize(self,SIZE_t capacity) nogil except -1
  cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1
  
  cdef np.ndarray _get_value_ndarray(self)
  cdef np.ndarray _get_node_ndarray(self)
  
  cpdef np.ndarray predict(self, object X)
  
  cpdef np.ndarray apply(self, object X)
  # cdef np.ndarray _apply_dense(self, object X)
  # cdef np.ndarray _apply_sparse_csr(self, object X)
  
  cpdef object decision_path(self, object X)
  # cdef object _decision_path_dense(serlf, object X)
  # cdef object _decision_path_sparse_csr(self,object X)
  
  # cpdef compute_group_importances(self,penality_function,normalize=*)

    ########################################## TESTS #############################################

  cpdef void test_resize_CARTGVTree(self, capacity)

  cpdef void test_add_node(self, CARTGVSplitter splitter, SIZE_t start, SIZE_t end)


cdef class CARTGVTreeBuilder():
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef CARTGVSplitter splitter                # The splitter of the CARTGV Tree

    cdef SIZE_t min_samples_split               # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf                # Minimum number of samples in a leaf
    cdef double min_weight_leaf                 # Minimum weight in a leaf
    cdef SIZE_t max_depth                       # Maximal tree depth
    cdef SIZE_t mgroup                          # Number of group selected during the RF
    cdef SIZE_t mvar                            # Number of variable in the group selected during the RF
    cdef double min_impurity_split              # The minimum impurity needed in a split
    cdef double min_impurity_decrease           # Impurity threshold for early stopping
    cdef TreeBuilder splitting_tree_builder     # The builder of the splitting trees

    cpdef void build(self, CARTGVTree tree, object X, np.ndarray y, object groups,
                np.ndarray sample_weight=*)
    cdef _check_input(self, object X, np.ndarray y, np.ndarray sample_weight)

    ########################################## TESTS #############################################

    cpdef void test_build(self, CARTGVTree tree, object X, np.ndarray y, object groups, np.ndarray sample_weight=*)

  
  