# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:05:19 2021

@author: Alphonse
"""
import numpy as np
cimport numpy as np

from sklearn.tree._tree cimport Tree, TreeBuilder, Node
from sklearn.tree._splitter cimport Splitter, SplitRecord

#from .CARTGVSplitter cimport CARTGVSplitter, CARTGVSplitRecord
from CARTGVSplitter cimport CARTGVSplitter, CARTGVSplitRecord

from sklearn.tree._tree cimport DTYPE_t
from sklearn.tree._tree cimport DOUBLE_t
from sklearn.tree._tree cimport SIZE_t
from sklearn.tree._tree cimport INT32_t
from sklearn.tree._tree cimport UINT32_t

ctypedef struct CARTGVNode:
  SIZE_t* childs                        # The childs of the node
  SIZE_t parent                         # The parent of the node
  unsigned char* splitting_tree         # The serialized splitting tree of the node
  DOUBLE_t impurity                     # The impurity of the node
  SIZE_t n_node_samples                 # The number of samples in the node
  DOUBLE_t weighted_n_node_samples      # The number of weighted samples in the node
  int group                             # The group selected for the split of the node
  int n_childs                          # The number of childs of the node
  int current_child                     # The current number of children of the node
  int start                             # The starting position in the samples array
  int end                               # The ending position in the samples array
  int depth                             # The depth of the node in the tree

#cdef class CARTGVNodeClass():

    #cdef CARTGVNode *ptr

    #np.ndarray childs
    #SIZE_t parent
    #Tree splitting_tree
    #DOUBLE_t impurity
    #SIZE_t n_node_samples
    #DOUBLE_t weighted_n_node_samples
    #int group                             # The group selected for the split of the node
    #int n_childs                          # The number of childs of the node
    #int current_child                     # The current number of children of the node
    #int start                             # The starting position in the samples array
    #int end                               # The ending position in the samples array
    #int depth                             # The depth of the node in the tree

    #cdef CARTGVNodeClass from_ptr(CARTGVNode *ptr)

cdef class CARTGVTree():
  
  # Input/Output layout
  cdef public SIZE_t n_groups           # Number of group
  cdef int n_features                   # Total number of variables
  cdef public np.ndarray len_groups     # Number of features per group
  cdef SIZE_t* n_classes                # Number of classes un y[:, k]
  cdef public SIZE_t n_outputs          # Number of outputs in k
  cdef public SIZE_t max_n_classes      # max(n_classes)
  
  # Inner structures : values are stored separately from node structure,
  # since size is determined at runtime.
  cdef public SIZE_t max_depth          # Max depth of the tree
  cdef public SIZE_t node_count         # Counter for node IDs
  cdef public int n_leaves              # Number of leaves
  cdef public SIZE_t capacity           # Capacity of tree, in terms of nodes
  cdef CARTGVNode* nodes                # Array of nodes
  cdef SIZE_t* nodes_cart_idx           # Array of index of the leaves in splitting tree
  cdef double* value                    #(capacity, n_outputs, max_n_classes) array of values
  cdef SIZE_t value_stride              # = n_outputs * max_n_classes
  cdef object groups                    # The different group

  #Methods

  # Add a node to the tree
  #
  # params parent : a SIZE_t representing the id of the parent node
  # params is_leaf : a bint, a boolean, True if the node is a leaf, false otherwise
  # params splitting_tree : an unsigned char*, the serialized splitting tree of the node
  # params impurity : a double, the impurity of the node
  # params n_node_samples : a SIZE_t, the number of samples in the node
  # params n_childs : an int, the number of childs of the node, can be 0 if it is a leaf
  # params weighted_n_samples : a double, the number of weighted samples
  # params group : an int, the group that was used for splitting the node
  # params start : an int, the starting position in the samples array
  # params end : an int, the ending position in the samples array
  # params depth : an int, the depth of the node in the tree
  # params cart_idx : a SIZE_t, the index of the node in the splitting tree
  cdef SIZE_t _add_node(self, SIZE_t parent, bint is_leaf,
                        unsigned char* splitting_tree, double impurity,
                        SIZE_t n_node_samples, int n_childs,
                        double weighted_n_samples, int group, int start, int end, int depth, SIZE_t cart_idx) nogil except -1

  # Resize the number of nodes the tree can accept
  #
  # params capacity : a SIZE_t, the new capacity of the tree
  cdef int _resize(self,SIZE_t capacity) nogil except -1

  # Resize the number of nodes the tree can accept
  #
  # params capacity : a SIZE_t, the new capacity of the tree
  cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1

  # Get the array value of the tree
  cdef np.ndarray _get_value_ndarray(self)

  # Get the array of nodes of the tree (Doesn't work yet)
  cdef np.ndarray _get_node_ndarray(self)

  # Get the array of index in their corresponding parent splitting trees for each node
  cdef np.ndarray _get_nodes_cart_idx_ndarray(self)

  # Predict the value(s) or class(es) for new observations
  #
  # params X, an object (list, ndarray), the data to be predicted
  cpdef np.ndarray predict(self, object X)

  # Used by the predict function, select the correct function to do the computing (dense, sparse)
  #
  # params X, an object (list, ndarray), the data to be predicted
  cpdef np.ndarray apply(self, object X)

  # Used by the apply function for dense data, does the computing for the predict (passing through the tree)
  #
  # params X, an object (list, ndarray), the data to be predicted
  cdef np.ndarray _apply_dense(self, object X)
  # cdef np.ndarray _apply_sparse_csr(self, object X)

  #cpdef np.ndarray sobol_indice(self, object X, int group_j, int[::1] in_bag_idx, int[::1] oob_idx)

  #cdef np.ndarray apply_sobol(self, object X, int group_j, int[::1] in_bag_idx, int[::1] oob_idx)

  # Give the decision path for a new observation
  #
  # params X, an object (list, ndarray), the data
  cpdef object decision_path(self, object X)
  # cdef object _decision_path_dense(serlf, object X)
  # cdef object _decision_path_sparse_csr(self,object X)
  
  # cpdef compute_group_importances(self,penality_function,normalize=*)

    ########################################## TESTS #############################################

  cpdef void test_resize_CARTGVTree(self, capacity)

  cpdef void test_add_node(self, CARTGVSplitter splitter, SIZE_t start, SIZE_t end)


cdef class CARTGVTreeBuilder():
    # The CARTGVTreeBuilder recursively builds a CARTGVTree object from training samples,
    # using a CARTGVSplitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef CARTGVSplitter splitter                # The splitter of the CARTGV Tree

    cdef SIZE_t min_samples_split               # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf                # Minimum number of samples in a leaf
    cdef double min_weight_leaf                 # Minimum weight in a leaf
    cdef SIZE_t max_depth                       # Maximal tree depth
    cdef double min_impurity_split              # The minimum impurity needed in a split
    cdef double min_impurity_decrease           # Impurity threshold for early stopping
    #cdef TreeBuilder splitting_tree_builder     # The builder of the splitting trees

    # Build a CARTGV tree from training samples
    #
    # params tree, a CARTGVTree, the tree that will be filled
    # params X, an object (list, ndarray), the training samples
    # params y, a ndarray, the responses of the training samples
    # params groups, an object (list of list, ndarray dim >=2), a list of list containing the index of each variables of each group
    # Example : [[0,1,2],[2,3],[1,4,5]], three groups of different size, with variables in multiple groups
    # params len_groups, a ndarray, a list of the length of each group
    # params pen, an object (None, "root", "size", "log", or a function with 1 param), the penality function for the impurity
    # params sample_weight, a ndarray, the weight of each sample (can be None)
    cpdef void build(self, CARTGVTree tree, object X, np.ndarray y, object groups, np.ndarray len_groups,
                object pen, np.ndarray sample_weight=*)

    # Check the inputs format
    #
    # params X, an object (list, ndarray), the training samples
    # params y, a ndarray, the responses of the training samples
    # params sample_weight, a ndarray, the weight of each sample (can be None)
    cdef _check_input(self, object X, np.ndarray y, np.ndarray sample_weight)

    ########################################## TESTS #############################################

    cpdef void test_build(self, CARTGVTree tree, object X, np.ndarray y, object groups, np.ndarray len_groups, object pen, np.ndarray sample_weight=*)

  
  