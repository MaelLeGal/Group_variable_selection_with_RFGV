import numpy as np
cimport numpy as np
import sys

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
    double improvement              # Impurity improvement given parent node.
    double* impurity_childs         # Impurity of the split childs.
    SIZE_t* starts                  # Array containing the index at which each childs start in the samples array.
    SIZE_t* ends                    # Array containing the index of the end of each childs in the samples array.
    unsigned char* splitting_tree   # The splitting tree serialized
    int n_childs                    # The number of childs in the splitting tree
    int group

cdef class CARTGVSplitter():
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef CARTGVCriterion criterion              # Impurity criterion
    cdef public SIZE_t max_grouped_features     # Number of features to test
    cdef public SIZE_t min_samples_leaf         # Min samples in a leaf
    cdef public double min_weight_leaf          # Minimum weight in a leaf

    cdef object random_state                    # Random state
    cdef UINT32_t rand_r_state                  # sklearn_rand_r random number state

    cdef SIZE_t* samples                        # Sample indices in X, y
    cdef SIZE_t n_samples                       # X.shape[0]
    cdef double weighted_n_samples              # Weighted number of samples
    cdef int[:,:] groups                        # The groups
    cdef SIZE_t n_groups                        # The number of groups
    cdef int[:] len_groups                      # The length of each group
    cdef SIZE_t* features                       # Feature indices in X
    cdef SIZE_t n_features                      # X.shape[1]
    cdef object feature_values                  # temp. array holding feature values
    cdef int[:,:] classes                       # The classes in the responses
    cdef int[:] n_classes                       # The number of each classes in the responses

    cdef SIZE_t start                           # Start position for the current node
    cdef SIZE_t end                             # End position for the current node

    cdef DOUBLE_t* sample_weight                # The weight of each sample

    cdef TreeBuilder splitting_tree_builder     # The builder of the splitting tree
    cdef Tree splitting_tree                    # The splitting tree

#    cdef const DTYPE_t[:,:] X                               # The datas
    cdef const DOUBLE_t[:, ::1] y               # The responses

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

    cdef void node_value(self, double* dest) nogil

    cdef double node_impurity(self) nogil

    cdef np.ndarray group_sample(self, int[:] group, int len_group, int start, int end)

    cdef int reset_scikit_learn_instances(self, np.ndarray y, int len_group)

    cdef int splitting_tree_construction(self, np.ndarray Xf, np.ndarray y)

#    cdef int get_splitting_tree_n_leaves(self)

    cdef int get_splitting_tree_leaves(self, Node** sorted_leaves)

    cdef int get_splitting_tree_leaves_samples_and_pos(self, SIZE_t** starts, SIZE_t** ends, Node* sorted_leaves, SIZE_t n_leaves, SIZE_t*** samples_leaves, SIZE_t n_samples)

    cdef int switch_best_splitting_tree(self, double current_proxy_improvement, double* best_proxy_improvement, CARTGVSplitRecord* best, CARTGVSplitRecord* current, SIZE_t* starts, SIZE_t* ends, SIZE_t n_leaves, int group, SIZE_t** sorted_obs)

    cdef int node_split(self, double impurity, CARTGVSplitRecord* split, SIZE_t* n_constant_features)

    ########################################## TESTS #############################################

    cpdef int test_init(self, object X, DOUBLE_t[:, ::1] y,
                  np.ndarray sample_weight, object groups)

    cpdef int test_node_reset(self, SIZE_t start, SIZE_t end, double weighted_n_node_samples)

    cpdef double test_node_value(self, double dest)

    cpdef double test_node_impurity(self)

    cpdef np.ndarray test_group_sample(self, int[:] group, int len_group, int start, int end)

    cpdef int test_reset_scikit_learn_instances(self, np.ndarray y, int len_group)

    cpdef int test_splitting_tree_construction(self, np.ndarray Xf, np.ndarray y)

    cpdef int test_get_splitting_tree_leaves(self)

    cpdef int test_get_splitting_tree_leaves_samples_and_pos(self)

    cpdef int test_switch_best_splitting_tree(self)

    cpdef int test_node_split(self)

cdef class BaseDenseCARTGVSplitter(CARTGVSplitter):

    cdef SIZE_t n_total_samples
    cdef const DTYPE_t[:,:] X