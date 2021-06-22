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
    int group                       # The group used for the split

cdef class CARTGVSplitter():
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef CARTGVCriterion criterion              # Impurity criterion
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
    cdef int max_depth                          # The maximum depth of the splitting tree
    cdef SIZE_t min_samples_split               # The minimum samples needed in a node of the splitting tree for splitting
    cdef double min_impurity_decrease           # The minimum value of impurity decrease for the splitting tree
    cdef double min_impurity_split              # The minimum value of impurity to split in the splitting tree
    cdef object mvar                            # The number of variables in the group we will use to create the splitting tree
    cdef int mgroup                             # The number of groups we will use to create all the splitting trees
    cdef object split_criterion                 # A string for type of criterion we want to use for the splitting trees ("gini","mse",...)

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

    # Initialise the CARTGVSplitter with the samples, responses and groups
    #
    # params X : an object (list, ndarray) the training samples
    # params y : a memoryview (ndarray), the responses of the training samples
    # params sample_weight : a DOUBLE_t*, the weight of each sample (can be None)
    # params groups : an object (list, ndarray), the groups of variables
    # params len_groups : a ndarray, the length of each group of variables
    cdef int init(self, object X, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, object groups, np.ndarray len_groups) except -1

    # Reset the parameters for node selection
    #
    # params start : a SIZE_t, the starting position in the samples array
    # params end : a SIZE_t, the ending position in the samples array
    # params weighted_n_node_samples : a double*
    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1

    # Get the value of the node into the dest array
    #
    # params dest : a double*, the array that will receive the values of the node
    cdef void node_value(self, double* dest) nogil

    # Get the impurity of the node, call the same function from the criterion
    cdef double node_impurity(self) nogil

    # Get the samples used in the current node between start and end in the samples array, only get the variable of the selected group
    #
    # params group : a memoryview (ndarray), the group of variable used
    # params len_group : an int, the length of the group
    # params start : an int, the starting position of the current node in the samples array
    # params end : an int, the ending position of the current node in the samples array
    cdef np.ndarray group_sample(self, int[:] group, int len_group, int start, int end)

    # Reset the Scikit-learn instances used for constructing the splitting trees
    #
    # params y : a ndarray, the responses
    # params group : an int, the group index used for the splitting tree
    # params len_group : an int, the length of the group
    cdef int reset_scikit_learn_instances(self, np.ndarray y, int group, int len_group)

    # Build the splitting tree
    #
    # params Xf : a ndarray, the samples of the current nodes (output of group_sample)
    # params y : a ndarray, the responses for Xf
    cdef int splitting_tree_construction(self, np.ndarray Xf, np.ndarray y)

#    cdef int get_splitting_tree_n_leaves(self)

    # Get the information of the leaves of the splitting tree
    #
    # params sorted_leaves : a Node**, an empty array of Node that will be filled with the leaves of the splitting tree
    cdef int get_splitting_tree_leaves(self, Node** sorted_leaves)

    # Get the samples array of the splitting tree ordered and the position in this array for each leaf
    #
    # params starts : a SIZE_t**, an empty array that will be filled with the starting position of the leaves
    # params ends : a SIZE_t **, an empty array that will be filled with the ending position of the leaves
    # params sorted_leaves : a Node*, an array containing the leaves of the splitting tree (output of get_splitting_tree_n_leaves)
    # params n_leaves : a SIZE_t, the number of leaves
    # params n_samples : a SIZE_t, the number of samples
    cdef int get_splitting_tree_leaves_pos(self, SIZE_t** starts, SIZE_t** ends, Node* sorted_leaves, SIZE_t n_leaves, SIZE_t n_samples)

    # Switch the best splitting tree for the current splitting tree if the current splitting tree is better
    #
    # params current_proxy_improvement : a double, the impurity improvement for the current splitting tree
    # params best_proxy_improvement : a double*, the impurity improvement of the best splitting tree
    # params best :  a CARTGVSplitRecord*, the structure that hold the informations about the best split
    # params starts : a SIZE_t*, The starting position in the samples array for each child
    # params ends : a SIZE_t*, the ending position in the samples array for each child
    # params n_leaves : a SIZE_t, the number of leaves of the splitting tree = number of children
    # params group : an int, the group used to create the current splitting tree
    # params sorted_obs : a SIZE_t*, an empty array that will be filled with the ordered samples array from the Scikit-learn Splitter
    cdef int switch_best_splitting_tree(self, double current_proxy_improvement, double* best_proxy_improvement, CARTGVSplitRecord* best, SIZE_t* starts, SIZE_t* ends, SIZE_t n_leaves, int group, SIZE_t* sorted_obs)

    # Call the methods above to split a node
    #
    # params impurity : a double, the impurity of the current node
    # params split : a CARTGVSplitRecord*, the structure that will hold the informations about the best split
    # params n_constant_features : a SIZE_t* (Not usefull)
    # params parent_start : an int, the starting position in the samples array of the parrent node
    # params parent_end : an int, the ending position in the samples array of the parrent node (Not usefull)
    cdef int node_split(self, double impurity, CARTGVSplitRecord* split, SIZE_t* n_constant_features, int parent_start, int parent_end)

    ########################################## TESTS #############################################

    cpdef int test_init(self, object X, DOUBLE_t[:, ::1] y,
                  np.ndarray sample_weight, object groups, np.ndarray len_groups)

    cpdef int test_node_reset(self, SIZE_t start, SIZE_t end, double weighted_n_node_samples)

    cpdef double test_node_value(self, double dest)

    cpdef double test_node_impurity(self)

    cpdef np.ndarray test_group_sample(self, int[:] group, int len_group, int start, int end)

    cpdef int test_reset_scikit_learn_instances(self, np.ndarray y, int group, int len_group)

    cpdef int test_splitting_tree_construction(self, np.ndarray Xf, np.ndarray y)

    cpdef int test_get_splitting_tree_leaves(self)

    cpdef int test_get_splitting_tree_leaves_samples_and_pos(self)

    cpdef int test_switch_best_splitting_tree(self)

    cpdef int test_node_split(self)

cdef class BaseDenseCARTGVSplitter(CARTGVSplitter):

    cdef SIZE_t n_total_samples
    cdef const DTYPE_t[:,:] X