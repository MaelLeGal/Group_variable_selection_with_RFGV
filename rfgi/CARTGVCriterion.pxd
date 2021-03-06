import numpy as np
cimport numpy as np

from sklearn.tree._tree cimport DTYPE_t
from sklearn.tree._tree cimport DOUBLE_t
from sklearn.tree._tree cimport SIZE_t
from sklearn.tree._tree cimport INT32_t
from sklearn.tree._tree cimport UINT32_t

cdef class CARTGVCriterion():

    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics
    # such as the mean in regression and class probabilities in classification.

    # Internal structures
    cdef const DOUBLE_t[:, ::1] y        # Values of y
    cdef DOUBLE_t* sample_weight         # Sample weights

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t* starts                  # The starts of each child in the samples array
    cdef SIZE_t* ends                    # The ends of each child in the samples array

    cdef SIZE_t n_outputs                # Number of outputs
    cdef SIZE_t n_samples                # Number of samples
    cdef SIZE_t n_node_samples           # Number of samples in the node (end-start)
    cdef double weighted_n_samples       # Weighted number of samples (in total)
    cdef double weighted_n_node_samples  # Weighted number of samples in the node
    cdef double* weighted_n_childs       # Weighted number of samples in the childs
    cdef double* impurity_childs         # The impurity of each child
    cdef int n_childs                    # The number of childs

    cdef double* sum_total          # For classification criteria, the sum of the
                                    # weighted count of each label. For regression,
                                    # the sum of w*y. sum_total[k] is equal to
                                    # sum_{i=start}^{end-1} w[samples[i]]*y[samples[i], k],
                                    # where k is output index.
    cdef double** sum_childs        # The sum of the count of each label for each child

    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods

    # Initialise the CARTGVCriterion
    #
    # params y : a memoryview (ndarray), the responses of current sample
    # params sample_weight : a DOUBLE_t*, the weight of each sample
    # params weighted_n_samples : a double, the total weight of all samples
    # params samples : a SIZE_t*, the indices array of the samples
    # params n_samples : a SIZE_t, the number of samples
    # params start : a SIZE_t, the starting position in the samples array
    # params end : a SIZE_t, the ending position in the samples array
    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t n_samples , SIZE_t start,
                  SIZE_t end) nogil except -1

    # Reset the criterion
    cdef int reset(self) nogil except -1

    # Update the criterion with the informations about the children of the current node
    #
    # params starts : a SIZE_t*, the starting positions of the children of the current node
    # params ends : a SIZE_t*, the ending positions of the children of the current node
    # params n_childs : an int, the number of children
    cdef int update(self, SIZE_t* starts, SIZE_t* ends,int n_childs) nogil except -1

    # Compute the impurity of the node
    cdef double node_impurity(self) nogil

    # Compute the impurity of the children
    #
    # params impurity_childs : a double**, an empty array for the children impurity
    cdef void children_impurity(self, double** impurity_childs) nogil

    # Get the node value
    #
    # params dest : a double*, a double where the value will be put
    cdef void node_value(self, double* dest) nogil

    # Compute the proxy impurity improvement
    cdef double proxy_impurity_improvement(self) nogil

    # Compute the real impurity improvement
    #
    # params impurity_parent : a double, the impurity of the parent
    # params impurity_childs : a double*, the impurities of the children
    cdef double impurity_improvement(self, double impurity_parent,double* impurity_childs) nogil

cdef class CARTGVClassificationCriterion(CARTGVCriterion):
    """Abstract criterion for classification."""

    cdef SIZE_t* n_classes
    cdef SIZE_t sum_stride

cdef class CARTGVRegressionCriterion(CARTGVCriterion):
    """Abstract criterion for regression."""

    cdef double sq_sum_total
