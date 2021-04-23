# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:23:33 2021

@author: Alphonse
"""

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

from libc.stdlib cimport calloc
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
import importlib
import faulthandler
import sys


np.import_array()

from numpy.math cimport INFINITY

# _temp = __import__('scikit-learn.sklearn.tree._utils', globals(), locals(), ['safe_realloc', 'sizet_ptr_to_ndarray'], 0)
# sizet_ptr_to_ndarray = _temp.sizet_ptr_to_ndarray
# safe_realloc = _temp.safe_realloc

from sklearn.tree._utils cimport sizet_ptr_to_ndarray, safe_realloc

# from sklearn.tree._criterion cimport Criterion

# EPSILON is used in the Poisson criterion
cdef double EPSILON = 10 * np.finfo('double').eps

cdef class CARTGVCriterion():
    """Interface for impurity criteria.
    This object stores methods on how to calculate how good a split is using
    different metrics.
    """

    def __dealloc__(self):
        """Destructor."""
        free(self.sum_total)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    @property
    def y(self):
        return self.y

    @property
    def sample_weight(self):
        if self.sample_weight is not NULL:
            return np.asarray(<DOUBLE_t[:self.n_samples]>self.sample_weight)
        else:
            return None

    @property
    def samples(self):
        return np.asarray(<SIZE_t[:self.n_samples]>self.samples)

    @property
    def starts(self):
        return np.asarray(<SIZE_t[:self.n_childs+1]>self.starts) #TODO vérifier que le +1 est nécessaire

    @property
    def ends(self):
        return np.asarray(<SIZE_t[:self.n_childs+1]>self.ends) #TODO vérifier que le +1 est nécessaire

    @property
    def n_outputs(self):
        return self.n_outputs

    @property
    def n_samples(self):
        return self.n_samples

    @property
    def n_node_samples(self):
        return self.n_node_samples

    @property
    def weighted_n_samples(self):
        return self.weighted_n_samples

    @property
    def weighted_n_node_samples(self):
        return self.weighted_n_node_samples

    @property
    def weighted_n_childs(self):
        return np.asarray(<double[:self.n_childs]>self.weighted_n_childs)

    @property
    def impurity_childs(self):
        return np.asarray(<double[:self.n_childs]>self.impurity_childs)

    @property
    def n_childs(self):
        return self.n_childs

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Placeholder for a method which will initialize the criterion.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : double
            The total weight of the samples being considered
        samples : array-like, dtype=SIZE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node
        """
        pass

    cpdef int test_init(self, const DOUBLE_t[:, ::1] y, np.ndarray sample_weight,
                  double weighted_n_samples, np.ndarray samples, SIZE_t start,
                  SIZE_t end):

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data
        cdef SIZE_t* samples_ptr = <SIZE_t*> samples.data

        return self.init(y, sample_weight_ptr, weighted_n_samples, samples_ptr, start, end)

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start.
        This method must be implemented by the subclass.
        """
        pass

    cpdef int test_reset(self):
        return self.reset()

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end.
        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, SIZE_t* starts, SIZE_t* ends, int n_childs) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.
        This updates the collected statistics by moving samples[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.
        Parameters
        ----------
        starts : SIZE_t*
            New starting index position of the samples for each child
        ends : SIZE_t*
            New ending index position of the samples for each child
        n_childs : int
            The number of childs
        """
        pass

    cdef double node_impurity(self) nogil:
        """Placeholder for calculating the impurity of the node.
        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of samples[start:end]. This is the
        primary function of the criterion class. The smaller the impurity the
        better.
        """
        pass

    cdef void children_impurity(self, double* impurity_childs) nogil:
        """Placeholder for calculating the impurity of children.
        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of samples[start:pos] + the impurity
        of samples[pos:end].
        Parameters
        ----------
        impurity_childs : double* pointer
            The memory address where the impurity of each child should be
            stored.
        """
        pass

    cdef void node_value(self, double* dest) nogil:
        """Placeholder for storing the node value.
        Placeholder for a method which will compute the node value
        of samples[start:end] and save the value into dest.
        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """
        pass

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef double* impurity_childs = <double*> malloc(self.n_childs * sizeof(double))
        self.children_impurity(impurity_childs)

        cdef double res = 0
        cdef int n_childs = self.n_childs
        for i in range(n_childs):
          res += -self.weighted_n_childs[i] * impurity_childs[i]

        return res

    cdef double impurity_improvement(self, double impurity_parent,double* impurity_childs) nogil:
        """Compute the improvement in impurity.
        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,
        Parameters
        ----------
        impurity_parent : double
            The initial impurity of the parent node before the split
        impurity_chimds : double pointer
            The impurity of each child
        Return
        ------
        double : improvement in impurity after the split occurs
        """
        cdef double res = 0
        cdef int n_childs = self.n_childs

        for i in range(n_childs):
          res -= self.weighted_n_childs[i] / self.weighted_n_node_samples * impurity_childs[i]

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent + res))
      
cdef class CARTGVClassificationCriterion(CARTGVCriterion):
    """Abstract criterion for classification."""

    def __cinit__(self, SIZE_t n_outputs,
                  np.ndarray[SIZE_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """
        self.sample_weight = NULL

        self.samples = NULL
        self.starts
        self.ends
        self.impurity_childs
        self.n_childs = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_childs

        # Count labels for each output
        self.sum_total = NULL
        self.sum_childs
        self.n_classes = NULL

        safe_realloc(&self.n_classes, n_outputs)

        cdef SIZE_t k = 0
        cdef SIZE_t sum_stride = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > sum_stride:
                sum_stride = n_classes[k]

        self.sum_stride = sum_stride

        cdef SIZE_t n_elements = n_outputs * sum_stride
        self.sum_total = <double*> calloc(n_elements, sizeof(double))
        self.sum_childs = <double**> calloc(n_elements, sizeof(double))

        if (self.sum_total == NULL or
            self.sum_childs == NULL):
            raise MemoryError()

        faulthandler.enable()

    def __dealloc__(self):
        """Destructor."""
        free(self.n_classes)

    def __reduce__(self):
        return (type(self),
                (self.n_outputs,
                  sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        """Initialize the criterion.
        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : double
            The total weight of all samples
        samples : array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        start : SIZE_t
            The first sample to use in the mask
        end : SIZE_t
            The last sample to use in the mask
        """
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.starts = [start]
        self.ends = [end]
#        with gil:
#            print(np.asarray(<SIZE_t[:1]> self.starts))
#            print(np.asarray(<SIZE_t[:1]> self.ends)
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        for k in range(self.n_outputs):
            memset(sum_total + offset, 0, n_classes[k] * sizeof(double))
            offset += self.sum_stride

        for p in range(start, end):
            i = samples[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0
            if sample_weight != NULL:
                w = sample_weight[i]

#            with gil:
#                print(len(self.y))
#                print(i)

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = <SIZE_t> self.y[i, k]
                sum_total[k * self.sum_stride + c] += w

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        cdef double* sum_total = self.sum_total
        cdef double** sum_childs = self.sum_childs

        cdef SIZE_t* n_classes = self.n_classes
        cdef int n_childs = self.n_childs
        cdef SIZE_t k
        cdef SIZE_t i

        self.weighted_n_childs = <double*> calloc(self.n_outputs,sizeof(double))

        for k in range(self.n_outputs):
          for i in range(n_childs):
            
            memset(sum_childs[i], 0, n_classes[k] * sizeof(double))

          sum_total += self.sum_stride
          sum_childs += self.sum_stride
          
        return 0


    cdef int update(self, SIZE_t* starts, SIZE_t* ends,int n_childs) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        starts : SIZE_t*
            The new starting position for which to move samples for each child
        ends : SIZE_t*
            The new ending position for which to move samples for each child
        n_childs : int
            The number of child
        """
        with gil:
            print("Criterion update")

        cdef double** sum_childs = self.sum_childs
        cdef double* sum_total = self.sum_total

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t n_elements = self.n_outputs * self.sum_stride

        self.n_childs = n_childs
        sum_childs = <double**> calloc(n_childs,sizeof(double*))
        self.weighted_n_childs = <double*> calloc(n_childs,sizeof(double))

        with gil:
            print(np.asarray(<double[:n_childs]> self.weighted_n_childs))

        # Loop on each child
        for j in range(n_childs):
          sum_childs[j] = <double*> calloc(n_elements,sizeof(double))

          # Loop between the start and end for the current child
          for p in range (starts[j],ends[j]):
            i = samples[p]

            if sample_weight != NULL:
              w = sample_weight[i]

            # increment the count of each label for each child
            for k in range(self.n_outputs):
              label_index = k * self.sum_stride +  <SIZE_t> self.y[i, k]
              sum_childs[j][label_index] += w

#            with gil:
#                print(np.asarray(<double[:n_childs*sizeof(double)]> self.weighted_n_childs))
            self.weighted_n_childs[j] += w

        for k in range(self.n_outputs):
            for j in range(n_childs):
                sum_childs[j][k] += self.sum_stride
            sum_total += self.sum_stride

        self.sum_childs = sum_childs
        self.sum_total = sum_total
        self.starts = starts
        self.ends = ends

        with gil:
            print(np.asarray(<double[:n_childs]> self.weighted_n_childs))

        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_childs) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] and save it into dest.
        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        cdef double* sum_total = self.sum_total
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memcpy(&dest, &sum_total, n_classes[k] * sizeof(double))
            dest += self.sum_stride
            sum_total += self.sum_stride

cdef class CARTGVGini(CARTGVClassificationCriterion):
    """Gini Index impurity criterion.
    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let
        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)
    be the proportion of class k observations in node m.
    The Gini Index is then defined as:
        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the Gini criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double gini = 0.0
        cdef double sq_count
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            sq_count = 0.0

            for c in range(n_classes[k]):
                count_k = sum_total[c]
                sq_count += count_k * count_k

            gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)

            sum_total += self.sum_stride

        return gini / self.n_outputs

    cdef void children_impurity(self, double* impurity_childs) nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]) using the Gini index.
        Parameters
        ----------
        impurity_childs : double pointer
            The array where the impurity of each child will be set
        """
        cdef SIZE_t* n_classes = self.n_classes
        cdef double** sum_childs = self.sum_childs
        cdef double* gini_childs = <double*> calloc(self.n_childs,sizeof(double))
        cdef double* sq_count_childs = <double*> calloc(self.n_childs,sizeof(double))
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t l
        cdef SIZE_t c
        cdef SIZE_t m
        cdef int n_childs = self.n_childs

        with gil:
            print(np.asarray(<double[:n_childs]> self.weighted_n_childs))

        for k in range(self.n_outputs):
          for i in range(n_childs):
            sq_count_childs[i] = 0.0

          for c in range(n_classes[k]):
              for j in range(n_childs):
                count_k = sum_childs[j][c]
                sq_count_childs[j] += count_k * count_k

          for l in range(n_childs):
            gini_childs[l] += 1.0 - sq_count_childs[l] / (self.weighted_n_childs[l] * self.weighted_n_childs[l])
            sum_childs[l] += self.sum_stride

        for m in range(n_childs):
          impurity_childs[m] = gini_childs[m] / self.n_outputs

        self.sum_childs = sum_childs
