from libc.stdlib cimport calloc
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
import importlib
#import faulthandler
import sys

np.import_array()

from numpy.math cimport INFINITY

from sklearn.tree._utils cimport sizet_ptr_to_ndarray, safe_realloc

# EPSILON is used in the Poisson criterion
cdef double EPSILON = 10 * np.finfo('double').eps

cdef class CARTGVCriterion():

    """Interface for impurity criteria.
    This object stores methods on how to calculate how good a split is using
    different metrics.
    """

    def __dealloc__(self):
        """Destructor."""
        for i in range(self.n_childs):
            free(self.sum_childs[i])
        free(self.sum_childs)
        free(self.sum_total)
        free(self.weighted_n_childs)

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
        if self.samples is not NULL:
            return np.asarray(<SIZE_t[:self.n_samples]>self.samples)
        else:
            return None

    @property
    def starts(self):
        if self.starts is not NULL:
            return np.asarray(<SIZE_t[:self.n_childs]>self.starts)
        else:
            return None

    @property
    def ends(self):
        if self.ends is not NULL:
            return np.asarray(<SIZE_t[:self.n_childs]>self.ends)
        else:
            return None

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
        if self.weighted_n_childs is not NULL:
            return np.asarray(<double[:self.n_childs]>self.weighted_n_childs)
        else:
            return None

    @property
    def impurity_childs(self):
        if self.impurity_childs is not NULL:
            return np.asarray(<double[:self.n_childs]>self.impurity_childs)
        else:
            return None

    @property
    def n_childs(self):
        return self.n_childs

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t n_samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        pass

    cdef int reset(self) nogil except -1:
        pass

    cdef int update(self, SIZE_t* starts, SIZE_t* ends,int n_childs) nogil except -1:
        pass

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double** impurity_childs) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        pass

    cdef double proxy_impurity_improvement(self) nogil:

        cdef double* impurity_childs = <double*> malloc(self.n_childs * sizeof(double))
        self.children_impurity(&impurity_childs)

        cdef double res = 0
        cdef int n_childs = self.n_childs
        for i in range(n_childs):
          res -= self.weighted_n_childs[i] * impurity_childs[i]

        free(impurity_childs)

        return res

    cdef double impurity_improvement(self, double impurity_parent, double* impurity_childs) nogil:

        cdef double res = 0
        cdef int n_childs = self.n_childs

        for i in range(n_childs):
          res -= self.weighted_n_childs[i] / self.weighted_n_node_samples * impurity_childs[i]

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent + res))

    ########################################## TESTS #############################################

    cpdef int test_init(self, const DOUBLE_t[:, ::1] y, np.ndarray sample_weight,
                  double weighted_n_samples, np.ndarray samples, SIZE_t start,
                  SIZE_t end):
        pass

    cpdef int test_reset(self):
        pass

    cpdef int test_update(self):
        pass

    cpdef void test_node_value(self):
        pass

    cpdef double test_node_impurity(self):
        pass

    cpdef void test_children_impurity(self):
        pass

    cpdef double test_proxy_impurity_improvement(self):
        return self.proxy_impurity_improvement()

    cpdef double test_impurity_improvement(self, double impurity_parent, np.ndarray impurity_childs):
        return self.impurity_improvement(impurity_parent,<double*>impurity_childs.data)

cdef class CARTGVClassificationCriterion(CARTGVCriterion):

    @property #TODO vérifier que c'est bien taille self.n_outputs
    def n_classes(self):
        if self.n_classes is not NULL:
            return np.asarray(<SIZE_t[:self.n_outputs]> self.n_classes)
        else:
            return None

    @property
    def sum_stride(self):
        return self.sum_stride

    @property
    def sum_total(self):
        if self.sum_total is not NULL:
            return np.asarray(<double[:self.n_outputs * self.sum_stride]>self.sum_total)
        else:
            return None

    def __cinit__(self, SIZE_t n_outputs, np.ndarray[SIZE_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """
#        faulthandler.enable()

        self.sample_weight = NULL

        self.samples = NULL
        self.starts = NULL
        self.ends = NULL
        self.impurity_childs = NULL
        self.n_childs = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_childs = NULL

        # Count labels for each output
        self.sum_total = NULL
        self.sum_childs = NULL
        self.n_classes = NULL

        safe_realloc(&self.n_classes, n_outputs)

        cdef SIZE_t k = 0
        cdef SIZE_t sum_stride = 0

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

    def __dealloc__(self):
        """Destructor."""

    def __reduce__(self):
        return (type(self),
                (self.n_outputs,
                  sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples,SIZE_t n_samples, SIZE_t start, SIZE_t end) nogil except -1:
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
        self.n_samples = n_samples
        self.starts = [start]
        self.ends = [end]
        self.n_childs = 0 #TODO ATTENTION
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

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = <SIZE_t> self.y[i, k]
                sum_total[k * self.sum_stride + c] += w

            self.weighted_n_node_samples += w

#        with gil:
#            print(np.asarray(<double[:self.n_outputs*self.sum_stride]>sum_total))


        self.reset()

        return 0

    cdef int reset(self) nogil except -1: #TODO Methode inutile

#        cdef double* sum_total = self.sum_total
        cdef double** sum_childs = self.sum_childs

        cdef SIZE_t* n_classes = self.n_classes
        cdef int n_childs = self.n_childs
        cdef SIZE_t k
        cdef SIZE_t i

        return 0

    cdef int update(self, SIZE_t* starts, SIZE_t* ends,int n_childs) nogil except -1:

        cdef double** sum_childs = self.sum_childs
#        cdef double* sum_total = self.sum_total

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t i,j,k,l,m,n,o
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t n_elements = self.n_outputs * self.sum_stride

        self.n_childs = n_childs
        sum_childs = <double**> malloc(n_childs*sizeof(double*))
        for m in range(n_childs):
            sum_childs[m] = <double*> calloc(n_elements,sizeof(double))
        self.weighted_n_childs = <double*> calloc(n_childs,sizeof(double))

        for j in range(n_childs):
            for k in range(starts[j],ends[j]):
                i = samples[k]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for l in range(self.n_outputs):
                    label_index = l * self.sum_stride +  <SIZE_t> self.y[i, l]
                    sum_childs[j][label_index] += w

                self.weighted_n_childs[j] += w


#        for n in range(self.n_outputs):
#            for c in range(n_classes[n]):
#                sum_total[c] += self.sum_stride

        self.sum_childs = sum_childs
#        self.sum_total = sum_total
        self.starts = starts
        self.ends = ends

        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double** impurity_childs) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        cdef double* sum_total = self.sum_total
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k
#        with gil:
#            print(n_classes[0])
#            print(np.asarray(<double[:self.n_outputs*self.sum_stride]>sum_total))
#            print(self.sum_stride)
        for k in range(self.n_outputs):
            memcpy(dest, sum_total, n_classes[k] * sizeof(double))
            dest += self.sum_stride
            sum_total += self.sum_stride

#            with gil:
#                print(self.sum_stride)
#                print(np.asarray(sum_total))

#            dest = <double*> calloc(n_classes[k],sizeof(double))
#            dest = sum_total
#            dest += self.sum_stride
#            for l in range(n_classes[k]):
#                sum_total[l] += self.sum_stride


    ########################################## TESTS #############################################

    cpdef int test_init(self, const DOUBLE_t[:, ::1] y, np.ndarray sample_weight,
                  double weighted_n_samples, np.ndarray samples, SIZE_t start,
                  SIZE_t end):

        cdef DOUBLE_t* sample_weight_ = NULL
        if sample_weight is not None:
            sample_weight_ = <DOUBLE_t*> sample_weight.data
        cdef SIZE_t* samples_ = <SIZE_t*> samples.data

        return self.init(y, sample_weight_, weighted_n_samples, samples_,len(samples), start, end)

    cpdef int test_reset(self):

        res = self.reset()

        return res

    cpdef int test_update(self):

        n_childs = 5

        cdef SIZE_t* starts = <SIZE_t*> malloc(n_childs*sizeof(SIZE_t))
        cdef SIZE_t* ends = <SIZE_t*> malloc(n_childs*sizeof(SIZE_t))

        starts[0] = 0
        starts[1] = 215
        starts[2] = 308
        starts[3] = 315
        starts[4] = 333

        ends[0] = 215
        ends[1] = 308
        ends[2] = 314
        ends[3] = 333
        ends[4] = 334

        res = self.update(starts, ends, n_childs)

        return res

    cpdef void test_node_value(self):
        n_childs = 5

        cdef SIZE_t* starts = <SIZE_t*> malloc(n_childs*sizeof(SIZE_t))
        cdef SIZE_t* ends = <SIZE_t*> malloc(n_childs*sizeof(SIZE_t))

        starts[0] = 0
        starts[1] = 215
        starts[2] = 308
        starts[3] = 315
        starts[4] = 333

        ends[0] = 215
        ends[1] = 308
        ends[2] = 314
        ends[3] = 333
        ends[4] = 334

        self.update(starts, ends, n_childs)

        cdef double* dest = <double*> malloc(self.n_classes[0]*sizeof(double))

        self.node_value(dest)

    cpdef double test_node_impurity(self):
        pass

    cpdef void test_children_impurity(self):
        pass

cdef class CARTGVGini(CARTGVClassificationCriterion):

    cdef double node_impurity(self) nogil:

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

#            for c in range(n_classes[k]):
#                sum_total[c] += self.sum_stride

        return gini / self.n_outputs

    cdef void children_impurity(self, double** impurity_childs) nogil:

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

        for k in range(self.n_outputs):
          for i in range(n_childs):
            sq_count_childs[i] = 0.0


          for j in range(n_childs):
            for c in range(n_classes[k]):
                count_k = sum_childs[j][c]
                sq_count_childs[j] += count_k * count_k

          for l in range(n_childs):
            gini_childs[l] += 1.0 - sq_count_childs[l] / (self.weighted_n_childs[l] * self.weighted_n_childs[l])
            for c in range(n_classes[k]):
                sum_childs[l] += self.sum_stride
          for m in range(n_childs):
              impurity_childs[0][m] = gini_childs[m] / self.n_outputs

        free(gini_childs)
        free(sq_count_childs)



    ########################################## TESTS #############################################

    cpdef double test_node_impurity(self):
        return self.node_impurity()

    cpdef void test_children_impurity(self):

        cdef double* impurity_childs = <double*> malloc(self.n_childs * sizeof(double))

        self.children_impurity(&impurity_childs)

cdef class CARTGVRegressionCriterion(CARTGVCriterion):

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        """Initialize attributes for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """
#        faulthandler.enable()

        self.sample_weight = NULL

        self.samples = NULL
        self.starts = NULL
        self.ends = NULL
        self.impurity_childs = NULL
        self.n_childs = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_childs = NULL

        self.sq_sum_total = 0.0

        # Count labels for each output
        self.sum_total = NULL
        self.sum_childs = NULL

        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_childs = <double**> calloc(n_outputs, sizeof(double))

        if (self.sum_total == NULL or
            self.sum_childs == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""

    def __reduce__(self):
        return (type(self),
                (self.n_outputs, self.n_samples),
                self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples, SIZE_t n_samples, SIZE_t start, SIZE_t end) nogil except -1:
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
        self.n_childs = 0 #TODO ATTENTION
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        self.sq_sum_total = 0.0
        self.sum_total = <double*> calloc(self.n_outputs, sizeof(double))

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik

        for p in range(start, end):
            i = samples[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0
            if sample_weight != NULL:
                w = sample_weight[i]

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        self.reset()

        return 0

    cdef int reset(self) nogil except -1: #TODO Methode inutile

        cdef double* sum_total = self.sum_total
        cdef double** sum_childs = self.sum_childs

        cdef int n_childs = self.n_childs
        cdef SIZE_t k
        cdef SIZE_t i

        return 0

    cdef int update(self, SIZE_t* starts, SIZE_t* ends,int n_childs) nogil except -1:

        cdef double** sum_childs = self.sum_childs
        cdef double* sum_total = self.sum_total

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t i,j,k,l,m,n,o
        cdef DOUBLE_t w = 1.0

        self.n_childs = n_childs
        sum_childs = <double**> malloc(n_childs*sizeof(double*))
        for m in range(n_childs):
            sum_childs[m] = <double*> calloc(self.n_outputs,sizeof(double))
        self.weighted_n_childs = <double*> calloc(n_childs,sizeof(double))

        for j in range(n_childs):
            for k in range(starts[j],ends[j]):
                i = samples[k]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for l in range(self.n_outputs):
                    sum_childs[j][l] += w * self.y[i, l]

                self.weighted_n_childs[j] += w

        self.sum_childs = sum_childs
        self.sum_total = sum_total
        self.starts = starts
        self.ends = ends

        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double** impurity_childs) nogil:
        pass

    #TODO Check ZeroDivisionError: float division
    cdef void node_value(self, double* dest) nogil:
        cdef SIZE_t k
#        with gil:
#            print(self.sum_total[0])
#            print(self.weighted_n_node_samples)
        for k in range(self.n_outputs):

            dest[k] = self.sum_total[k] / self.weighted_n_node_samples

cdef class CARTGVMSE(CARTGVRegressionCriterion):

    cdef double node_impurity(self) nogil:

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

    cdef double proxy_impurity_improvement(self) nogil:

        cdef SIZE_t k,j,i
        cdef double* proxy_impurity_childs = <double*> calloc(self.n_childs,sizeof(double))
        cdef double** sum_childs = self.sum_childs

        for j in range(self.n_childs):
            for k in range(self.n_outputs):
                proxy_impurity_childs[j] += sum_childs[j][k] * sum_childs[j][k]

        cdef double res = 0
        cdef int n_childs = self.n_childs
        for i in range(n_childs):
          res += proxy_impurity_childs[i] / self.weighted_n_childs[i]

        return res

    cdef void children_impurity(self, double** impurity_childs) nogil:

        cdef double** sum_childs = self.sum_childs
        cdef SIZE_t k
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t l
        cdef SIZE_t c
        cdef SIZE_t m
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik
        cdef double* sq_sum_childs = <double*> calloc(self.n_childs,sizeof(double))
        cdef int n_childs = self.n_childs
        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        for i in range(n_childs):
            for j in range(self.starts[i],self.ends[i]):
                k = samples[j]

                if sample_weight != NULL:
                    w = sample_weight[k]

                for l in range(self.n_outputs):
                    y_ik = self.y[k,l]
                    sq_sum_childs[i] += w * y_ik * y_ik

        for i in range(n_childs):
            impurity_childs[0][i] = sq_sum_childs[i] / self.weighted_n_childs[i]
        for i in range(n_childs):
            for j in range(self.n_outputs):
                impurity_childs[0][i] -= (sum_childs[i][j] / self.weighted_n_childs[i]) ** 2
        for i in range(n_childs):
            impurity_childs[0][i] /= self.n_outputs

        free(sq_sum_childs)