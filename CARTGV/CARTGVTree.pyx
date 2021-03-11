# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:02:22 2021

@author: Alphonse
"""

from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX

import numpy as np
import pickle
cimport numpy as np

from sklearn.tree._tree cimport Tree, TreeBuilder, Node
from sklearn.tree._splitter cimport Splitter, SplitRecord

from CARTGVSplitter cimport CARTGVSplitter

np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

from sklearn.tree._utils cimport Stack, StackRecord, PriorityHeap, PriorityHeapRecord, safe_realloc, sizet_ptr_to_ndarray



cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(np.ndarray arr, PyObject* obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

# Build the corresponding numpy dtype for Node.
# This works by casting `dummy` to an array of Node of length 1, which numpy
# can construct a `dtype`-object for. See https://stackoverflow.com/q/62448946
# for a more detailed explanation.
cdef Node dummy;
NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

cdef class CARTGVTree():
  
    def __cinit__(self, int n_grouped_features, np.ndarray[SIZE_t, ndim=1] n_classes, int n_outputs):
          """Constructor."""
          # Input/Output layout
          self.n_grouped_features = n_grouped_features
          self.n_outputs = n_outputs
          self.n_classes = NULL
          safe_realloc(&self.n_classes, n_outputs)
      
          self.max_n_classes = np.max(n_classes)
          self.value_stride = n_outputs * self.max_n_classes
      
          cdef SIZE_t k
          for k in range(n_outputs):
              self.n_classes[k] = n_classes[k]
      
          # Inner structures
          self.max_depth = 0
          self.node_count = 0
          self.capacity = 0
          self.value = NULL
          self.nodes = NULL
  
      
    def __reduce__(self):
          """Reduce re-implementation, for pickling."""
          return (CARTGVTree, (self.n_grouped_features,
                         sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                         self.n_outputs), self.__getstate__())

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)

        if (node_ndarray.dtype != NODE_DTYPE):
            # possible mismatch of big/little endian due to serialization
            # on a different architecture. Try swapping the byte order.  
            node_ndarray = node_ndarray.byteswap().newbyteorder()
            if (node_ndarray.dtype != NODE_DTYPE):
                raise ValueError('Did not recognise loaded array dytpe')

        if (node_ndarray.ndim != 1 or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(CARTGVNode))
        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))
    
    
    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        # safe_realloc(self.nodes, capacity)
        # safe_realloc(self.value, capacity * self.value_stride)

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0
    
      
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_leaf,
                          Tree splitting_tree, double impurity,
                          SIZE_t n_node_samples, int n_childs,
                          double weighted_n_node_samples) nogil except -1:
        """
        Add a node to the tree.
        The new node registers itself as the child of its parent.
        Returns (SIZE_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count
        cdef int i
        
        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX

        cdef CARTGVNode* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            self.nodes[parent].childs[self.nodes[parent].n_childs+1] = node_id
          
        if is_leaf:
            for i in range(node.n_childs):
              node.childs[i] = _TREE_LEAF
            node.splitting_tree = <char*>_TREE_UNDEFINED #TODO définir un _TREE_UNDEFINED en tant que char*

        else:
            # childs will be set later
            # with gil:
            #   node.splitting_tree = pickle.dumps(splitting_tree)
            node.n_childs = n_childs

        self.node_count += 1

        return node_id

    # cpdef np.ndarray predict(self, object X):
    #     """Predict target for X."""
    #     out = self._get_value_ndarray().take(self.apply(X), axis=0,
    #                                          mode='clip')
    #     if self.n_outputs == 1:
    #         out = out.reshape(X.shape[0], self.max_n_classes)
    #     return out

    # cpdef np.ndarray apply(self, object X):
    #     """Finds the terminal region (=leaf node) for each sample in X."""
    #     if issparse(X):
    #         return self._apply_sparse_csr(X)
    #     else:
    #         return self._apply_dense(X)

    # cdef inline np.ndarray _apply_dense(self, object X):
    #     """Finds the terminal region (=leaf node) for each sample in X."""

    #     # Check input
    #     if not isinstance(X, np.ndarray):
    #         raise ValueError("X should be in np.ndarray format, got %s"
    #                          % type(X))

    #     if X.dtype != DTYPE:
    #         raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

    #     # Extract input
    #     cdef const DTYPE_t[:, :] X_ndarray = X
    #     cdef SIZE_t n_samples = X.shape[0]

    #     # Initialize output
    #     cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
    #     cdef SIZE_t* out_ptr = <SIZE_t*> out.data

    #     # Initialize auxiliary data-structure
    #     cdef CARTGVNode* node = NULL
    #     cdef Node* splitting_tree_node = NULL
    #     cdef SIZE_t i = 0

    #     with nogil:
    #         for i in range(n_samples):
    #             node = self.nodes
    #             # While node not a leaf
    #             while node.childs[0] != _TREE_LEAF:
    #               # splitting_tree_node = node.splitting_tree._apply_dense(X)
    #               splitting_tree_node = node.splitting_tree.nodes
    #               while splitting_tree_node.left_child != _TREE_LEAF:
    #                   # ... and node.right_child != _TREE_LEAF:
    #                   if X_ndarray[i, splitting_tree_node.feature] <= splitting_tree_node.threshold:
    #                       splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.left_child]
    #                   else:
    #                       splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.right_child]
    #               node = splitting_tree_node
    #             out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

    #     return out

    # #TODO : A modifier pour correspondre a CARTGV
    # cdef inline np.ndarray _apply_sparse_csr(self, object X):
    #     """Finds the terminal region (=leaf node) for each sample in sparse X.
    #     """
    #     # Check input
    #     if not isinstance(X, csr_matrix):
    #         raise ValueError("X should be in csr_matrix format, got %s"
    #                          % type(X))

    #     if X.dtype != DTYPE:
    #         raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

    #     # Extract input
    #     cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
    #     cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
    #     cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

    #     cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
    #     cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
    #     cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

    #     cdef SIZE_t n_samples = X.shape[0]
    #     cdef SIZE_t n_features = X.shape[1]

    #     # Initialize output
    #     cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
    #                                                    dtype=np.intp)
    #     cdef SIZE_t* out_ptr = <SIZE_t*> out.data

    #     # Initialize auxiliary data-structure
    #     cdef DTYPE_t feature_value = 0.
    #     cdef CARTGVNode* node = NULL
    #     cdef Node* splitting_tree_node = NULL
    #     cdef DTYPE_t* X_sample = NULL
    #     cdef SIZE_t i = 0
    #     cdef INT32_t k = 0

    #     # feature_to_sample as a data structure records the last seen sample
    #     # for each feature; functionally, it is an efficient way to identify
    #     # which features are nonzero in the present sample.
    #     cdef SIZE_t* feature_to_sample = NULL

    #     safe_realloc(&X_sample, n_features)
    #     safe_realloc(&feature_to_sample, n_features)

    #     with nogil:
    #         memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

    #         for i in range(n_samples):
    #             node = self.nodes

    #             for k in range(X_indptr[i], X_indptr[i + 1]):
    #                 feature_to_sample[X_indices[k]] = i
    #                 X_sample[X_indices[k]] = X_data[k]

    #             # While node not a leaf
    #             while node.childs != _TREE_LEAF:
    #                 splitting_tree_node = node.splitting_tree.nodes
    #                 while splitting_tree_node.left_child != _TREE_LEAF: 
    #                   # ... and node.right_child != _TREE_LEAF:
    #                   if feature_to_sample[splitting_tree_node.feature] == i:
    #                       feature_value = X_sample[splitting_tree_node.feature]
  
    #                   else:
    #                       feature_value = 0.
  
    #                   if feature_value <= splitting_tree_node.threshold:
    #                       splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.left_child]
    #                   else:
    #                       splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.right_child]
    #                 node = splitting_tree_node
    #             out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

    #         # Free auxiliary arrays
    #         free(X_sample)
    #         free(feature_to_sample)

    #     return out

    # cpdef object decision_path(self, object X):
    #     """Finds the decision path (=node) for each sample in X."""
    #     if issparse(X):
    #         return self._decision_path_sparse_csr(X)
    #     else:
    #         return self._decision_path_dense(X)

    # cdef inline object _decision_path_dense(self, object X):
    #     """Finds the decision path (=node) for each sample in X."""

    #     # Check input
    #     if not isinstance(X, np.ndarray):
    #         raise ValueError("X should be in np.ndarray format, got %s"
    #                          % type(X))

    #     if X.dtype != DTYPE:
    #         raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

    #     # Extract input
    #     cdef const DTYPE_t[:, :] X_ndarray = X
    #     cdef SIZE_t n_samples = X.shape[0]

    #     # Initialize output
    #     cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
    #     cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

    #     cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
    #                                                (1 + self.max_depth),
    #                                                dtype=np.intp)
    #     cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

    #     # Initialize auxiliary data-structure
    #     cdef CARTGVNode* node = NULL
    #     cdef Node* splitting_tree_node = NULL
    #     cdef SIZE_t i = 0

    #     with nogil:
    #         for i in range(n_samples):
    #             node = self.nodes
    #             indptr_ptr[i + 1] = indptr_ptr[i]

    #             # Add all external nodes
    #             while node.childs != _TREE_LEAF:
    #                 # ... and node.right_child != _TREE_LEAF:
    #                 splitting_tree_node = node.splitting_tree.nodes
                    
    #                 indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
    #                 indptr_ptr[i + 1] += 1
                    
    #                 while splitting_tree_node.left_child != _TREE_LEAF:
    #                   if X_ndarray[i, splitting_tree_node.feature] <= splitting_tree_node.threshold:
    #                       splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.left_child]
    #                   else:
    #                       splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.right_child]
                
    #                 node = splitting_tree_node
                    
    #             # Add the leave node
    #             indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
    #             indptr_ptr[i + 1] += 1

    #     indices = indices[:indptr[n_samples]]
    #     cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
    #                                            dtype=np.intp)
    #     out = csr_matrix((data, indices, indptr),
    #                      shape=(n_samples, self.node_count))

    #     return out

    # #TODO : A modifier pour correspondre a CARTGV
    # cdef inline object _decision_path_sparse_csr(self, object X):
    #     """Finds the decision path (=node) for each sample in X."""

    #     # Check input
    #     if not isinstance(X, csr_matrix):
    #         raise ValueError("X should be in csr_matrix format, got %s"
    #                          % type(X))

    #     if X.dtype != DTYPE:
    #         raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

    #     # Extract input
    #     cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
    #     cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
    #     cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

    #     cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
    #     cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
    #     cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

    #     cdef SIZE_t n_samples = X.shape[0]
    #     cdef SIZE_t n_features = X.shape[1]

    #     # Initialize output
    #     cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
    #     cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

    #     cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
    #                                                (1 + self.max_depth),
    #                                                dtype=np.intp)
    #     cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

    #     # Initialize auxiliary data-structure
    #     cdef DTYPE_t feature_value = 0.
    #     cdef Node* node = NULL
    #     cdef DTYPE_t* X_sample = NULL
    #     cdef SIZE_t i = 0
    #     cdef INT32_t k = 0

    #     # feature_to_sample as a data structure records the last seen sample
    #     # for each feature; functionally, it is an efficient way to identify
    #     # which features are nonzero in the present sample.
    #     cdef SIZE_t* feature_to_sample = NULL

    #     safe_realloc(&X_sample, n_features)
    #     safe_realloc(&feature_to_sample, n_features)

    #     with nogil:
    #         memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

    #         for i in range(n_samples):
    #             node = self.nodes
    #             indptr_ptr[i + 1] = indptr_ptr[i]

    #             for k in range(X_indptr[i], X_indptr[i + 1]):
    #                 feature_to_sample[X_indices[k]] = i
    #                 X_sample[X_indices[k]] = X_data[k]

    #             # While node not a leaf
    #             while node.childs != _TREE_LEAF:
    #                 indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
    #                 indptr_ptr[i + 1] += 1
    #                 splitting_tree_node = node.splitting_tree.nodes
    #                 while splitting_tree_node.left_child != _TREE_LEAF:
    #                 # ... and node.right_child != _TREE_LEAF:

    #                   if feature_to_sample[splitting_tree_node.feature] == i:
    #                       feature_value = X_sample[splitting_tree_node.feature]
  
    #                   else:
    #                       feature_value = 0.
  
    #                   if feature_value <= splitting_tree_node.threshold:
    #                       splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.left_child]
    #                   else:
    #                     splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.right_child]
    #                 node = splitting_tree_node
                    
    #             # Add the leave node
    #             indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
    #             indptr_ptr[i + 1] += 1

    #         # Free auxiliary arrays
    #         free(X_sample)
    #         free(feature_to_sample)

    #     indices = indices[:indptr[n_samples]]
    #     cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
    #                                            dtype=np.intp)
    #     out = csr_matrix((data, indices, indptr),
    #                      shape=(n_samples, self.node_count))

    #     return out

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    #TODO : Finir de l'adapter pour CARTGV
    # cpdef compute_group_importances(self, penality_function, normalize=True):
    #     """Computes the importance of each group (aka grouped variables)."""
    #     cdef Node* left
    #     cdef Node* right
    #     cdef CARTGVNode* nodes = self.nodes
    #     cdef CARTGVNode* node = nodes
    #     cdef CARTGVNode* end_node = node + self.node_count
    #     cdef int n_childs
    #     cdef double childs_impurity_sum
    #     cdef int dj

    #     cdef double normalizer = 0.

    #     cdef np.ndarray[np.float64_t, ndim=1] importances
    #     importances = np.zeros((self.n_features,))
    #     cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

    #     with nogil:
    #         while node != end_node:
    #             if node.childs[0] != _TREE_LEAF:
    #                 n_childs = node.n_childs
    #                 childs_impurity_sum = 0.0
    #                 for i in range(n_childs):
    #                   child_impurity_sum += self.nodes[node.childs[i]].weighted_n_node_samples * self.nodes[node.childs[i]].impurity

    #                 dj = len(self.groups[node.group])
    #                 importance_data[node.group] += (penality_function(dj) * (
    #                     node.weighted_n_node_samples * node.impurity -
    #                     child_impurity_sum))
    #             node += 1

    #     importances /= nodes[0].weighted_n_node_samples

    #     if normalize:
    #         normalizer = np.sum(importances)

    #         if normalizer > 0.0:
    #             # Avoid dividing by zero (e.g., when root is pure)
    #             importances /= normalizer

    #     return importances

    # def compute_partial_dependence(self, DTYPE_t[:, ::1] X,
    #                                int[::1] target_features,
    #                                double[::1] out):
    #     """Partial dependence of the response on the ``target_feature`` set.
    #     For each sample in ``X`` a tree traversal is performed.
    #     Each traversal starts from the root with weight 1.0.
    #     At each non-leaf node that splits on a target feature, either
    #     the left child or the right child is visited based on the feature
    #     value of the current sample, and the weight is not modified.
    #     At each non-leaf node that splits on a complementary feature,
    #     both children are visited and the weight is multiplied by the fraction
    #     of training samples which went to each child.
    #     At each leaf, the value of the node is multiplied by the current
    #     weight (weights sum to 1 for all visited terminal nodes).
    #     Parameters
    #     ----------
    #     X : view on 2d ndarray, shape (n_samples, n_target_features)
    #         The grid points on which the partial dependence should be
    #         evaluated.
    #     target_features : view on 1d ndarray, shape (n_target_features)
    #         The set of target features for which the partial dependence
    #         should be evaluated.
    #     out : view on 1d ndarray, shape (n_samples)
    #         The value of the partial dependence function on each grid
    #         point.
    #     """
    #     cdef:
    #         double[::1] weight_stack = np.zeros(self.node_count,
    #                                             dtype=np.float64)
    #         SIZE_t[::1] node_idx_stack = np.zeros(self.node_count,
    #                                               dtype=np.intp)
    #         SIZE_t sample_idx
    #         SIZE_t feature_idx
    #         int stack_size
    #         double left_sample_frac
    #         double current_weight
    #         double total_weight  # used for sanity check only
    #         CARTGVNode *current_node  # use a pointer to avoid copying attributes
    #         Node* splitting_tree_node = NULL
    #         SIZE_t current_node_idx
    #         bint is_target_feature
    #         SIZE_t _TREE_LEAF = TREE_LEAF  # to avoid python interactions

    #     for sample_idx in range(X.shape[0]):
    #         # init stacks for current sample
    #         stack_size = 1
    #         node_idx_stack[0] = 0  # root node
    #         weight_stack[0] = 1  # all the samples are in the root node
    #         total_weight = 0

    #         while stack_size > 0:
    #             # pop the stack
    #             stack_size -= 1
    #             current_node_idx = node_idx_stack[stack_size]
    #             current_node = &self.nodes[current_node_idx]

    #             if current_node.childs == _TREE_LEAF:
    #                 # leaf node
    #                 out[sample_idx] += (weight_stack[stack_size] *
    #                                     self.value[current_node_idx])
    #                 total_weight += weight_stack[stack_size]
    #             else:
    #                 # non-leaf node

    #                 # determine if the split feature is a target feature
    #                 is_target_group = False
    #                 for group_idx in range(target_groups.shape[0]):
    #                     if target_groups[group_idx] == current_node.group: #TODO : Rajouter ce champ dans CARTGVNode ?
    #                         is_target_group = True
    #                         break

    #                 if is_target_group:
    #                     splitting_tree_node = current_node.splitting_tree.nodes
    #                     while splitting_tree_node != _TREE_LEAF:
    #                       if X_ndarray[i, splitting_tree_node.feature] <= splitting_tree_node.threshold:
    #                           splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.left_child]
    #                       else:
    #                           splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.right_child]
    #                     node_idx_stack[stack_size] = splitting_tree_node
    #                     stack_size += 1
    #                 else:
    #                     # In this case, we push both children onto the stack,
    #                     # and give a weight proportional to the number of
    #                     # samples going through each branch.
    #                     for i in range(len(current_node.childs)):
    #                       node_idx_stack[stack_size] = current_node.childs[i]
    #                       child_sample_frac = (self.nodes[current_node.childs[i]].weighted_n_node_samples /
    #                                            current_node.weighted_n_node_samples)
    #                       current_weight = weight_stack[stack_size]
    #                       weight_stack[stack_size] = current_weight * child_sample_frac
    #                       stack_size += 1

    #         # Sanity check. Should never happen.
    #         if not (0.999 < total_weight < 1.001):
    #             raise ValueError("Total weight should be 1.0 but was %.9f" %
    #                              total_weight)

cdef class CARTGVTreeBuilder():
    
    def __cinit__(self, CARTGVSplitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, SIZE_t mgroup, SIZE_t mvar, 
                  double min_impurity_decrease, double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.mgroup = mgroup
        self.mvar = mvar
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.splitting_tree_builder = TreeBuilder(splitter, min_samples_split, min_samples_leaf, min_weight_leaf,
                                                  max_depth, min_impurity_decrease, min_impurity_split)

    cpdef build(self, CARTGVTree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef CARTGVSplitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t mgroup = self.mgroup
        cdef SIZE_t mvar = self.mvar
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef CARTGVSplitRecord split
        cdef SIZE_t node_id

        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0
        cdef int n_childs
        cdef Tree splitting_tree

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        with nogil:
            # push root node onto stack
            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0) #TODO créer une nouvelle class Stack_Record sans is_left
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))

                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                with gil:
                  splitting_tree = pickle.loads(split.splitting_tree)
                node_id = tree._add_node(parent, is_leaf, splitting_tree, impurity, n_node_samples, n_childs,
                                         weighted_n_node_samples)

                if node_id == SIZE_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:
                    n_childs = split.n_childs
                    for i in range(n_childs):
                      with gil:
                        rc = stack.push(split.starts[i],split.ends[i],depth + 1, node_id,0,split.impurity_childs[i], n_constant_features)
                      if rc == -1:
                        break
                    if rc == -1:
                      break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()
            
    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, y, sample_weight

