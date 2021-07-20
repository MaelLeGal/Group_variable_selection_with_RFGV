# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:02:22 2021

@author: Alphonse
"""

from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free, realloc, malloc
from libc.math cimport fabs, sqrt, fmax, log
from libc.string cimport memcpy, strcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX

import numpy as np
import pickle
cimport numpy as np
import sys
import faulthandler
import matplotlib.pyplot as plt


#from sklearn.tree._tree cimport Tree, TreeBuilder, Node
#from sklearn.tree._splitter cimport Splitter, SplitRecord

#import importlib
#tree = importlib.import_module("scikit-learn.sklearn.tree")

from CARTGVSplitter cimport CARTGVSplitter
from CARTGVCriterion cimport CARTGVCriterion

np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from sklearn.base import is_classifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

from sklearn.tree._utils cimport PriorityHeap, PriorityHeapRecord, sizet_ptr_to_ndarray
from CARTGVutils cimport safe_realloc, StackRecord, Stack



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
cdef double EPSILON = 0.1#np.finfo('double').eps

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
#cdef CARTGVNode dummy;
#NODE_DTYPE = np.asarray(<CARTGVNode[:1]>(&dummy)).dtype
#child_type = np.asarray(<SIZE_t[:1]>(&dummy.childs)).dtype
#print(child_type)
#splitting_tree_type = np.asarray(<unsigned char[:1]>(&dummy.splitting_tree)).dtype
#print(splitting_tree_type)
NODE_DTYPE = np.dtype([('childs',np.ndarray),
                        ('parent','i8'),
                        ('splitting_tree',np.str_),
                        ('impurity','f8'),
                        ('n_node_samples','i8'),
                        ('weighted_n_node_samples','f8'),
                        ('group','i8'),
                        ('n_childs','i8'),
                        ('current_child','i8'),
                        ('start','i8'),
                        ('end','i8'),
                        ('depth','i8')])
#NODE_DTYPE = CARTGVNodeClass
ctypedef struct CARTGVNode_PyObject:
#  PyObject_HEAD;
  SIZE_t* childs;                        # The childs of the node
  SIZE_t parent;                         # The parent of the node
  unsigned char* splitting_tree;         # The serialized splitting tree of the node
  DOUBLE_t impurity;                     # The impurity of the node
  SIZE_t n_node_samples;                 # The number of samples in the node
  DOUBLE_t weighted_n_node_samples;      # The number of weighted samples in the node
  int group;                             # The group selected for the split of the node
  int n_childs;                          # The number of childs of the node
  int current_child;                     # The current number of children of the node
  int start;                             # The starting position in the samples array
  int end;                               # The ending position in the samples array
  int depth;

#cdef CARTGVNodeClass dummy;
#NODE_DTYPE = np.asarray(dummy).dtype

#NODE_DTYPE = PyArray_RegisterDataType(CARTGVNode_PyObject)

#print(NODE_DTYPE)
#dt = np.dtype([('childs', np.dtype('intp')),
#               ('parent', np.dtype('intp')),
#               ('splitting_tree','B'),
#               ('impurity', np.dtype('float64')),
#               ('n_node_samples', np.dtype('intp')),
#               ('weighted_n_node_samples',np.dtype('float64')),
#               ('group',np.dtype('intp')),
#               ('n_childs',np.dtype('intp'))])
#print(dt)

cdef class CARTGVNodeClass():

    cdef CARTGVNode *ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        free(self.ptr)
        self.ptr = NULL

    @property
    def childs(self):
        return np.asarray(<SIZE_t[:self.ptr.n_childs]> self.ptr.childs) if self.ptr is not NULL else None

    @property
    def splitting_tree(self):
        if self.ptr is not NULL:
            if(self.ptr.n_childs > 0 and self.ptr.splitting_tree != b''):
                return pickle.loads(self.ptr.splitting_tree)
            else:
                return None
        else:
            return None

    @property
    def parent(self):
        return self.ptr.parent if self.ptr is not NULL else None

    @property
    def impurity(self):
        return self.ptr.impurity if self.ptr is not NULL else None

    @property
    def n_node_samples(self):
        return self.ptr.n_node_samples if self.ptr is not NULL else None


    @property
    def weighted_n_node_samples(self):
        return self.ptr.weighted_n_node_samples if self.ptr is not NULL else None

    @property
    def group(self):
        return self.ptr.group if self.ptr is not NULL else None

    @property
    def n_childs(self):
        return self.ptr.n_childs if self.ptr is not NULL else None

    @property
    def current_child(self):
        return self.ptr.current_child if self.ptr is not NULL else None

    @property
    def start(self):
        return self.ptr.start if self.ptr is not NULL else None

    @property
    def end(self):
        return self.ptr.end if self.ptr is not NULL else None

    @property
    def depth(self):
        return self.ptr.depth if self.ptr is not NULL else None

    @staticmethod
    cdef CARTGVNodeClass from_ptr(CARTGVNode *ptr):

        # Call to __new__ bypasses __init__ constructor
        cdef CARTGVNodeClass wrapper = CARTGVNodeClass.__new__(CARTGVNodeClass)
        wrapper.ptr = ptr
        return wrapper


cdef class CARTGVTree():
    """
    Class CARTGVTree, represent the tree created with multiple group of variable.
    It uses uses the construction of CART tree as part of its construction.
    """

#    @property
#    def n_grouped_features(self):
#        return self.n_grouped_features
#
#    @property
#    def n_outputs(self):
#        return self.n_outputs
#
    @property
    def n_classes(self):
        """
        Return the number of classes
        """
        if self.n_classes != NULL:
            return np.asarray(<SIZE_t[:self.n_outputs]>self.n_classes)
        else:
            return None
#
#    @property
#    def max_n_classes(self):
#        return self.max_n_classes
#
    @property
    def value_stride(self):
        """
        Return the value stride
        """
        return self.value_stride

#    @property
#    def max_depth(self):
#        return self.max_depth
#
#    @property
#    def node_count(self):
#        return self.node_count

#    @property
#    def n_leaves(self):
#        return self.n_leaves
#
#    @property
#    def capacity(self):
#        return self.capacity
#
    @property
    def value(self):
        """
        Return the values of each node (The number of observation in each class for the current node)
        """
        return self._get_value_ndarray()[:self.node_count]
#        if self.value != NULL:
#            return np.asarray(<double[:self.capacity*self.value_stride]>self.value)
#        else:
#            return None

    @property
    def nodes(self):
        """
        Return an array of nodes
        """
        if self.nodes != NULL:
            arr = self._get_node_ndarray()
#            return None #TODO find a way to return the node array
#            return np.asarray(<CARTGVNode[:self.capacity]>self.nodes)
            return arr
        else:
            return None

    @property
    def nodes_splitting_trees(self):
        """
        Return the splitting tree of each node
        """
        if self.nodes != NULL:
            arr = np.ndarray(self.node_count,dtype=Tree)
            for i in range(self.node_count):
                if(self.nodes[i].n_childs > 0 and self.nodes[i].splitting_tree != b''):
                    arr[i] = pickle.loads(self.nodes[i].splitting_tree)
                else:
                    arr[i] = None
            return arr
        else:
            return None
    @property
    def nodes_childs(self):
        """
        Return the ids of children of each nodes
        """
        if self.nodes != NULL:
            arr = np.ndarray((self.node_count))
            for i in range(self.node_count):
                if(self.nodes[i].n_childs > 0):
                    np.insert(arr,i,np.asarray(<SIZE_t[:self.nodes[i].n_childs]> self.nodes[i].childs), axis=0)
                else:
                    np.insert(arr,i, None, axis=0) #arr[i] = None
            return arr
        else:
            return None

    @property
    def nodes_parent(self):
        """
        Return the id of the parent of each node
        """
        if self.nodes != NULL:
            arr = np.ndarray(self.node_count)
            for i in range(self.node_count):
                arr[i] = self.nodes[i].parent
            return arr
        else:
            return None

    @property
    def nodes_impurities(self):
        """
        Return the impurity of each node
        """
        if self.nodes != NULL:
            arr = np.ndarray(self.node_count)
            for i in range(self.node_count):
                arr[i] = self.nodes[i].impurity
            return arr
        else:
            return None

    @property
    def nodes_n_node_samples(self):
        """
        Return the number of samples/observations in each node
        """
        if self.nodes != NULL:
            arr = np.ndarray(self.node_count)
            for i in range(self.node_count):
                arr[i] = self.nodes[i].n_node_samples
            return arr
        else:
            return None

    @property
    def nodes_weighted_n_node_samples(self):
        """
        Return the number of weighted samples/observations in each node
        """
        if self.nodes != NULL:
            arr = np.ndarray(self.node_count)
            for i in range(self.node_count):
                arr[i] = self.nodes[i].weighted_n_node_samples
            return arr
        else:
            return None

    @property
    def nodes_group(self):
        """
        Return an array containing the groups that cut each nodes (value = -1 if the node wasnt't splitted)
        """
        if self.nodes != NULL:
            arr = np.ndarray(self.node_count)
            for i in range(self.node_count):
                arr[i] = self.nodes[i].group
            return arr
        else:
            return None

    @property
    def nodes_n_childs(self):
        """
        Return the number of children for each node in an array
        """
        if self.nodes != NULL:
            arr = np.ndarray(self.node_count)
            for i in range(self.node_count):
                arr[i] = self.nodes[i].n_childs
            return arr
        else:
            return None

    @property
    def nodes_depths(self):
        """
        Return the depth of each node
        """
        if self.nodes != NULL:
            arr = np.ndarray(self.node_count)
            for i in range(self.node_count):
                arr[i] = self.nodes[i].depth
            return arr
        else:
            return None

    @property
    def nodes_cart_idx(self):
        """
        Return the place of the node in the splitting tree
        """
        if self.nodes != NULL:
            arr = np.ndarray(self.node_count)
            for i in range(self.node_count):
                arr[i] = self.nodes_cart_idx[i]
            return arr
        else:
            return None

    def __cinit__(self, int n_groups, np.ndarray len_groups, object groups, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes, int n_outputs):
          """
          Constructor of the CARTGVTree class
          params n_groups : an int, the number of groups
          params len_groups : a numpy ndarray, An array containing the length of each group
          params groups : a numpy ndarray, A 2D array containing the index of each variable for each group (example : data = (V1,V2,V3), groups = [[V1,V2],[V1,V3],[V2]])
          params n_features : an int, the number of variables in the datas
          params n_classes : a numpy ndarray #TODO
          params n_outputs : an int, the number of outputs, the number of dimension of the responses
          """

          #Enable error tracking
          faulthandler.enable()

          # Input/Output layout
          self.n_groups = n_groups
          self.len_groups = len_groups
          self.groups = groups
          self.n_features = n_features
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
          self.n_leaves = 0
          self.capacity = 0
          self.value = NULL
          self.nodes = NULL
          self.nodes_cart_idx = NULL

    def __dealloc__(self):
        """
        Free the allocated memory used or managed by the object
        """
        for i in range(self.node_count):
            free(self.nodes[i].childs)
            if(self.nodes[i].splitting_tree != b''):
                free(self.nodes[i].splitting_tree)
        free(self.nodes)
        free(self.nodes_cart_idx)
        free(self.n_classes)
        free(self.value)
      
    def __reduce__(self):
          """Reduce re-implementation, for pickling."""
          return (CARTGVTree, (self.n_groups,
                         self.len_groups,
                         self.groups,
                         self.n_features,
                         sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                         self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count

        d["values"] = self._get_value_ndarray()
        d["groups"] = self.groups
        d["n_leaves"] = self.n_leaves
        d["len_groups"] = self.len_groups
        d["nodes_cart_idx"] = self._get_nodes_cart_idx_ndarray()

        print(d)

        d["nodes"] = self._get_node_ndarray()

#        print(d)
        return d

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

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)
        safe_realloc(&self.nodes_cart_idx, capacity)
#
#        self.nodes = <CARTGVNode*> realloc(self.nodes, capacity)
#        self.value = <double*> realloc(self.value, capacity * self.value_stride)


        # value memory is initialised to 0 to enable classifier argmax
        # if capacity > self.capacity:
        #     memset(<void*>(self.value + self.capacity * self.value_stride), 0,
        #            (capacity - self.capacity) * self.value_stride *
        #            sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity


        self.capacity = capacity
        return 0
    
      
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_leaf,
                          unsigned char* splitting_tree, double impurity,
                          SIZE_t n_node_samples, int n_childs,
                          double weighted_n_node_samples, int group, int start, int end, int depth, SIZE_t cart_idx) nogil except -1:
        """
        Add a node to the tree.
        The new node registers itself as the child of its parent.
        Returns (SIZE_t)(-1) on error.
        params parent : A SIZE_t, the parent of the node
        params is_leaf : A bint, a boolean true if the node is a leaf, false otherwise
        params splitting_tree : An unsigned char*, the serialized node splitting_tree
        params impurity : A double, the impurity of the node
        params n_node_samples : A SIZE_t, the number of samples in the node
        params group : An int, the group used to split the node
        params start : An int, the index at which we start in the samples array to take the correct samples/observations in the node
        params end : An int, the index at which we end in the samples array to take the correct samples/observations in the node
        params depth : An int, the depth of the node
        params cart_idx : A SIZE_t, the index of the node in the leaves of the splitting tree
        """

        cdef SIZE_t node_id = self.node_count #The current node id
        cdef int i

        #Check if the number of nodes is bigger than the capacity
        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX

        # Creation of the node and setting of it's field
        cdef CARTGVNode* node = &self.nodes[node_id]
#        node = CARTGVNode()
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        node.n_childs = n_childs
        node.parent = parent
        node.childs = <SIZE_t*> malloc(n_childs*sizeof(SIZE_t))
        node.current_child = 0
        node.start = start
        node.end = end
        node.depth = depth

        self.nodes_cart_idx[node_id] = cart_idx

        # Check if the parent is undefined. If it isn't give this node id as the child of this node parent.
        if parent != _TREE_UNDEFINED:
                self.nodes[parent].childs[self.nodes[parent].current_child] = node_id
                self.nodes[parent].current_child += 1

        # Check if the current node is a leaf, if it is, define it as a leaf with _TREE_LEAF and _TREE_UNDEFINED
        if is_leaf:
#            node.splitting_tree = <unsigned char*> malloc(sizeof(SIZE_t))
            node.splitting_tree = '' #splitting_tree #TODO is it necessary ? Empty string for no splitting tree ?
            node.group = -2

        # If it isn't a leaf, assign the splitting tree to the node.
        else:
            #childs will be set later
            with gil:
                node.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree)*sizeof(unsigned char))
                memcpy(node.splitting_tree, splitting_tree, sys.getsizeof(splitting_tree)*sizeof(unsigned char))
#                node.splitting_tree = splitting_tree
                node.group = group

        self.node_count += 1

        return node_id

    cpdef np.ndarray predict(self, object X):
        """
        Predict target for X.
        params X : an object, the data to predict
        """
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                              mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray apply(self, object X):
        """
        Finds the terminal region (=leaf node) for each sample in X.
        params X : an object, the data for which we want the terminal regions
        """
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)

    cdef inline np.ndarray _apply_dense(self, object X):
        """
        Finds the terminal region (=leaf node) for each sample in X.
        params X : an object, the data for which we want the terminal regions
        """

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                            % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef CARTGVNode* node = NULL
        cdef Node* splitting_tree_node = NULL
        cdef SIZE_t i = 0

        cdef CARTGVNode current_node
        X = np.array(X)
        prediction = np.array([], dtype=int)
        prob = []

        # Check if there's more than one sample to predict
        if(X.ndim > 1):
            nb_samples = X.shape[0]
        else:
            nb_samples = 1

        # Loop on the number of samples to predict
        for i in range(nb_samples):
            current_node = self.nodes[0]
            current_split_tree = None

#            leaf_idx = 0
            # Loop until we arrive at a leaf
            while current_node.n_childs != 0:

                # Loads the splitting tree
                current_split_tree = pickle.loads(current_node.splitting_tree)

                group = self.groups[current_node.group]
                len_group = self.len_groups[current_node.group]
                if X.ndim > 1:
                    Xf = np.empty((X.shape[0],len_group), dtype=np.float32)
                else:
                    Xf = np.empty(len_group, dtype=np.float32)

                for j in range(len_group):
                    if X.ndim > 1:
                        Xf[i,j] = X[i,group[j]]
                    else:
                        Xf[j] = X[group[j]]

                # Get the id of the splitting tree lead in which the data went
                if X.ndim > 1:
                    leaf_id = current_split_tree.apply(Xf[i].reshape((1,len(Xf[i]))))
                else:
                    leaf_id = current_split_tree.apply(Xf.reshape((1,len(Xf))))

                # Get the index of the child in the current node
                idx_cart_tree = np.take(np.asarray(<SIZE_t[:self.node_count]>self.nodes_cart_idx), np.asarray(<SIZE_t[:current_node.n_childs]>current_node.childs))

                # Get the index in the splitting tree of the leaf
                leaf_idx = np.where(idx_cart_tree == leaf_id[0])[0][0]

                out_ptr[i] = <SIZE_t>(&self.nodes[np.asarray(<SIZE_t[:current_node.n_childs]>current_node.childs)[leaf_idx]] - self.nodes)

                # Get the next node
                current_node = self.nodes[np.asarray(<SIZE_t[:current_node.n_childs]>current_node.childs)[leaf_idx]]


        return out

#        with nogil:
#            for i in range(n_samples):
#                node = self.nodes
#                # While node not a leaf
#                while node.childs[0] != _TREE_LEAF:
#                    # splitting_tree_node = node.splitting_tree._apply_dense(X)
#                    splitting_tree_node = node.splitting_tree.nodes
#                    while splitting_tree_node.left_child != _TREE_LEAF:
#                        # ... and node.right_child != _TREE_LEAF:
#                        if X_ndarray[i, splitting_tree_node.feature] <= splitting_tree_node.threshold:
#                            splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.left_child]
#                        else:
#                            splitting_tree_node = &node.splitting_tree.nodes[splitting_tree_node.right_child]
#                    node = splitting_tree_node #TODO changer cette ligne assigniation Node a CARTGVNode ici
#                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

#        return out

    # cdef inline np.ndarray _apply_sparse_csr(self, object X):
    #     """Finds the terminal region (=leaf node) for each sample in sparse X.
    #     """
    #     # Check input
    #     if not isinstance(X, csr_matrix):
    #         raise ValueError("X should be in csr_matrix format, got %s"
    #                           % type(X))

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
    #                                                     dtype=np.intp)
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
    #                 node = splitting_tree_node  #TODO changer cette ligne assigniation Node a CARTGVNode ici
    #             out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

    #         # Free auxiliary arrays
    #         free(X_sample)
    #         free(feature_to_sample)

    #     return out

#    cpdef np.ndarray sobol_indice(self, object X, int group_j, int[::1] in_bag_idx, int[::1] oob_idx):
#
#        return self.apply_sobol(X, group_j, in_bag_idx, oob_idx)
#
#
#    cdef np.ndarray apply_sobol(self, object X, int group_j, int[::1] in_bag_indx, int[::1] oob_indx):
#
#        cdef np.ndarray in_bag_idx = np.asarray(in_bag_indx)
#        cdef np.ndarray oob_idx = np.asarray(oob_indx)
#
#                # Check input
#        if not isinstance(X, np.ndarray):
#            raise ValueError("X should be in np.ndarray format, got %s"
#                            % type(X))
#
#        if X.dtype != DTYPE:
#            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
#
#        # Extract input
#        cdef const DTYPE_t[:, :] X_ndarray = X
#        cdef SIZE_t n_samples = X.shape[0]
#
#        # Initialize output
#        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
#        cdef SIZE_t* out_ptr = <SIZE_t*> out.data
#
#        # Initialize auxiliary data-structure
#        cdef CARTGVNode* node = NULL
#        cdef Node* splitting_tree_node = NULL
#        cdef SIZE_t i = 0
#
#        cdef CARTGVNode current_node
#        X = np.array(X)
#        prediction = np.array([], dtype=int)
#        prob = []
#
#        # Check if there's more than one sample to predict
#        if(X.ndim > 1):
#            nb_samples = X.shape[0]
#        else:
#            nb_samples = 1
#
##        cdef CARTGVNode* final_nodes = <CARTGVNode*> malloc(sizeof(CARTGVNode)) #Tableau à réalouer à chaque nouveau noeud final
##        cdef int final_node_counter = 0
##
##        cdef CARTGVNode* stack = <CARTGVNode*> malloc(sizeof(CARTGVNode))
##        cdef int counter = 0
#
#
#        pred = []
#        list_final_nodes = []
#        # Loop on the number of samples to predict
#        print("NB samples : "  + str(nb_samples))
#        for i in range(nb_samples):
##            stack[0] = self.nodes[0]
##            counter += 1
#            final_nodes = []
#            stack = []
#            stack.append(0)
#            while stack:
#                print(len(stack))
#                id = stack.pop()
#                print("id : " + str(id))
#                current_node = self.nodes[id]
#
##            while counter > 0:
##                current_node = stack[counter-1]
##                counter -= 1
#
#                if current_node.n_childs != 0:
#                    if current_node.group != group_j:
#
#                        # Loads the splitting tree
#                        current_split_tree = pickle.loads(current_node.splitting_tree)
#
#                        group = self.groups[current_node.group]
#                        len_group = self.len_groups[current_node.group]
#                        if X.ndim > 1:
#                            Xf = np.empty((X.shape[0],len_group), dtype=np.float32)
#                        else:
#                            Xf = np.empty(len_group, dtype=np.float32)
#
#                        for j in range(len_group):
#                            if X.ndim > 1:
#                                Xf[i,j] = X[i,group[j]]
#                            else:
#                                Xf[j] = X[group[j]]
#
#                        # Get the id of the splitting tree lead in which the data went
#                        if X.ndim > 1:
#                            leaf_id = current_split_tree.apply(Xf[i].reshape((1,len(Xf[i]))))
#                        else:
#                            leaf_id = current_split_tree.apply(Xf.reshape((1,len(Xf))))
#
#                        # Get the index of the child in the current node
#                        idx_cart_tree = np.take(np.asarray(<SIZE_t[:self.node_count]>self.nodes_cart_idx), np.asarray(<SIZE_t[:current_node.n_childs]>current_node.childs))
#
#                        # Get the index in the splitting tree of the leaf
#                        leaf_idx = np.where(idx_cart_tree == leaf_id[0])[0][0]
#
#                        out_ptr[i] = <SIZE_t>(&self.nodes[np.asarray(<SIZE_t[:current_node.n_childs]>current_node.childs)[leaf_idx]] - self.nodes)
#
#                        # Get the next node
##                        stack[counter] = self.nodes[np.asarray(<SIZE_t[:current_node.n_childs]>current_node.childs)[leaf_idx]]
##                        counter += 1
#                        stack.append(np.asarray(<SIZE_t[:current_node.n_childs]>current_node.childs)[leaf_idx])
#
#                    else:
##                        print(np.asarray(<SIZE_t[:current_node.n_childs]>current_node.childs))
##                        stack = <CARTGVNode*> realloc(stack, (counter+current_node.n_childs)*sizeof(CARTGVNode))
##                        for j in range(current_node.n_childs):
##                            stack[counter+j] = self.nodes[current_node.childs[j]]
##                        counter += current_node.n_childs
#
#                         for j in range(current_node.n_childs):
#                            stack.append(current_node.childs[j])
#                else:
##                    final_nodes = <CARTGVNode*> realloc(final_nodes, (final_node_counter+1)*sizeof(CARTGVNode))
##                    final_nodes[final_node_counter] = current_node
##                    final_node_counter += 1
#
#                    final_nodes.append(id)
#
#            print(final_nodes)
#            print(self._get_value_ndarray()[final_nodes])
#
#            list_final_nodes.append(final_nodes)
##            np.append(prediction,np.argmax(np.sum(self._get_value_ndarray()[final_nodes], axis=0)))
##            print(prediction)
#        print(np.asarray(in_bag_idx))
#
#        in_bag_collection = np.array(list_final_nodes)[in_bag_idx]
#        oob_collection = np.array(list_final_nodes)[oob_idx]
#
#        print(in_bag_collection)
#        print(oob_collection)
#
#        for k in range(len(oob_collection)):
#            shared_nodes = []
#            for l in range(len(in_bag_collection)):
##                shared_nodes = np.where(np.intersect1d(oob_collection[k], in_bag_collection[l])) #pas bon
#                shared_nodes = np.nonzero(np.in1d(oob_collection[k], in_bag_collection[l]))
##        shared_nodes = np.intersect1d(in_bag_collection,oob_collection)
#
#            print(shared_nodes)
#
#            if len(shared_nodes[0]) == 0:
#                idx = np.where(self.nodes[k].parent in in_bag_collection)
#            else:
#                idx = shared_nodes
##                parent_nodes = []
##                parent_nodes.append(self.nodes[k].parent)
##                final_nodes = parent_nodes
#
#            pred.append(np.argmax(np.sum(self._get_value_ndarray()[idx], axis=0)))
#
#        return np.array(pred)
#
##            current_node = self.nodes[0]
##            current_split_tree = None


    cpdef object decision_path(self, object X):
        """
        Finds the decision path (=node) for each sample in X.
        params X : an object, the data for which we want the decision path
        """
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)

    # cdef inline object _decision_path_dense(self, object X):
    #     """Finds the decision path (=node) for each sample in X."""

    #     # Check input
    #     if not isinstance(X, np.ndarray):
    #         raise ValueError("X should be in np.ndarray format, got %s"
    #                           % type(X))

    #     if X.dtype != DTYPE:
    #         raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

    #     # Extract input
    #     cdef const DTYPE_t[:, :] X_ndarray = X
    #     cdef SIZE_t n_samples = X.shape[0]

    #     # Initialize output
    #     cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
    #     cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

    #     cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
    #                                                 (1 + self.max_depth),
    #                                                 dtype=np.intp)
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
    #                                             dtype=np.intp)
    #     out = csr_matrix((data, indices, indptr),
    #                       shape=(n_samples, self.node_count))

    #     return out

    # #TODO : A modifier pour correspondre a CARTGV
    # cdef inline object _decision_path_sparse_csr(self, object X):
    #     """Finds the decision path (=node) for each sample in X."""

    #     # Check input
    #     if not isinstance(X, csr_matrix):
    #         raise ValueError("X should be in csr_matrix format, got %s"
    #                           % type(X))

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
    #                                                 (1 + self.max_depth),
    #                                                 dtype=np.intp)
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
    #                                             dtype=np.intp)
    #     out = csr_matrix((data, indices, indptr),
    #                       shape=(n_samples, self.node_count))

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
        strides[0] = sizeof(CARTGVNode)
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

    cdef np.ndarray _get_nodes_cart_idx_ndarray(self):
        """
        Return an array containing the indexes of the nodes in their respective parent splitting tree
        """
        return np.asarray(<SIZE_t[:self.node_count]>self.nodes_cart_idx)

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
    #                                 int[::1] target_features,
    #                                 double[::1] out):
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
    #                                             current_node.weighted_n_node_samples)
    #                       current_weight = weight_stack[stack_size]
    #                       weight_stack[stack_size] = current_weight * child_sample_frac
    #                       stack_size += 1

    #         # Sanity check. Should never happen.
    #         if not (0.999 < total_weight < 1.001):
    #             raise ValueError("Total weight should be 1.0 but was %.9f" %
    #                               total_weight)


    ########################################## TESTS #############################################

    cpdef void test_resize_CARTGVTree(self, capacity):

        self._resize(capacity)

    cpdef void test_add_node(self, CARTGVSplitter splitter, SIZE_t start, SIZE_t end):

        cdef SIZE_t parent = -2
        cdef CARTGVSplitRecord split
        cdef bint is_leaf = False
        cdef SIZE_t n_constant_features = 0
        cdef Tree splitting_tree
        cdef double weighted_n_node_samples
        cdef double min_impurity_decrease = 0.1
        cdef parent_start = start
        cdef parent_end = end

        n_node_samples = end - start
        splitter.node_reset(start, end, &weighted_n_node_samples)

        impurity = splitter.node_impurity()
        splitter.node_split(impurity, &split, &n_constant_features, parent_start, parent_end)

        is_leaf = (is_leaf or (split.improvement + EPSILON < min_impurity_decrease))

        splt_tree = split.splitting_tree
#        splitting_tree = pickle.loads(splt_tree)
#        for i in range(100000):
        self._add_node(parent, is_leaf, splt_tree, impurity, n_node_samples, split.n_childs, weighted_n_node_samples, split.group, start, end, 0, 0)

#        print(self.nodes[0].n_childs)
#        print(np.asarray(<SIZE_t[:self.nodes[0].n_childs]>self.nodes[0].childs))
#        print(self.nodes[0].parent)
#        print(self.nodes[0].splitting_tree)
#        print(self.nodes[0].impurity)
#        print(self.nodes[0].n_node_samples)
#        print(self.nodes[0].weighted_n_node_samples)
#        print(self.nodes[0].group)

cdef class CARTGVTreeBuilder():

    @property
    def splitter(self):
        """
        Return the splitter of the class
        """
        return self.splitter

    @property
    def min_samples_split(self):
        """
        Return the minimal number of samples in a node needed to split it
        """
        return self.min_samples_split

    @property
    def min_samples_leaf(self):
        """
        Return the minimal number under which the nodes are considered leaves
        """
        return self.min_samples_leaf

    @property
    def min_weight_leaf(self):
        """
        Return the minimal weigth in a node under which it is considered a leaf
        """
        return self.min_weight_leaf

    @property
    def max_depth(self):
        """
        Return the maximal depth at which the tree can grow
        """
        return self.max_depth

    @property
    def mgroup(self):
        """
        Return the number of group that will be selected to split each node
        """
        return self.mgroup

    @property
    def mvar(self):
        """
        Return the number of variables in the selected group that will be used to build the splitting tree
        """
        return self.mvar

    @property
    def min_impurity_decrease(self):
        """
        Return the minimal decrease in impurity needed above which the node isn't considerd a leaf
        """
        return self.min_impurity_decrease

    @property
    def min_impurity_split(self):
        """
        Return the minimal impurity under which the node is considered a leaf
        """
        return self.min_impurity_split

    def __cinit__(self, CARTGVSplitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object max_depth, double min_impurity_decrease, double min_impurity_split):
        """
        The constructor of the CARTGVTreeBuilder class
        params splitter : A CARTGVSplitter, the splitter instance that will split the nodes
        params min_samples_split : A SIZE_t, the minimal number of samples in a node needed to split it
        params min_samples_split : A SIZE_t, the minimal number under which the nodes are considered leaves
        params min_weight_leaf : A double, the minimal weigth in a node under which it is considered a leaf
        params max_depth : An object, the maximal depth at which the tree can grow
        params min_impurity_decrease : A double, the minimal decrease in impurity needed above which the node isn't considerd a leaf
        params min_impurity_split : A double the minimal impurity under which the node is considered a leaf
        """
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = (np.iinfo(np.int32).max if max_depth is None
                     else max_depth)
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
#        self.splitting_tree_builder = TreeBuilder(splitter, min_samples_split, min_samples_leaf, min_weight_leaf,
#                                                  max_depth, min_impurity_decrease, min_impurity_split)
        faulthandler.enable()

    cpdef void build(self, CARTGVTree tree, object X, np.ndarray y, object groups, np.ndarray len_groups,
                object pen, np.ndarray sample_weight=None):
        """
        Build a decision tree from the training set (X, y).
        params tree : An instance of CARTGVTree that will be fiiled by
        params X : An object, the datas
        params y : A numpy ndarray,  the responses
        params groups : An object, an array containing the groups
        params len_groups : A numpy ndarray, the length of each group
        params pen : An object (string, function with only one parameter), the penality function
        params sample_weight : A numpy ndarray, the weight  of each samples
        """
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

        # Resize the tree, it's field such as nodes
        tree._resize(init_capacity)

        # Parameters
        cdef CARTGVSplitter splitter = self.splitter                        # The splitter to create our tree
        cdef SIZE_t max_depth = self.max_depth                              # The maximum depth of our splitting tree
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf                # The minimum number of observation/sample in a leaf
        cdef double min_weight_leaf = self.min_weight_leaf                  # The minimum weight in a leaf
        cdef SIZE_t min_samples_split = self.min_samples_split              # The minimum number needed to split a node
        cdef double min_impurity_decrease = self.min_impurity_decrease      # The minimum decrease in impurity needed for a node
        cdef double min_impurity_split = self.min_impurity_split            # The minimum impurity decrease needed for a split
        cdef bint impurity_childs_bool = True

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, groups, len_groups)

        cdef SIZE_t start                                                   # The starting position in the sample array for the node
        cdef SIZE_t end                                                     # The end position in the sample array for the node
        cdef SIZE_t depth                                                   # The depth of the node
        cdef SIZE_t parent                                                  # The parent of the node
        cdef SIZE_t n_node_samples = splitter.n_samples                     # The number of samples/observations in the node
        cdef double weighted_n_samples = splitter.weighted_n_samples        # The weight of the samples in the node
        cdef double weighted_n_node_samples                                 # The number of weighted sample in the node
        cdef CARTGVSplitRecord split                                        # The structure that contains the split informations
        cdef SIZE_t node_id                                                 # The current node id

        cdef double impurity = INFINITY                                     # The current impurity
        cdef SIZE_t n_constant_features                                     # The number of constant features (not used)
        cdef bint is_leaf = False                                           # Boolean, True if the current node is a leaf, False otherwise
        cdef bint first = 1                                                 # Boolean, True if it is the first node added to the tree, false otherwise
        cdef SIZE_t max_depth_seen = -1                                     #
        cdef int rc = 0                                                     # Return code of the stack
        cdef int n_childs                                                   # The number of childs for the node
        cdef Tree splitting_tree                                            # The splitting tree of the node

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)                        # The stack that contains the records
        cdef StackRecord stack_record                                       # A record for the stack

        cdef SIZE_t cart_idx = -2
        cdef SIZE_t* leaves_ids

        with nogil:
            # push root node onto stack
            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0, -2)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            # Loop until the stack is empty (if the last leaf has been treated)
            while not stack.is_empty():
#                with gil:
#                    print("### LOOP " + str(tree.node_count) + " ####")
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features
                cart_idx = stack_record.cart_idx

                n_node_samples = end - start

                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf or
                           impurity <= min_impurity_split)
#                           or (n_node_samples == 2 and impurity == 0.5)) #TODO check si cela ne cause pas d'autre erreur

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))

                # If the node isn't a leaf, we call the splitter to split it
                if not is_leaf:
                    with gil:
                        splitter.node_split(impurity, &split, &n_constant_features, start, end) # TODO Make the function no gil
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18

                    is_leaf = (is_leaf or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                    if not is_leaf and split.n_childs == 1:
                        is_leaf = True
                        split.n_childs = 0
                        split.group = -1
                        with gil:
                            splitting_tree = None

                else:
                    with gil:
                        split.n_childs = 0
                        split.group = -1
                        splitting_tree = None

                # Get the penality function and compute the impurity
                with gil:
                    penality = 0
                    if pen == None:
                        penality = 1
                    elif pen == "root":
                        penality = 1.0/sqrt(len_groups[split.group])
                    elif pen == "size":
                        penality = 1.0/len_groups[split.group]
                    elif pen == "log":
                        penality = 1.0/fmax(log(len_groups[split.group]),1)
                    elif callable(pen):
                        penality = pen(len_groups[split.group])
                    else:
                        penality = 1

                    impurity = penality*impurity

                # Add the node to the tree
                node_id = tree._add_node(parent, is_leaf, split.splitting_tree, impurity, n_node_samples, split.n_childs,
                                        weighted_n_node_samples, split.group, start, end, depth, cart_idx)

                if node_id == SIZE_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)

                # If the current node isn't a leaf, we add it's childs to the stack to be treated

                if not is_leaf:
                    n_childs = split.n_childs
                    with gil:
                        for i in range(n_childs):
                            split_tree = split.splitting_tree
                            split_tree = pickle.loads(split_tree)
                            cart_idx = np.where(split_tree.feature == -2)[0][i]
                            rc = stack.push(split.starts[i],split.ends[i],depth + 1, node_id,0,split.impurity_childs[i], n_constant_features, cart_idx)
                            if rc == -1:
                                break

                        free(split.splitting_tree)
                        free(split.starts)
                        free(split.ends)
                        free(split.impurity_childs)

                        if rc == -1:
                          break

                else:
                    tree.n_leaves +=1

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
        """
        Check input dtype, layout and format
        params X : An object, the data
        params y : a numpy ndarray the responses
        params sample_weight : a numpy ndarray, the weight of each sample
        """
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

    ########################################## TESTS #############################################

    cpdef void test_build(self, CARTGVTree tree, object X, np.ndarray y, object groups, np.ndarray len_groups, object pen, np.ndarray sample_weight=None):

        self.build(tree, X, y, groups, len_groups, pen, sample_weight)
#        for i in range(tree.node_count):
#            print("################## Node ID : " + str(i) + " ######################")
#            print("Node splitting tree : ")
#            print(tree.nodes[i].splitting_tree)
#            print("Node Impurity : " + str(tree.nodes[i].impurity))
#            print("Node parent : " + str(tree.nodes[i].parent))
#            print("Node n node samples : " + str(tree.nodes[i].n_node_samples))
#            print("Node n childs : " + str(tree.nodes[i].n_childs))
#            if(tree.nodes[i].n_childs > 0):
#                print("Node Childs : " + str(np.asarray(<SIZE_t[:tree.nodes[i].n_childs]>tree.nodes[i].childs)))
#
#                print("Test load splitting tree")
#                print(tree.nodes[i].splitting_tree)
#                splitting_tree = pickle.loads(tree.nodes[i].splitting_tree)
#                print(splitting_tree)
