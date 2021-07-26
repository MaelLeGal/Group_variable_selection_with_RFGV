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
#import faulthandler
import matplotlib.pyplot as plt

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

    @property
    def n_classes(self):
        """
        Return the number of classes
        """
        if self.n_classes != NULL:
            return np.asarray(<SIZE_t[:self.n_outputs]>self.n_classes)
        else:
            return None

    @property
    def value_stride(self):
        """
        Return the value stride
        """
        return self.value_stride

    @property
    def value(self):
        """
        Return the values of each node (The number of observation in each class for the current node)
        """
        return self._get_value_ndarray()[:self.node_count]

    @property
    def nodes(self):
        """
        Return an array of nodes
        """
        if self.nodes != NULL:
            arr = self._get_node_ndarray()
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

          outputs : An instance of CARTGVTree
          """

          #Enable error tracking
#          faulthandler.enable()

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

        outputs : An int, the id of the node
        """

        cdef SIZE_t node_id = self.node_count #The current node id
        cdef int i

        #Check if the number of nodes is bigger than the capacity
        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX

        # Creation of the node and setting of it's field
        cdef CARTGVNode* node = &self.nodes[node_id]
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
            node.splitting_tree = ''
            node.group = -2

        # If it isn't a leaf, assign the splitting tree to the node.
        else:
            #childs will be set later
            with gil:
                node.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree)*sizeof(unsigned char))
                memcpy(node.splitting_tree, splitting_tree, sys.getsizeof(splitting_tree)*sizeof(unsigned char))
                node.group = group

        self.node_count += 1

        return node_id

    cpdef np.ndarray predict(self, object X):
        """
        Predict target for X.
        params X : an object, the data to predict

        outputs : An array, the predictions of the tree
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

        outputs : An array, the terminal region (leaf node) of the data to predict.
        """
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)

    cdef inline np.ndarray _apply_dense(self, object X):
        """
        Finds the terminal region (=leaf node) for each sample in X.
        params X : an object, the data for which we want the terminal regions

        outputs : An array, the terminal region (leaf node) of the data to predict.
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

        outputs : An instance of CARTGVTreeBuilder
        """
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = (np.iinfo(np.int32).max if max_depth is None
                     else max_depth)
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

#        faulthandler.enable()

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

        outputs : A tuples containing the data, the responses and the sample weight
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
