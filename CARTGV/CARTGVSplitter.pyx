# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:54:51 2021

@author: Alphonse
"""
from CARTGVCriterion cimport CARTGVCriterion

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdlib cimport malloc, free, calloc
from libc.stdio cimport printf

import numpy as np
#import dill as dill
import pickle as pickle
import sys
cimport numpy as np
np.import_array()

from scipy.sparse import csc_matrix

from sklearn.tree._utils cimport log, rand_int, rand_uniform, RAND_R_MAX, safe_realloc
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import BestFirstTreeBuilder, DepthFirstTreeBuilder
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._criterion import Gini
from sklearn.utils.validation import check_random_state

#from _tree cimport Tree, TreeBuilder, Node
#from _splitter cimport Splitter

#from sklearn.tree._tree cimport Tree, TreeBuilder, Node
#from sklearn.tree._splitter cimport Splitter

#from ..scikit_learn.sklearn.tree._tree cimport Tree, TreeBuilder, Node
#from ..scikit_learn.sklearn.tree._splitter cimport Splitter

import importlib
tree = importlib.import_module("scikit-learn.sklearn.tree")

# from numpy import float32 as DTYPE
# from numpy import float64 as DOUBLE

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

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
cdef DTYPE_t EXTRACT_NNZ_SWITCH = 0.1

cdef inline void _init_split(CARTGVSplitRecord* self) nogil:
    self.impurity_childs = [0]
    self.starts = [0]
    self.ends = [0]
    self.improvement = -INFINITY
    self.splitting_tree
    self.n_childs = 0

cdef class CARTGVSplitter():

    """Abstract splitter class.
    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(self, CARTGVCriterion criterion, SIZE_t max_grouped_features, int n_groups,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.
        max_features : SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.
        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.
        min_weight_leaf : double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.
        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """

        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        # self.feature_values = NULL

        self.sample_weight = NULL

        self.max_grouped_features = max_grouped_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        
        self.groups = np.empty((n_groups,max_grouped_features),dtype=int)
        self.n_groups = n_groups
        self.len_groups = np.empty((n_groups),dtype=int)

    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)
        # free(self.constant_features)
        # free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, object groups) except -1:
        """Initialize the splitter
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j, k
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        # safe_realloc(&self.feature_values, n_samples)
        # safe_realloc(&self.constant_features, n_features)

        print(y.shape)

        self.y = y

        self.sample_weight = sample_weight

        self.X = X
        
        self.groups = groups
        cdef int n_groups = groups.shape[0]
        self.n_groups = n_groups
        
        for k in range(n_groups):
          self.len_groups[k] = len(groups[k])

        n_outputs =  y.shape[1]
        classes = []
        n_classes = []
        y_encoded = np.zeros((n_samples,1), dtype=int)
        for k in range(n_outputs):
          classes_k, y_encoded[:,k] = np.unique(y[:,k], return_inverse=True)
          classes.append(classes_k)
          n_classes.append(classes_k.shape[0])

        n_classes = np.array(n_classes, dtype=np.intp)

        self.splitting_tree = Tree(n_features,n_classes, n_outputs)

        max_features = max(1, int(np.sqrt(n_features))) #len(groups[0])
        max_leaf_nodes = X.shape[0]
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = (0.25 * n_samples)
        max_depth = 3
        min_impurity_decrease = 0.
        min_impurity_split = 0
        random_state = check_random_state(0)

        criterion = Gini(n_outputs,n_classes)
        splitter = BestSplitter(criterion,max_features,min_samples_leaf,min_weight_leaf,random_state)
        self.splitting_tree_builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                   min_samples_leaf,
                                   min_weight_leaf,
                                   max_depth,
                                   min_impurity_decrease,
                                   min_impurity_split)
        
        return 0


    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:
        """Reset splitter on node samples[start:end].
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=double pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(self, double impurity, CARTGVSplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end]
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef int[:,:] groups = self.groups
        cdef SIZE_t n_groups = self.n_groups
        cdef int[:] group
        cdef int[:] len_groups = self.len_groups
        cdef int len_group
        cdef SIZE_t feature

        # cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t n_visited_grouped_features = 0
        cdef SIZE_t max_grouped_features = self.max_grouped_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef CARTGVSplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY

        cdef SIZE_t f_j
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t k
        cdef SIZE_t l
        cdef SIZE_t m
        cdef SIZE_t n

        cdef SIZE_t* sorted_obs
        cdef SIZE_t* tmp_sorted_obs
        cdef int n_samples
        cdef int n_nodes
        cdef Node* sorted_leaves
        cdef int n_leaves
        cdef SIZE_t** samples_leaves
        cdef int* starts = [0]
        cdef int* ends = [0]
        cdef int previous_pos = 0
        with gil:
           Xf = np.empty((end-start,1)) #Obligatoire sinon erreur de compilation ...
           best_splitting_tree = None
        cdef bytes splitting_tree

        _init_split(&best)

        while (n_visited_grouped_features < max_grouped_features):

            n_visited_grouped_features += 1

            with gil:
              f_j = 0#np.random.randint(0,max_grouped_features)

            group = groups[f_j]
            len_group = len_groups[f_j]

            with gil:
               Xf = np.empty((end-start,len_group)) # Récupère la shape correcte des données

            # Take the observations columns of group f_j between indexes start and end
            for i in range(start,end):
              for l in range(len_group):
                with gil:
                  Xf[i][l] = self.X[samples[i],group[l]]
                                 
            # Evaluate all splits
            self.criterion.reset()

            with gil:
              print("Start splitting_tree build")
              y = np.array(self.y).reshape(Xf.shape[0],1)
              # Create the splitting tree
              self.splitting_tree_builder.build(self.splitting_tree, Xf, y)
            
            with gil:
                  print("End splitting_tree build")
            # Get the leaves and their number, the samples and their number
            with gil:
              n_leaves = self.splitting_tree.n_leaves
            n_nodes = self.splitting_tree.node_count
            sorted_leaves = <Node*> malloc((n_leaves)*sizeof(Node))

            samples_leaves = <SIZE_t**> malloc(n_leaves * sizeof(SIZE_t*))

            tmp_sorted_obs = self.splitting_tree_builder.splitter.samples
            n_samples = self.splitting_tree_builder.splitter.n_samples

            with gil:
                print("End memory allocation and variable initialization")
            n = 0
            for k in range(n_nodes):
              if self.splitting_tree.nodes[k].left_child == _TREE_LEAF and self.splitting_tree.nodes[k].right_child == _TREE_LEAF:
                with gil:
                    sorted_leaves[n].left_child = self.splitting_tree.nodes[k].left_child
                    sorted_leaves[n].right_child = self.splitting_tree.nodes[k].left_child
                    sorted_leaves[n].feature = self.splitting_tree.nodes[k].feature
                    sorted_leaves[n].threshold = self.splitting_tree.nodes[k].threshold
                    sorted_leaves[n].impurity = self.splitting_tree.nodes[k].impurity
                    sorted_leaves[n].n_node_samples = self.splitting_tree.nodes[k].n_node_samples
                    sorted_leaves[n].weighted_n_node_samples = self.splitting_tree.nodes[k].weighted_n_node_samples
                    n+=1

            with gil:
                print("End node recuperation")

            # Get the samples for each leaves and their start and end position in the samples array
            previous_pos = 0
            for j in range(n_leaves):
              if(previous_pos + sorted_leaves[j].n_node_samples < n_samples):
                samples_leaves[j] = <SIZE_t*> malloc(sorted_leaves[j].n_node_samples*sizeof(SIZE_t))
                for m in range(previous_pos, previous_pos + sorted_leaves[j].n_node_samples):
                  samples_leaves[j][m] = tmp_sorted_obs[m]

              starts[j] = previous_pos
              ends[j] = previous_pos + sorted_leaves[j].n_node_samples
              previous_pos = previous_pos + sorted_leaves[j].n_node_samples

            with gil:
                print("Start criterion update")
                print(starts[0])
                print(starts[1])
                print(starts[2])

            self.criterion.update(starts,ends, n_leaves)

            with gil:
                print("End criterion update")

            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

            with gil:
                print("End proxy impurity improvement")

            if current_proxy_improvement > best_proxy_improvement:
                best_proxy_improvement = current_proxy_improvement
                with gil:
                  print("Current best tree found")
                  best_splitting_tree = self.splitting_tree
                  splitting_tree = pickle.dumps(best_splitting_tree,0)
                  current.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree)*sizeof(unsigned char))
                  current.splitting_tree = splitting_tree
                current.starts = starts
                current.ends = ends
                current.n_childs = n_leaves
                best = current  # copy
                sorted_obs = tmp_sorted_obs

        # Update the samples array
        self.samples = sorted_obs

        self.criterion.reset()
        with gil:
          print(best.n_childs)
          bst_splitting_tree = best.splitting_tree
          best_splitting_tree = pickle.loads(bst_splitting_tree)
          self.criterion.update(best.starts,best.ends,best_splitting_tree.n_leaves)
        self.criterion.children_impurity(best.impurity_childs)
        best.improvement = self.criterion.impurity_improvement(impurity, best.impurity_childs)

        # Return values
        split[0] = best
        return 0


    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""
        with gil:
            print("Criterion node value start")
            self.criterion.node_value(dest)
            print("Criterion node value end")

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()
