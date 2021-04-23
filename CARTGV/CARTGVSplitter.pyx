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
import random
import faulthandler
cimport numpy as np
np.import_array()

from scipy.sparse import csc_matrix

from sklearn.tree._utils cimport log, rand_int, rand_uniform, RAND_R_MAX, safe_realloc
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import BestFirstTreeBuilder, DepthFirstTreeBuilder
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._criterion import Gini
from sklearn.utils.validation import check_random_state
from sklearn.tree._tree import DOUBLE

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
    """
    Initialize a CARTGVSplitRecord
    splitting tree will be initialized later as we dont know its size yet
    """
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
    @property
    def X(self):
        return self.X

    @property
    def y(self):
        return self.y

    @property
    def criterion(self):
        return self.criterion

    @property
    def splitting_tree_builder(self):
        return self.splitting_tree_builder

    @splitting_tree_builder.setter
    def splitting_tree_builder(self,splitting_tree_builder_): #TODO retirer les setters une fois qu'ils ne sont plus utiles pour les tests
        self.splitting_tree_builder = <TreeBuilder>splitting_tree_builder_

    @property
    def splitting_tree(self):
        return self.splitting_tree

    @splitting_tree.setter
    def splitting_tree(self, splitting_tree_):
        self.splitting_tree = <Tree>splitting_tree_

    @property
    def groups(self):
        return self.groups

    @property
    def n_groups(self):
        return self.n_groups

    @property
    def len_groups(self):
        return self.len_groups

    @property
    def start(self):
        return self.start

    @property
    def end(self):
        return self.end

    @property
    def n_features(self):
        return self.n_features

    @property
    def weighted_n_samples(self):
        return self.weighted_n_samples

    @property
    def n_samples(self):
        return self.n_samples

    @property
    def samples(self):
        return np.asarray(<SIZE_t[:self.n_samples]>self.samples)

    @property
    def rand_r_state(self):
        return self.rand_r_state

    @property
    def random_state(self):
        return self.random_state

    @property
    def n_classes(self):
        return self.n_classes


    def __cinit__(self, CARTGVCriterion criterion, SIZE_t max_grouped_features, int n_groups,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        """
        Parameters
        ----------
        criterion : CARTGVCriterion
            The criterion to measure the quality of a split.
        max_grouped_features : SIZE_t
            The maximal number of randomly selected group of features which can be
            considered for a split.
        n_groups : int
            The number of groups
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
        faulthandler.enable()

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
        cdef SIZE_t n_samples
        cdef SIZE_t* samples
        cdef double weighted_n_samples = 0.0

        cdef SIZE_t n_features
        cdef SIZE_t* features

        cdef SIZE_t i, j, k

        self.rand_r_state = 2547
#        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX) #
        n_samples,n_features = X.shape

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        samples = safe_realloc(&self.samples, n_samples)

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

        features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

#         safe_realloc(&self.feature_values, n_samples)
#         safe_realloc(&self.constant_features, n_features)

        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y,(-1,1))

        y = np.asarray(y)

        self.sample_weight = sample_weight

        self.X = X

        self.groups = groups
        cdef int n_groups = groups.shape[0]
        self.n_groups = n_groups

        for k in range(n_groups):
          self.len_groups[k] = <int>len(groups[k])

        n_outputs =  y.shape[1]
        classes = []
        n_classes = []
        print(np.asarray(y).shape)
#        y_encoded = np.zeros(y.shape, dtype=int)
#        y_encoded = np.asarray(y_encoded)
#        print(np.asarray(y_encoded).shape)
        for k in range(n_outputs):
          classes_k = np.unique(y[:,k]) #, y_encoded[:,k] #, return_inverse=True
          classes.append(classes_k)
          n_classes.append(classes_k.shape[0])
#        y = y_encoded

        n_classes = np.array(n_classes, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y,dtype=DOUBLE)

        # Reshape the responses to correpsond to the format expected
        self.y = np.asarray(y) #.reshape(y.shape[0],y.shape[1])

        # Create the splitting tree
        self.splitting_tree = Tree(5,n_classes, n_outputs) #self.n_features

        # TODO make those variable settable parameters
        max_features = len(groups[0]) #max(1, int(np.sqrt(n_features))) #len(groups[0])
        max_leaf_nodes = -1 #X.shape[0]
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = 0.0 #(0.25 * n_samples)
        max_depth = 3
        min_impurity_decrease = 0.
        min_impurity_split = 0.
        random_state = check_random_state(2547)

        # Create the Criterion, Splitter et TreeBuilder for the splitting tree
        print("Init criterion start")
        print(n_outputs)
        print(n_classes.shape)

        criterion = Gini(n_outputs,n_classes)

        print("Init criterion end")
        print("Init splitter start")
        print(max_features)
        print(min_samples_leaf)
        print(min_weight_leaf)

        splitter = BestSplitter(criterion,max_features,min_samples_leaf,min_weight_leaf,random_state=random_state)

        print("Init splitter end")
        print("Init splitting tree builder start")

        print(min_samples_split)
        print(min_samples_leaf)
        print(min_weight_leaf)
        print(max_depth)
        print(min_impurity_decrease)
        print(min_impurity_split)

        self.splitting_tree_builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                   min_samples_leaf,
                                   min_weight_leaf,
                                   max_depth,
                                   min_impurity_decrease,
                                   min_impurity_split)

        print("Init splitting tree builder end")

        return 0

    cpdef int test_init(self,
                  object X,
                  DOUBLE_t[:, ::1] y,
                  np.ndarray sample_weight, object groups):
        res = -1
        if(sample_weight == None):
            res = self.init(X,y,NULL,groups)
        else:
            res = self.init(X,y,<DOUBLE_t*>sample_weight.data, groups)
        return res

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

    cpdef int test_node_reset(self, SIZE_t start, SIZE_t end,
                                double weighted_n_node_samples):
        res = self.node_reset(start,end,&weighted_n_node_samples) #<double*>&wns.data
        return res

    cpdef np.ndarray group_sample(self, int[:] group, int len_group, int start, int end):
        print("Group sample")
        cdef SIZE_t i
        cdef SIZE_t j

        Xf = np.empty((end-start,len_group)) # Récupère la shape correcte des données

        for i in range(start,end):
              for j in range(len_group):
                Xf[i][j] = self.X[self.samples[i],group[j]]

        return Xf

    cpdef int splitting_tree_construction(self, np.ndarray Xf, np.ndarray y):
        print("Splitting tree construction")
        print(type(Xf))
        print(type(y))
#        print(Xf)
#        print(y)
        print(np.array(Xf).shape)
        print(np.array(y).shape)
        print(self.splitting_tree)
#        print(self.splitting_tree_builder.min_samples_split)      # Minimum number of samples in an internal node
#        print(self.splitting_tree_builder.min_samples_leaf)        # Minimum number of samples in a leaf
#        print(self.splitting_tree_builder.min_weight_leaf)         # Minimum weight in a leaf
#        print(self.splitting_tree_builder.max_depth)               # Maximal tree depth
#        print(self.splitting_tree_builder.min_impurity_split)
#        print(self.splitting_tree_builder.min_impurity_decrease)
#        print(self.splitting_tree_builder.splitter)

        print(self.splitting_tree_builder.splitter.start)
        print(self.splitting_tree_builder.splitter.end)

        self.splitting_tree_builder.build(self.splitting_tree, Xf, y) # TODO Rajouter une exception en cas d'erreur

        print(self.splitting_tree_builder.splitter.start)
        print(self.splitting_tree_builder.splitter.end)

        print(self.splitting_tree_builder.splitter.max_features)
        print(self.splitting_tree_builder.splitter.min_samples_leaf)
        print(self.splitting_tree_builder.splitter.min_weight_leaf)
        print(np.asarray(<SIZE_t[:self.splitting_tree_builder.splitter.n_samples]>self.splitting_tree_builder.splitter.samples))
        print(self.splitting_tree_builder.splitter.n_samples)
        print(self.splitting_tree_builder.splitter.weighted_n_samples)
        print(np.asarray(<SIZE_t[:self.splitting_tree_builder.splitter.n_features]>self.splitting_tree_builder.splitter.features))
#        print(self.splitting_tree_builder.splitter.constant_features)
        print(self.splitting_tree_builder.splitter.n_features)
#        print(self.splitting_tree_builder.splitter.feature_values)
        print(type(self.splitting_tree_builder.splitter.y))
#        print(np.asarray(self.splitting_tree_builder.splitter.y)) #TODO erreur avec y ??
#        print(self.splitting_tree_builder.splitter.sample_weight)

        print("######## Splitting tree ########")
        print(self.splitting_tree)
        print(self.splitting_tree.node_count)
        for i in range(self.splitting_tree.node_count):
            print(self.splitting_tree.nodes[i].n_node_samples)

        return 0

    cdef int get_splitting_tree_n_leaves(self):
        print("Get splitting tree n leaves")
        n_leaves = 0
        for i in range(len(self.splitting_tree.children_left)):
            if(self.splitting_tree.children_left[i] == -1 and self.splitting_tree.children_right[i] == -1):
                n_leaves += 1

        return n_leaves

    cdef int get_splitting_tree_leaves(self, Node** sorted_leaves):
        print("Get splitting tree leaves")
        cdef SIZE_t n_nodes = self.splitting_tree.node_count

        n = 0
        for k in range(n_nodes):
          if self.splitting_tree.nodes[k].left_child == _TREE_LEAF and self.splitting_tree.nodes[k].right_child == _TREE_LEAF:
            sorted_leaves[0][n].left_child = self.splitting_tree.nodes[k].left_child
            sorted_leaves[0][n].right_child = self.splitting_tree.nodes[k].right_child
            sorted_leaves[0][n].feature = self.splitting_tree.nodes[k].feature
            sorted_leaves[0][n].threshold = self.splitting_tree.nodes[k].threshold
            sorted_leaves[0][n].impurity = self.splitting_tree.nodes[k].impurity
            sorted_leaves[0][n].n_node_samples = self.splitting_tree.nodes[k].n_node_samples
            sorted_leaves[0][n].weighted_n_node_samples = self.splitting_tree.nodes[k].weighted_n_node_samples
            n+=1

        return 0

    cdef int get_splitting_tree_leaves_samples_and_pos(self, SIZE_t** starts, SIZE_t** ends, Node* sorted_leaves, SIZE_t n_leaves, SIZE_t*** samples_leaves, SIZE_t n_samples):
        print("Get splitting tree leaves samples and pos")
        # Get the samples for each leaves and their start and end position in the samples array
        cdef SIZE_t previous_pos = 0
        cdef SIZE_t* tmp_sorted_obs = self.splitting_tree_builder.splitter.samples

        for j in range(n_leaves):
          if(previous_pos + sorted_leaves[j].n_node_samples < n_samples):
            samples_leaves[0][j] = <SIZE_t*> malloc(sorted_leaves[j].n_node_samples*sizeof(SIZE_t))
            for m in range(previous_pos, previous_pos + sorted_leaves[j].n_node_samples):
              samples_leaves[0][j][m] = tmp_sorted_obs[m]

          starts[0][j] = previous_pos
          ends[0][j] = previous_pos + sorted_leaves[j].n_node_samples
          previous_pos += sorted_leaves[j].n_node_samples

        print("Get splitting tree leaves samples and pos end")
        return 0

    cdef int switch_best_splitting_tree(self, double current_proxy_improvement, double* best_proxy_improvement, CARTGVSplitRecord* best, CARTGVSplitRecord* current, SIZE_t* starts, SIZE_t* ends, SIZE_t n_leaves, SIZE_t** sorted_obs):
        print("Switch best splitting tree")
        if current_proxy_improvement > best_proxy_improvement[0]:
          best_proxy_improvement[0] = current_proxy_improvement
          best_splitting_tree = self.splitting_tree
          splitting_tree = pickle.dumps(best_splitting_tree,0)                      # PLANTE ici, peut être lié à la réinitialisation de l'arbre
          current.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree)*sizeof(unsigned char))
          current.splitting_tree = splitting_tree
          current.starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
          current.ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
          current.starts = starts
          current.ends = ends
          current.n_childs = n_leaves
          best = current  # copy
          sorted_obs[0] = self.splitting_tree_builder.splitter.samples

#          print("free start")
#          free(current.splitting_tree)
#          free(current.starts)
#          free(current.ends)
#          print("free end")

        return 0

    cpdef int reset_scikit_learn_instances(self, np.ndarray y, int[:] len_groups):
        print("Reset scikit learn instances")

        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y,(-1,1))

        y = np.asarray(y)

        n_outputs =  y.shape[1]
        classes = []
        n_classes = []

        for k in range(n_outputs):
          classes_k = np.unique(y[:,k])
          classes.append(classes_k)
          n_classes.append(classes_k.shape[0])

        n_classes = np.array(n_classes, dtype=np.intp)

        print("Reset splitting tree start")
        # Reset the splitting tree for the next loop iteration
        self.splitting_tree = Tree(5,n_classes, n_outputs) #self.n_features

        print("Reset splitting tree end")

        max_features = max(len_groups) #len(groups[f_j])
        max_leaf_nodes = -1 #self.X.shape[0]
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = 0.0
        max_depth = 3
        min_impurity_decrease = 0.
        min_impurity_split = 0.
#         random_state = check_random_state(2547)

#         Create the Criterion, Splitter et TreeBuilder for the splitting tree

        print("Reset criterion start")
        print(n_outputs)
        print(n_classes.shape)
        criterion = Gini(n_outputs,n_classes)

        print("Reset criterion end")
        print("Reset splitter start")
        print(max_features)
        print(min_samples_leaf)
        print(min_weight_leaf)
        splitter = BestSplitter(criterion,max_features,min_samples_leaf,min_weight_leaf,random_state=check_random_state(2547)) #Plante ici

        print("Reset splitter end")
        print("Reset splitting tree builder start")

#        print(min_samples_split)
#        print(min_samples_leaf)
#        print(min_weight_leaf)
#        print(max_depth)
#        print(min_impurity_decrease)
#        print(type(min_samples_split))
#        print(type(min_samples_leaf))
#        print(type(min_weight_leaf))
#        print(type(max_depth))
#        print(type(min_impurity_decrease))
#        print(type(min_impurity_split))

#        print(splitter)
#        print(splitter.max_features)
#        print(splitter.min_samples_leaf)
#        print(splitter.min_weight_leaf)
#        print(splitter.samples)
#        print(splitter.n_samples)
#        print(splitter.weighted_n_samples)
#        print(splitter.features)
#        print(splitter.constant_features)
#        print(splitter.n_features)
#        print(splitter.feature_values)
#        print(splitter.start)
#        print(splitter.end)
#        print(splitter.y)
#        print(splitter.sample_weight)
        print(self.splitting_tree_builder)

        print(id(self.splitting_tree_builder))

        self.splitting_tree_builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                   min_samples_leaf,
                                   min_weight_leaf,
                                   max_depth,
                                   min_impurity_decrease,
                                   min_impurity_split) #Plante ici
        print("Reset splitting tree builder end")

        return 0

    cdef int node_split_v2(self, double impurity, CARTGVSplitRecord* split, SIZE_t* n_constant_features):

        cdef SIZE_t n_visited_grouped_features = 0                      # The number of group visited
        cdef SIZE_t max_grouped_features = self.max_grouped_features    # The max number of group we will visit

        cdef SIZE_t start = self.start                                  # The start of the node in the sample array
        cdef SIZE_t end = self.end
        cdef SIZE_t* sorted_obs                                         # The sorted observation of the node

        cdef int[:,:] groups = self.groups                              # The different groups
        cdef int[:] group                                               # The selected group
        cdef int[:] len_groups = self.len_groups                        # The length of each group
        cdef int len_group

        cdef int n_leaves
        cdef Node* sorted_leaves                                        # The sorted leaves of the splitting tree
        cdef SIZE_t** samples_leaves                                    # The samples of each leaves
        cdef SIZE_t n_samples                                           # The number of observations
        cdef SIZE_t* starts                                             # The start of each leaves of the splitting tree
        cdef SIZE_t* ends

        cdef CARTGVSplitRecord best, current                            # The best and current split (splitting tree)
        cdef double current_proxy_improvement = -INFINITY               # The improvement in impurity for the current split
        cdef double best_proxy_improvement = -INFINITY                  # The improvement in impurity of the best split

        cdef np.ndarray Xf

        _init_split(&best)

        # Loop until the we've visited the max number of group
#        while (n_visited_grouped_features < max_grouped_features):
        for i in range(5):
            print("######################### LOOP " + str(i) + " ###########################")
            n_visited_grouped_features += 1

            # with gil:
            f_j = 0 #np.random.randint(0,max_grouped_features)           # Select a group at random

            group = groups[f_j]
            len_group = len_groups[f_j]

            Xf = self.group_sample(group, len_group, start, end)

            self.criterion.reset()

            y = np.asarray(self.y)

            self.splitting_tree_construction(Xf,y)

            n_leaves = self.splitting_tree.n_leaves #self.get_splitting_tree_n_leaves()

            sorted_leaves = <Node*> malloc(n_leaves*sizeof(Node))

            self.get_splitting_tree_leaves(&sorted_leaves)

            samples_leaves = <SIZE_t**> malloc(n_leaves * sizeof(SIZE_t*))
            n_samples = self.splitting_tree_builder.splitter.n_samples
            starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
            ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))

            self.get_splitting_tree_leaves_samples_and_pos(&starts, &ends, sorted_leaves, n_leaves, &samples_leaves, n_samples)

            self.criterion.update(starts,ends, n_leaves)

            # Compute the improvement for the current split
            current_proxy_improvement = self.criterion.proxy_impurity_improvement() #TODO Division par 0 dans children impurity à re vérifier

            self.switch_best_splitting_tree(current_proxy_improvement, &best_proxy_improvement, &best, &current, starts, ends, n_leaves, &sorted_obs)

            self.reset_scikit_learn_instances(y, len_groups)

#            print("start free")
#
#            free(ends)
#            free(starts)
#            free(samples_leaves)
#            free(sorted_leaves)
#
#            print("end free")

        self.samples = sorted_obs

        self.criterion.reset()

        best_splitting_tree = pickle.loads(best.splitting_tree) # PLANTE ici, mauvais caractère détécté

        n_leaves = self.splitting_tree.n_leaves #self.get_splitting_tree_n_leaves()

        # Update the criterion with the best starts and ends
        self.criterion.update(best.starts,best.ends,n_leaves)

        best.impurity_childs = <double*> malloc(n_leaves * sizeof(double))

        # Compute the impurity of each children
        self.criterion.children_impurity(best.impurity_childs)

        # Compute the improvement in impurity
        best.improvement = self.criterion.impurity_improvement(impurity, best.impurity_childs)

        # Return values
        split[0] = best

        free(best.impurity_childs)

        return 0

    cdef int node_split(self, double impurity, CARTGVSplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end]
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples                             # The samples
        cdef SIZE_t start = self.start                                  # The start of the node in the sample array
        cdef SIZE_t end = self.end                                      # The end of the node in the sample array

        cdef int[:,:] groups = self.groups                              # The different groups
        cdef SIZE_t n_groups = self.n_groups                            # The number of groups
        cdef int[:] group                                               # The selected group
        cdef int[:] len_groups = self.len_groups                        # The length of each group
        cdef int len_group                                              # The length of the selected group
        cdef SIZE_t feature

        cdef SIZE_t n_visited_grouped_features = 0                      # The number of group visited
        cdef SIZE_t max_grouped_features = self.max_grouped_features    # The max number of group we will visit
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf            # The minimum number of samples in a leaf
        cdef double min_weight_leaf = self.min_weight_leaf              # The minimum weight in a leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef CARTGVSplitRecord best, current                            # The best and current split (splitting tree)
        cdef double current_proxy_improvement = -INFINITY               # The improvement in impurity for the current split
        cdef double best_proxy_improvement = -INFINITY                  # The improvement in impurity of the best split

        cdef SIZE_t f_j                                                 # The index of the selected group
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t k
        cdef SIZE_t l
        cdef SIZE_t m
        cdef SIZE_t n

        cdef SIZE_t* sorted_obs                                         # The sorted observation of the node
        cdef SIZE_t* tmp_sorted_obs                                     # A temporary variable that will contains the sorted observations
        cdef SIZE_t n_samples                                           # The number of observations
        cdef SIZE_t n_nodes                                             # The number of nodes
        cdef Node* sorted_leaves                                        # The sorted leaves of the splitting tree
        cdef int n_leaves = 0                                           # The number of leaves of the splitting tree
        cdef SIZE_t** samples_leaves                                    # The samples of each leaves
        cdef SIZE_t* starts                                             # The start of each leaves
        cdef SIZE_t* ends                                               # The end of each leavess
        cdef SIZE_t previous_pos = 0
        with gil:
           Xf = np.empty((end-start,1)) #Obligatoire sinon erreur de compilation ...
        cdef bytes splitting_tree                                       # The variable that will contains the serialized splitting tree

        _init_split(&best)

        # Loop until the we've visited the max number of group
        while (n_visited_grouped_features < max_grouped_features):

            n_visited_grouped_features += 1

            with gil:
              f_j = 0 #np.random.randint(0,max_grouped_features)           # Select a group at random

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
              # Create the splitting tree
              print("splitting_tree build start")
              y = np.asarray(self.y)
              self.splitting_tree_builder.build(self.splitting_tree, Xf, y, None) # PLANTE ici, peut être lié au splitting tree et a sa réinitialisation à chaque tour de boucle, ou les shapes de Xf, et y
              print("splitting_tree build end")

              # Necessary loop to get the number of leaves as self.splitting_tree.n_leaves crash the program (the np.sum in the splitting_tree.n_leaves property has an error)
              n_leaves = 0
              for i in range(len(self.splitting_tree.children_left)):
                if(self.splitting_tree.children_left[i] == -1 and self.splitting_tree.children_right[i] == -1):
                    n_leaves += 1

            n_nodes = self.splitting_tree.node_count
            sorted_leaves = <Node*> malloc(n_leaves*sizeof(Node))

            samples_leaves = <SIZE_t**> malloc(n_leaves * sizeof(SIZE_t*))

            tmp_sorted_obs = self.splitting_tree_builder.splitter.samples
            n_samples = self.splitting_tree_builder.splitter.n_samples

            # Get the nodes if it is a leaf
            n = 0
            for k in range(n_nodes):
              if self.splitting_tree.nodes[k].left_child == _TREE_LEAF and self.splitting_tree.nodes[k].right_child == _TREE_LEAF:
                with gil:
#                    sorted_leaves[n] = self.splitting_tree.nodes[k]
                    sorted_leaves[n].left_child = self.splitting_tree.nodes[k].left_child
                    sorted_leaves[n].right_child = self.splitting_tree.nodes[k].right_child
                    sorted_leaves[n].feature = self.splitting_tree.nodes[k].feature
                    sorted_leaves[n].threshold = self.splitting_tree.nodes[k].threshold
                    sorted_leaves[n].impurity = self.splitting_tree.nodes[k].impurity
                    sorted_leaves[n].n_node_samples = self.splitting_tree.nodes[k].n_node_samples
                    sorted_leaves[n].weighted_n_node_samples = self.splitting_tree.nodes[k].weighted_n_node_samples
                    n+=1

            # Get the samples for each leaves and their start and end position in the samples array
            previous_pos = 0
            starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
            ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
            for j in range(n_leaves):
              if(previous_pos + sorted_leaves[j].n_node_samples < n_samples):
                samples_leaves[j] = <SIZE_t*> malloc(sorted_leaves[j].n_node_samples*sizeof(SIZE_t))
                for m in range(previous_pos, previous_pos + sorted_leaves[j].n_node_samples):
                  samples_leaves[j][m] = tmp_sorted_obs[m]

              starts[j] = previous_pos
              ends[j] = previous_pos + sorted_leaves[j].n_node_samples
              previous_pos += sorted_leaves[j].n_node_samples

            with gil:
                print("criterion update")
            # Update the criterion with the new starts and ends positions
            self.criterion.update(starts,ends, n_leaves)

            with gil:
                print("proxy improvement")
            # Compute the improvement for the current split
            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

            with gil:
                print("check best split")

            # Check if the current split is better than the current best split
            if current_proxy_improvement > best_proxy_improvement:
                best_proxy_improvement = current_proxy_improvement
                with gil:
                  best_splitting_tree = self.splitting_tree
                  print("dumps start")
                  splitting_tree = pickle.dumps(best_splitting_tree,0)                      # PLANTE ici, peut être lié à la réinitialisation de l'arbre
                  print("dumps end")
                  current.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree)*sizeof(unsigned char))
                  current.splitting_tree = splitting_tree
                current.starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
                current.ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
                current.starts = starts
                current.ends = ends
                current.n_childs = n_leaves
                best = current  # copy
                sorted_obs = tmp_sorted_obs

            with gil:
                n_outputs =  y.shape[1]
                classes = []
                n_classes = []

                for k in range(n_outputs):
                  classes_k = np.unique(y[:,k])
                  classes.append(classes_k)
                  n_classes.append(classes_k.shape[0])

                n_classes = np.array(n_classes, dtype=np.intp)

                # Reset the splitting tree for the next loop iteration
                self.splitting_tree = Tree(self.n_features,n_classes, n_outputs)

#                max_features = len(groups[f_j])
#                max_leaf_nodes = self.X.shape[0]
#                min_samples_leaf = 1
#                min_samples_split = 2
#                min_weight_leaf = 0.0
#                max_depth = 3
#                min_impurity_decrease = 0.
#                min_impurity_split = 0
##                random_state = check_random_state(2547)
#
#                # Create the Criterion, Splitter et TreeBuilder for the splitting tree
#                criterion = Gini(n_outputs,n_classes)
#                splitter = BestSplitter(criterion,max_features,min_samples_leaf,min_weight_leaf,check_random_state(2547))
#                self.splitting_tree_builder = DepthFirstTreeBuilder(splitter, min_samples_split,
#                                           min_samples_leaf,
#                                           min_weight_leaf,
#                                           max_depth,
#                                           min_impurity_decrease,
#                                           min_impurity_split)

        # Update the samples array
        self.samples = sorted_obs
        self.criterion.reset()
        with gil:
          best_splitting_tree = pickle.loads(best.splitting_tree) # PLANTE ici, mauvais caractère détécté

          # Necessary loop to get the number of leaves as self.splitting_tree.n_leaves crash the program (the np.sum in the splitting_tree.n_leaves property has an error)
          n_leaves = 0
          for i in range(len(best_splitting_tree.children_left)):
            if(best_splitting_tree.children_left[i] == -1 and best_splitting_tree.children_right[i] == -1):
                n_leaves += 1

        # Update the criterion with the best starts and ends
        self.criterion.update(best.starts,best.ends,n_leaves)
        best.impurity_childs = <double*> malloc(n_leaves * sizeof(double))

        # Compute the impurity of each children
        self.criterion.children_impurity(best.impurity_childs)

        # Compute the improvement in impurity
        best.improvement = self.criterion.impurity_improvement(impurity, best.impurity_childs)

        # Return values
        split[0] = best
        return 0

    """
    --------------------------------------------------------------------------------------------------------------------
                                                        TEST NODE SPLIT
    --------------------------------------------------------------------------------------------------------------------
    """

    cpdef int test_node_split(self,
                        double impurity,
                        SIZE_t n_constant_features):
        print("################## TEST NODE SPLIT ###################")
        cdef CARTGVSplitRecord split
        res = self.node_split_v2(impurity,&split,&n_constant_features)
        return res

    cpdef int test_one_split(self, double imurity, SIZE_t n_constant_features):
        # Find the best split
        cdef SIZE_t* samples = self.samples                             # The samples
        cdef SIZE_t start = self.start                                  # The start of the node in the sample array
        cdef SIZE_t end = self.end                                      # The end of the node in the sample array

        cdef int[:,:] groups = self.groups                              # The different groups
        cdef SIZE_t n_groups = self.n_groups                            # The number of groups
        cdef int[:] group                                               # The selected group
        cdef int[:] len_groups = self.len_groups                        # The length of each group
        cdef int len_group                                              # The length of the selected group

        cdef SIZE_t f_j                                                 # The index of the selected group
        cdef SIZE_t i
        cdef SIZE_t l

        Xf = np.empty((end-start,1)) #Obligatoire sinon erreur de compilation ...
        cdef bytes splitting_tree                                       # The variable that will contains the serialized splitting tree

        f_j = 0 #np.random.randint(0,max_grouped_features)           # Select a group at random

        group = groups[f_j]
        len_group = len_groups[f_j]

        Xf = np.empty((end-start,len_group)) # Récupère la shape correcte des données

        # Take the observations columns of group f_j between indexes start and end
        for i in range(start,end):
          for l in range(len_group):
            Xf[i][l] = self.X[samples[i],group[l]]

        # Evaluate all splits
        self.criterion.reset()

        y = np.asarray(self.y)

        self.splitting_tree_builder.build(self.splitting_tree, Xf, y, None) # PLANTE ici, peut être lié au splitting tree et a sa réinitialisation à chaque tour de boucle, ou les shapes de Xf, et y

#        print(self.splitting_tree.nodes[0])
#        print(self.splitting_tree.nodes[1])
#        print(self.splitting_tree.nodes[2])
#        print(self.splitting_tree.nodes[3])
#        print(self.splitting_tree.nodes[4])
#        print(self.splitting_tree.nodes[5])
#        print(self.splitting_tree.nodes[6])
#        print(self.splitting_tree.nodes[7])
#        print(self.splitting_tree.nodes[8])
#        print(self.splitting_tree.nodes[9])
#        print(self.splitting_tree.nodes[10])

        return 0

    cpdef int test_n_split(self, double impurity, SIZE_t n_constant_features, int n_split, int tree_look):
        # Find the best split
        cdef SIZE_t* samples = self.samples                             # The samples
        cdef SIZE_t start = self.start                                  # The start of the node in the sample array
        cdef SIZE_t end = self.end                                      # The end of the node in the sample array

        cdef int[:,:] groups = self.groups                              # The different groups
        cdef SIZE_t n_groups = self.n_groups                            # The number of groups
        cdef int[:] group                                               # The selected group
        cdef int[:] len_groups = self.len_groups                        # The length of each group
        cdef int len_group                                              # The length of the selected group

        cdef SIZE_t f_j                                                 # The index of the selected group
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t l

        tree = []

        Xf = np.empty((end-start,1)) #Obligatoire sinon erreur de compilation ...

        for j in range(n_split):

            f_j = 0 #np.random.randint(0,max(len_groups))           # Select a group at random
            group = groups[f_j]
            len_group = len_groups[f_j]

            Xf = np.empty((end-start,len_group)) # Récupère la shape correcte des données

            # Take the observations columns of group f_j between indexes start and end
            for i in range(start,end):
              for l in range(len_group):
                Xf[i][l] = self.X[samples[i],group[l]]

            # Evaluate all splits
            self.criterion.reset()

            y = np.asarray(self.y)

            print(self.y)
            print(type(self.y))
            print(type(y))
            print(type(Xf))
            print(type(self.splitting_tree))
            print(Xf.shape)
            print(y.shape)
            print(self.splitting_tree)
            print(self.splitting_tree.n_classes[0])
            print(self.splitting_tree_builder)

            self.splitting_tree_builder.build(self.splitting_tree, Xf, y, None) # PLANTE ici, peut être lié au splitting tree et a sa réinitialisation à chaque tour de boucle, ou les shapes de Xf, et y
            tree.append(self.splitting_tree)

            n_outputs =  y.shape[1]
            classes = []
            n_classes = []

            for k in range(n_outputs):
              classes_k = np.unique(y[:,k])
              classes.append(classes_k)
              n_classes.append(classes_k.shape[0])

            n_classes = np.array(n_classes, dtype=np.intp)

            # Reset the splitting tree for the next loop iteration
            self.splitting_tree = Tree(self.n_features,n_classes, n_outputs)

            max_features = len(groups[f_j])
            max_leaf_nodes = self.X.shape[0]
            min_samples_leaf = 1
            min_samples_split = 2
            min_weight_leaf = 0.0
            max_depth = 3
            min_impurity_decrease = 0.
            min_impurity_split = 0
            random_state = check_random_state(2547)

            # Create the Criterion, Splitter et TreeBuilder for the splitting tree
            criterion = Gini(n_outputs,n_classes)
            splitter = BestSplitter(criterion,max_features,min_samples_leaf,min_weight_leaf,random_state)
            self.splitting_tree_builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                       min_samples_leaf,
                                       min_weight_leaf,
                                       max_depth,
                                       min_impurity_decrease,
                                       min_impurity_split)

        self.splitting_tree = tree[tree_look]

        return 0

    cpdef unsigned char[::1] test_best_node_split(self, double impurity, SIZE_t n_constant_features):
        # Find the best split
        cdef SIZE_t* samples = self.samples                             # The samples
        cdef SIZE_t start = self.start                                  # The start of the node in the sample array
        cdef SIZE_t end = self.end                                      # The end of the node in the sample array

        cdef int[:,:] groups = self.groups                              # The different groups
        cdef SIZE_t n_groups = self.n_groups                            # The number of groups
        cdef int[:] group                                               # The selected group
        cdef int[:] len_groups = self.len_groups                        # The length of each group
        cdef int len_group                                              # The length of the selected group
        cdef SIZE_t feature

        cdef SIZE_t n_visited_grouped_features = 0                      # The number of group visited
        cdef SIZE_t max_grouped_features = self.max_grouped_features    # The max number of group we will visit
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf            # The minimum number of samples in a leaf
        cdef double min_weight_leaf = self.min_weight_leaf              # The minimum weight in a leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef CARTGVSplitRecord best, current, tree1, tree2                            # The best and current split (splitting tree)
        cdef double current_proxy_improvement = -INFINITY               # The improvement in impurity for the current split
        cdef double best_proxy_improvement = -INFINITY                  # The improvement in impurity of the best split

        cdef SIZE_t f_j                                                 # The index of the selected group
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t k
        cdef SIZE_t l
        cdef SIZE_t m
        cdef SIZE_t n
        cdef SIZE_t o

        cdef SIZE_t* sorted_obs                                         # The sorted observation of the node
        cdef SIZE_t* tmp_sorted_obs                                     # A temporary variable that will contains the sorted observations
        cdef SIZE_t n_samples                                           # The number of observations
        cdef SIZE_t n_nodes                                             # The number of nodes
        cdef Node* sorted_leaves                                        # The sorted leaves of the splitting tree
        cdef int n_leaves = 0                                           # The number of leaves of the splitting tree
        cdef SIZE_t** samples_leaves                                    # The samples of each leaves
        cdef SIZE_t* starts                                             # The start of each leaves
        cdef SIZE_t* ends                                               # The end of each leavess
        cdef SIZE_t previous_pos = 0
        Xf = np.empty((end-start,1)) #Obligatoire sinon erreur de compilation ...
        cdef bytes splitting_tree                                       # The variable that will contains the serialized splitting tree

        _init_split(&best)

        print("test best node split")
        # Loop until the we've visited the max number of group
        for o in range(2):

            f_j = 0 #np.random.randint(0,max_grouped_features)           # Select a group at random

            group = groups[f_j]
            len_group = len_groups[f_j]

            print(start)
            print(end)

            Xf = np.empty((end-start,len_group)) # Récupère la shape correcte des données

            # Take the observations columns of group f_j between indexes start and end
            for i in range(start,end):
              for l in range(len_group):
                Xf[i][l] = self.X[samples[i],group[l]]

            y = np.asarray(self.y)

            n_outputs =  y.shape[1]
            classes = []
            n_classes = []

            y_encoded = np.zeros(y.shape, dtype=int)

            for k in range(n_outputs):
              classes_k, y_encoded[:, k] = np.unique(y[:,k], return_inverse=True)
              classes.append(classes_k)
              n_classes.append(classes_k.shape[0])
            y = y_encoded

            n_classes = np.array(n_classes, dtype=np.intp)

            max_features = len(groups[f_j])
            max_leaf_nodes = self.X.shape[0]
            min_samples_leaf = 1
            min_samples_split = 2
            min_weight_leaf = 0.0
            max_depth = 3
            min_impurity_decrease = 0
            min_impurity_split = 0
#                random_state = check_random_state(2547)

            # Create the Criterion, Splitter et TreeBuilder for the splitting tree
#            print(n_classes)
#            print(n_outputs)
            criterion = Gini(n_outputs,n_classes)
#            print(criterion)
            splitter = BestSplitter(criterion,max_features,min_samples_leaf,min_weight_leaf,check_random_state(2547))
#            print(splitter)
#            print(min_samples_split)
#            print(min_samples_leaf)
#            print(min_weight_leaf)
#            print(max_depth)
#            print(min_impurity_decrease)
#            print(min_impurity_split)

            # Reset the splitting tree for the next loop iteration
            self.splitting_tree = Tree(len_group,n_classes, n_outputs)

            print(self.splitting_tree_builder)
            self.splitting_tree_builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                       min_samples_leaf,
                                       min_weight_leaf,
                                       max_depth,
                                       min_impurity_decrease,
                                       min_impurity_split)

            print(self.splitting_tree_builder)

            # Evaluate all splits
            self.criterion.reset()

            # Create the splitting tree
            print("splitting_tree build start")
            print(self.y)
            print(type(self.y))
#            print(np.asarray(self.y))
#            y = np.asarray(self.y)
            print(type(y))
            print(type(Xf))
            print(type(self.splitting_tree))
            print(Xf.shape)
            print(y.shape)
            print(self.splitting_tree)
            print(self.splitting_tree.n_classes[0])
#            y = self.y
#            print(Xf)
#            print(y)
            print(self.splitting_tree_builder)
            self.splitting_tree_builder.build(self.splitting_tree, Xf, y) # PLANTE ici, peut être lié au splitting tree et a sa réinitialisation à chaque tour de boucle, ou les shapes de Xf, et y
            print("splitting_tree build end")

            # Necessary loop to get the number of leaves as self.splitting_tree.n_leaves crash the program (the np.sum in the splitting_tree.n_leaves property has an error)
            n_leaves = 0
            for i in range(len(self.splitting_tree.children_left)):
                if(self.splitting_tree.children_left[i] == -1 and self.splitting_tree.children_right[i] == -1):
                    n_leaves += 1

            n_nodes = self.splitting_tree.node_count
            sorted_leaves = <Node*> malloc(n_leaves*sizeof(Node))

            samples_leaves = <SIZE_t**> malloc(n_leaves * sizeof(SIZE_t*))

            tmp_sorted_obs = self.splitting_tree_builder.splitter.samples
            n_samples = self.splitting_tree_builder.splitter.n_samples

            # Get the nodes if it is a leaf
            n = 0
            for k in range(n_nodes):
              if self.splitting_tree.nodes[k].left_child == _TREE_LEAF and self.splitting_tree.nodes[k].right_child == _TREE_LEAF:
#               sorted_leaves[n] = self.splitting_tree.nodes[k]
                sorted_leaves[n].left_child = self.splitting_tree.nodes[k].left_child
                sorted_leaves[n].right_child = self.splitting_tree.nodes[k].left_child
                sorted_leaves[n].feature = self.splitting_tree.nodes[k].feature
                sorted_leaves[n].threshold = self.splitting_tree.nodes[k].threshold
                sorted_leaves[n].impurity = self.splitting_tree.nodes[k].impurity
                sorted_leaves[n].n_node_samples = self.splitting_tree.nodes[k].n_node_samples
                sorted_leaves[n].weighted_n_node_samples = self.splitting_tree.nodes[k].weighted_n_node_samples
                n+=1

            # Get the samples for each leaves and their start and end position in the samples array
            previous_pos = 0
            starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
            ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
            for j in range(n_leaves):
              if(previous_pos + sorted_leaves[j].n_node_samples < n_samples):
                samples_leaves[j] = <SIZE_t*> malloc(sorted_leaves[j].n_node_samples*sizeof(SIZE_t))
                for m in range(previous_pos, previous_pos + sorted_leaves[j].n_node_samples):
                  samples_leaves[j][m] = tmp_sorted_obs[m]

              starts[j] = previous_pos
              ends[j] = previous_pos + sorted_leaves[j].n_node_samples
              previous_pos += sorted_leaves[j].n_node_samples

            print("criterion update")
            # Update the criterion with the new starts and ends positions
            self.criterion.update(starts,ends, n_leaves)

            print("proxy improvement")
            # Compute the improvement for the current split
            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

            print("check best split")

            # Check if the current split is better than the current best split
            if current_proxy_improvement > best_proxy_improvement:
                best_proxy_improvement = current_proxy_improvement
                best_splitting_tree = self.splitting_tree
                print("dumps start")
                splitting_tree = pickle.dumps(best_splitting_tree,0)                      # PLANTE ici, peut être lié à la réinitialisation de l'arbre
                print("dumps end")
                current.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree)*sizeof(unsigned char))
                print("malloc done")
                current.splitting_tree = splitting_tree
                print("struct assignation")
                current.starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
                current.ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
                current.starts = starts
                current.ends = ends
                current.n_childs = n_leaves
                best = current  # copy


#            best_splitting_tree = self.splitting_tree
#            print(best_splitting_tree)
#            splitting_tree = pickle.dumps(best_splitting_tree,0)                      # PLANTE ici, peut être lié à la réinitialisation de l'arbre
#            print("dumps")
#            current.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree)*sizeof(unsigned char))
#            print("memory allocation")
#            current.splitting_tree = splitting_tree
#            print("Struct assignation")
#            current.starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
#            current.ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
#            current.starts = starts
#            current.ends = ends
#            current.n_childs = n_leaves

#            if(o == 0):
#                print("first tree")
#                best_splitting_tree = self.splitting_tree
#                splitting_tree = pickle.dumps(best_splitting_tree,0)                      # PLANTE ici, peut être lié à la réinitialisation de l'arbre
#                current.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree)*sizeof(unsigned char))
#                current.splitting_tree = splitting_tree
#                current.starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
#                current.ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
#                current.starts = starts
#                current.ends = ends
#                current.n_childs = n_leaves
#                tree1 = current
#            if(o == 1):
#                print("second tree")
#                best_splitting_tree = self.splitting_tree
#                print(best_splitting_tree)
#                splitting_tree = pickle.dumps(best_splitting_tree,0)                      # PLANTE ici, peut être lié à la réinitialisation de l'arbre
#                print("dumps")
#                current.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree)*sizeof(unsigned char))
#                print("memory allocation")
#                current.splitting_tree = splitting_tree
#                print("Struct assignation")
#                current.starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
#                current.ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
#                current.starts = starts
#                current.ends = ends
#                current.n_childs = n_leaves
#                tree2 = current

#            print("start free")
#            free(current.starts)
#            print("free starts")
#            free(current.ends)
#            print("free ends")
#            free(current.splitting_tree)
#            print("free tree")

            print("end loop")

        print("return")
#        print(tree1.n_childs)
#        print(tree1.splitting_tree)
#        print(tree2.splitting_tree)
        print(best.splitting_tree)
        return np.asarray(<unsigned char[:sys.getsizeof(best.splitting_tree)*sizeof(unsigned char)]> best.splitting_tree) #tree1.splitting_tree, tree2.splitting_tree, best.splitting_tree

    cpdef unsigned char[::1] test_splitting_tree_into_struct(self, Tree splitting_tree):

        cdef CARTGVSplitRecord current                            # The best and current split (splitting tree)

        splitting_tree_s = pickle.dumps(splitting_tree,0)         # PLANTE ici, peut être lié à la réinitialisation de l'arbre
        current.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree_s)*sizeof(unsigned char))
        current.splitting_tree = splitting_tree_s

        return np.asarray(<unsigned char[:sys.getsizeof(splitting_tree_s)*sizeof(unsigned char)]> current.splitting_tree)

    cpdef int test_sklearn_builder_field(self):
        cdef SIZE_t* samples = self.samples                             # The samples
        cdef SIZE_t start = self.start                                  # The start of the node in the sample array
        cdef SIZE_t end = self.end                                      # The end of the node in the sample array

        cdef int[:,:] groups = self.groups                              # The different groups
        cdef SIZE_t n_groups = self.n_groups                            # The number of groups
        cdef int[:] group                                               # The selected group
        cdef int[:] len_groups = self.len_groups                        # The length of each group
        cdef int len_group                                              # The length of the selected group
        cdef SIZE_t feature

        cdef SIZE_t n_visited_grouped_features = 0                      # The number of group visited
        cdef SIZE_t max_grouped_features = self.max_grouped_features    # The max number of group we will visit
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf            # The minimum number of samples in a leaf
        cdef double min_weight_leaf = self.min_weight_leaf              # The minimum weight in a leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef CARTGVSplitRecord best, current                            # The best and current split (splitting tree)
        cdef double current_proxy_improvement = -INFINITY               # The improvement in impurity for the current split
        cdef double best_proxy_improvement = -INFINITY                  # The improvement in impurity of the best split

        cdef SIZE_t f_j                                                 # The index of the selected group
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t k
        cdef SIZE_t l
        cdef SIZE_t m
        cdef SIZE_t n

        cdef SIZE_t* sorted_obs                                         # The sorted observation of the node
        cdef SIZE_t* tmp_sorted_obs                                     # A temporary variable that will contains the sorted observations
        cdef SIZE_t n_samples                                           # The number of observations
        cdef SIZE_t n_nodes                                             # The number of nodes
        cdef Node* sorted_leaves                                        # The sorted leaves of the splitting tree
        cdef int n_leaves = 0                                           # The number of leaves of the splitting tree
        cdef SIZE_t** samples_leaves                                    # The samples of each leaves
        cdef SIZE_t* starts                                             # The start of each leaves
        cdef SIZE_t* ends                                               # The end of each leavess
        cdef SIZE_t previous_pos = 0

        Xf = np.empty((end-start,1)) #Obligatoire sinon erreur de compilation ...
        cdef bytes splitting_tree                                       # The variable that will contains the serialized splitting tree

        _init_split(&best)

        # Loop until the we've visited the max number of group
#        while (n_visited_grouped_features < max_grouped_features):

        n_visited_grouped_features += 1

        f_j = 0 #np.random.randint(0,max_grouped_features)           # Select a group at random

        group = groups[f_j]
        len_group = len_groups[f_j]

        Xf = np.empty((end-start,len_group)) # Récupère la shape correcte des données

        # Take the observations columns of group f_j between indexes start and end
        for i in range(start,end):
          for l in range(len_group):
            Xf[i][l] = self.X[samples[i],group[l]]

        # Evaluate all splits
        self.criterion.reset()

        # Create the splitting tree
        y = np.asarray(self.y)
        self.splitting_tree_builder.build(self.splitting_tree, Xf, y, None) # PLANTE ici, peut être lié au splitting tree et a sa réinitialisation à chaque tour de boucle, ou les shapes de Xf, et y

        # Necessary loop to get the number of leaves as self.splitting_tree.n_leaves crash the program (the np.sum in the splitting_tree.n_leaves property has an error)
        n_leaves = 0
        for i in range(len(self.splitting_tree.children_left)):
          if(self.splitting_tree.children_left[i] == -1 and self.splitting_tree.children_right[i] == -1):
              n_leaves += 1

        n_nodes = self.splitting_tree.node_count
        sorted_leaves = <Node*> malloc(n_leaves*sizeof(Node))

        samples_leaves = <SIZE_t**> malloc(n_leaves * sizeof(SIZE_t*))

        tmp_sorted_obs = self.splitting_tree_builder.splitter.samples
        n_samples = self.splitting_tree_builder.splitter.n_samples

        # Get the nodes if it is a leaf
        n = 0
        for k in range(n_nodes):
            if self.splitting_tree.nodes[k].left_child == _TREE_LEAF and self.splitting_tree.nodes[k].right_child == _TREE_LEAF:
                sorted_leaves[n].left_child = self.splitting_tree.nodes[k].left_child
                sorted_leaves[n].right_child = self.splitting_tree.nodes[k].right_child
                sorted_leaves[n].feature = self.splitting_tree.nodes[k].feature
                sorted_leaves[n].threshold = self.splitting_tree.nodes[k].threshold
                sorted_leaves[n].impurity = self.splitting_tree.nodes[k].impurity
                sorted_leaves[n].n_node_samples = self.splitting_tree.nodes[k].n_node_samples
                sorted_leaves[n].weighted_n_node_samples = self.splitting_tree.nodes[k].weighted_n_node_samples
                n+=1
                print(self.splitting_tree.nodes[k].left_child)
                print(self.splitting_tree.nodes[k].right_child)
                print(self.splitting_tree.nodes[k].feature)
                print(self.splitting_tree.nodes[k].threshold)
                print(self.splitting_tree.nodes[k].impurity)
                print(self.splitting_tree.nodes[k].n_node_samples)
                print(self.splitting_tree.nodes[k].weighted_n_node_samples)

        # Get the samples for each leaves and their start and end position in the samples array
        previous_pos = 0
        starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
        ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
        for j in range(n_leaves):
          if(previous_pos + sorted_leaves[j].n_node_samples < n_samples):
            samples_leaves[j] = <SIZE_t*> malloc(sorted_leaves[j].n_node_samples*sizeof(SIZE_t))
            for m in range(previous_pos, previous_pos + sorted_leaves[j].n_node_samples):
              samples_leaves[j][m] = tmp_sorted_obs[m]
              print(tmp_sorted_obs[m])

          starts[j] = previous_pos
          ends[j] = previous_pos + sorted_leaves[j].n_node_samples
          previous_pos += sorted_leaves[j].n_node_samples
          print("POSITION")
          print(starts[j])
          print(ends[j])

        print("TEST builder field")
        print(np.asarray(<SIZE_t[:sys.getsizeof(n_samples)*sizeof(SIZE_t)]> tmp_sorted_obs))
        print(n_samples)

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""
        with gil:
            print("Criterion node value start")
            self.criterion.node_value(dest)
            print("Criterion node value end")

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()
