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

    @property
    def splitting_tree(self):
        return self.splitting_tree

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

        # Reshape the responses to correpsond to the format expected
        self.y = np.asarray(y).reshape(y.shape[0],y.shape[1])

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
        y_encoded = np.zeros((n_samples,1), dtype=int)
        for k in range(n_outputs):
          classes_k, y_encoded[:,k] = np.unique(y[:,k], return_inverse=True)
          classes.append(classes_k)
          n_classes.append(classes_k.shape[0])

        n_classes = np.array(n_classes, dtype=np.intp)

        # Create the splitting tree
        self.splitting_tree = Tree(n_features,n_classes, n_outputs)

        # TODO make those variable settable parameters
        max_features = len(groups[0]) #max(1, int(np.sqrt(n_features))) #len(groups[0])
        max_leaf_nodes = X.shape[0]
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = 0.0 #(0.25 * n_samples)
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

            # Update the criterion with the new starts and ends positions
            self.criterion.update(starts,ends, n_leaves)

            # Compute the improvement for the current split
            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

            # Check if the current split is better than the current best split
            if current_proxy_improvement > best_proxy_improvement:
                best_proxy_improvement = current_proxy_improvement
                with gil:
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

    cpdef tuple test_node_split(self,
                        double impurity,
                        SIZE_t n_constant_features):

        cdef CARTGVSplitRecord split
        res = self.node_split(impurity,&split,&n_constant_features)
        return res #TODO, trouver un moyen de retourner le split

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
        print("splitting_tree build end")

        print(self.splitting_tree.nodes[0])
        print(self.splitting_tree.nodes[1])
        print(self.splitting_tree.nodes[2])
        print(self.splitting_tree.nodes[3])
        print(self.splitting_tree.nodes[4])
        print(self.splitting_tree.nodes[5])
        print(self.splitting_tree.nodes[6])
        print(self.splitting_tree.nodes[7])
        print(self.splitting_tree.nodes[8])
        print(self.splitting_tree.nodes[9])
        print(self.splitting_tree.nodes[10])

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
            tree.append(self.splitting_tree)
            print("splitting_tree build end")

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

        self.splitting_tree = tree[tree_look]

        return 0

    cpdef unsigned char[::1] test_splitting_tree_into_struct(self, Tree splitting_tree):

        cdef CARTGVSplitRecord current                            # The best and current split (splitting tree)

        splitting_tree_s = pickle.dumps(splitting_tree,0)         # PLANTE ici, peut être lié à la réinitialisation de l'arbre
        current.splitting_tree = <unsigned char*>malloc(sys.getsizeof(splitting_tree_s)*sizeof(unsigned char))
        current.splitting_tree = splitting_tree_s

        return np.asarray(<unsigned char[:sys.getsizeof(splitting_tree_s)*sizeof(unsigned char)]> current.splitting_tree)

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""
        with gil:
            print("Criterion node value start")
            self.criterion.node_value(dest)
            print("Criterion node value end")

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()
