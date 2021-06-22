from CARTGVCriterion cimport CARTGVCriterion

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdlib cimport malloc, free, calloc
from libc.stdio cimport printf
from libc.math cimport sqrt


import numpy as np
import pickle as pickle
import sys
import random
import faulthandler
cimport numpy as np
import matplotlib.pyplot as plt
import copy
np.import_array()

from sklearn.tree._utils cimport log, rand_int, rand_uniform, RAND_R_MAX, safe_realloc
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.tree._tree import BestFirstTreeBuilder, DepthFirstTreeBuilder
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._criterion import Gini, Entropy, MSE, FriedmanMSE, MAE, Poisson
from sklearn.utils.validation import check_random_state
from sklearn.tree._tree import DOUBLE

from sklearn.tree._tree cimport TreeBuilder, Tree
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._splitter cimport Splitter

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

CRITERIA_CLF = {"gini": Gini,
                "entropy": Entropy}
CRITERIA_REG = {"mse": MSE,
                "friedman_mse": FriedmanMSE,
                "mae": MAE,
                "poisson": Poisson}

cdef inline void _init_split(CARTGVSplitRecord* self) nogil:
    """
    Initialize a CARTGVSplitRecord
    splitting tree will be initialized later as we dont know its size yet
    """
    self.impurity_childs = [0]
    self.starts = [0]
    self.ends = [0]
    self.improvement = -INFINITY
    self.splitting_tree = NULL
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
    def features(self):
        if self.features != NULL:
            return np.asarray(<SIZE_t[:self.n_features]>self.features)
        else:
            return None

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
        if self.samples != NULL:
            return np.asarray(<SIZE_t[:self.n_samples]>self.samples)
        else:
            return None

    @property
    def sample_weight(self):
        if self.sample_weight != NULL:
            return np.asarray(<DOUBLE_t[:self.n_samples]>self.sample_weight)
        else:
            return None

    @property
    def rand_r_state(self):
        return self.rand_r_state

    @property
    def random_state(self):
        return self.random_state

    @property
    def n_classes(self):
        return self.n_classes


    def __cinit__(self, CARTGVCriterion criterion, int n_groups,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state, int max_depth, double min_impurity_decrease,
                  double min_impurity_split,
                  object mvar,
                  int mgroup,
                  object split_criterion):
        """
        Parameters
        ----------
        criterion : CARTGVCriterion
            The criterion to measure the quality of a split.
        mgroup : SIZE_t
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
        faulthandler.enable()
        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0

        self.sample_weight = NULL

        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

        self.groups = np.empty((n_groups,mgroup),dtype=int)
        self.n_groups = n_groups
        self.len_groups = np.empty((n_groups),dtype=int)

        self.splitting_tree_builder = None
        self.splitting_tree = None

        self.max_depth = max_depth
        if isinstance(mvar,int):
            self.mvar = mvar
        elif isinstance(mvar, list):
            if len(mvar) >= n_groups:
                self.mvar = np.array(mvar)
            else:
                raise ValueError('The size of mvar vector is to small compared to the number of groups, got size : ' + str(len(mvar)) + ", needed size : " + str(n_groups))
        elif isinstance(mvar,np.ndarray) and mvar.ndim == 1:
            if len(mvar) >= n_groups:
                self.mvar = mvar
            else:
                raise ValueError('The size of mvar vector is to small compared to the number of groups, got size : ' + str(len(mvar)) + ", needed size : " + str(n_groups))
        elif isinstance(mvar, str):
            self.mvar = mvar  # We need to know the size of each group for this one
        else:
            raise ValueError('mvar was not set properly. Please use an integer, a list, a numpy array or the strings : root or third')

        self.mgroup = mgroup
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        if isinstance(split_criterion,str):
            self.split_criterion = split_criterion
        elif isinstance(split_criterion,Criterion):
            self.split_criterion = split_criterion
        else:
            raise TypeError("The parameter type has the wrong type, must be type str or a Criterion and got : " + str(type))


    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass


    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, object groups, np.ndarray len_groups) except -1:

#        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX) #TODO remettre l'aléatoire
        self.rand_r_state = 2547

        cdef SIZE_t n_samples, n_features
        n_samples, n_features = X.shape

        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t* features = safe_realloc(&self.features,n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        self.y = y

        self.sample_weight = sample_weight

        self.groups = groups
        cdef int n_groups = groups.shape[0]
        self.n_groups = n_groups

#        for k in range(n_groups):
#          self.len_groups[k] = len(groups[k])
        self.len_groups = len_groups

        if isinstance(self.mvar,str):
            if self.mvar == "root":
                self.mvar = np.sqrt(self.len_groups)
            elif self.mvar == "third":
                self.mvar = np.divide(self.len_groups,3)
            else:
                raise ValueError('The string : ' + str(self.mvar) + " isn't recognised, please use either 'root' or 'third'")
        if isinstance(self.mvar, int):
            if self.mvar <= min([len(group) for group in groups]):
                self.mvar = np.repeat(self.mvar,n_groups)
            else:
                raise ValueError('The mvar value : ' + str(self.mvar) + " is bigger than the smallest group, maximum usable size is : " + str(min([len(group) for group in groups])))

        if self.mgroup == None:
            mgroup = len(groups)

        if self.mgroup > len(groups):
            raise ValueError("The mgroup value : " + str(self.mgroup) + " is bigger than the number of groups : " + str(len(groups)))

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:

            self.start = start
            self.end = end

            self.criterion.init(self.y,
                                self.sample_weight,
                                self.weighted_n_samples,
                                self.samples,
                                self.n_samples,
                                start,
                                end)

            weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

            return 0

    cdef np.ndarray group_sample(self, int[:] group, int len_group, int start, int end):
        pass

    cdef int reset_scikit_learn_instances(self, np.ndarray y, int group, int len_group):
        pass

    cdef int splitting_tree_construction(self, np.ndarray Xf, np.ndarray y):
        pass

    cdef int get_splitting_tree_leaves(self, Node** sorted_leaves):
        pass

    cdef int get_splitting_tree_leaves_pos(self, SIZE_t** starts, SIZE_t** ends, Node* sorted_leaves, SIZE_t n_leaves, SIZE_t n_samples):
        pass

    cdef int switch_best_splitting_tree(self, double current_proxy_improvement, double* best_proxy_improvement, CARTGVSplitRecord* best, SIZE_t* starts, SIZE_t* ends, SIZE_t n_leaves, int group, SIZE_t* sorted_obs):
        pass

    cdef int node_split(self, double impurity, CARTGVSplitRecord* split, SIZE_t* n_constant_features, int parent_start, int parent_end):
        pass

    cdef void node_value(self, double* dest) nogil:
        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        cdef double node_impurity = self.criterion.node_impurity()

        return node_impurity

    ################################# TEST ##################################

    cpdef int test_init(self,
                  object X,
                  DOUBLE_t[:, ::1] y,
                  np.ndarray sample_weight, object groups, np.ndarray len_groups):
        res = -1
        if(sample_weight == None):
            res = self.init(X,y,NULL,groups, len_groups)
        else:
            res = self.init(X,y,<DOUBLE_t*>sample_weight.data, groups, len_groups)
        return res

    cpdef int test_node_reset(self, SIZE_t start, SIZE_t end,
                                double weighted_n_node_samples):
        res = self.node_reset(start,end,&weighted_n_node_samples) #<double*>&wns.data
        return res

    cpdef np.ndarray test_group_sample(self, int[:] group, int len_group, int start, int end):
        pass

    cpdef int test_reset_scikit_learn_instances(self, np.ndarray y, int group, int len_group):
        pass

    cpdef int test_splitting_tree_construction(self, np.ndarray Xf, np.ndarray y):
        pass

    cpdef int test_get_splitting_tree_leaves(self):
        pass

    cpdef int test_get_splitting_tree_leaves_samples_and_pos(self):
        pass

    cpdef int test_switch_best_splitting_tree(self):
        pass

    cpdef int test_node_split(self):
        pass

    cpdef double test_node_value(self, double dest):
        self.node_value(&dest)
        return dest

    cpdef double test_node_impurity(self):
        return self.node_impurity()

cdef class BaseDenseCARTGVSplitter(CARTGVSplitter):

    cdef int init(self,
                object X,
                const DOUBLE_t[:, ::1] y,
                DOUBLE_t* sample_weight,
                object groups, np.ndarray len_groups) except -1:

        CARTGVSplitter.init(self,X,y,sample_weight,groups, len_groups)

        self.X = X

        return 0

cdef class BestCARTGVSplitter(BaseDenseCARTGVSplitter):

    def __reduce__(self):
        return (BestCARTGVSplitter, (self.criterion,
                                        self.mgroup,
                                        self.min_samples_leaf,
                                        self.min_weight_leaf,
                                        self.random_state), self.__getstate__())

    cdef np.ndarray group_sample(self, int[:] group, int len_group, int start, int end):
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t incr = 0

        Xf = np.empty((end-start,len_group)) # Récupère la shape correcte des données
        for i in range(start,end):
              for j in range(len_group):
                Xf[incr][j] = self.X[self.samples[i],group[j]]
              incr+=1

        return Xf

    cdef int reset_scikit_learn_instances(self, np.ndarray y, int group, int len_group):
        cdef SIZE_t k

        cdef SIZE_t n_outputs = y.shape[1]
        classes = []
        n_classes = []

        for k in range(n_outputs):
          classes_k = np.unique(self.y[:,k]) #TODO Test self.y instead of y
          classes.append(classes_k)
          n_classes.append(classes_k.shape[0])

        n_classes = np.array(n_classes, dtype=np.intp)

        #TODO find a better way to know if we are in classif or reg
        if not isinstance(self.split_criterion,Criterion):
            if self.split_criterion in CRITERIA_CLF:
                is_classification = True
            elif self.split_criterion in CRITERIA_REG:
                is_classification = False
                n_classes = np.array([1] * n_outputs, dtype=np.intp)
            else:
                raise ValueError("The Splitting tree criterion value "+ self.split_criterion +" isn't managed")

        self.splitting_tree = Tree(len_group, n_classes, n_outputs)

        cdef int max_features = self.mvar[group]
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t max_depth = self.max_depth
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split
        cdef object random_state = check_random_state(self.random_state)

        cdef Criterion criterion
        #TODO find a better way to know if we are in classif or reg
        if not isinstance(self.split_criterion,Criterion):
            if self.split_criterion in CRITERIA_CLF:
                is_classification = True
            elif self.split_criterion in CRITERIA_REG:
                is_classification = False
            else:
                raise ValueError("The Splitting tree criterion value "+ self.split_criterion +" isn't managed")
            if is_classification:
                criterion = CRITERIA_CLF[self.split_criterion](n_outputs,n_classes)
            else:
                criterion = CRITERIA_REG[self.split_criterion](n_outputs, self.X.shape[0])
        else:
            criterion = copy.deepcopy(self.split_criterion)

        cdef Splitter splitter = BestSplitter(criterion, max_features, min_samples_leaf, min_weight_leaf, random_state)

        cdef TreeBuilder depthFirstTreeBuilder = DepthFirstTreeBuilder(splitter, min_samples_split, min_samples_leaf, min_weight_leaf, max_depth, min_impurity_decrease, min_impurity_split)

        self.splitting_tree_builder = depthFirstTreeBuilder

        return 0

    cdef int splitting_tree_construction(self, np.ndarray Xf, np.ndarray y):

        self.splitting_tree_builder.build(self.splitting_tree, Xf, y)

        return 0

    cdef int get_splitting_tree_leaves(self, Node** sorted_leaves):
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

    cdef int get_splitting_tree_leaves_pos(self, SIZE_t** starts, SIZE_t** ends, Node* sorted_leaves, SIZE_t n_leaves, SIZE_t n_samples):
        cdef SIZE_t previous_pos = 0
        cdef SIZE_t* tmp_sorted_sample = self.splitting_tree_builder.splitter.samples

        for j in range(n_leaves):
          starts[0][j] = previous_pos
          ends[0][j] = previous_pos + sorted_leaves[j].n_node_samples
          previous_pos += sorted_leaves[j].n_node_samples

        return 0

    cdef int switch_best_splitting_tree(self, double current_proxy_improvement, double* best_proxy_improvement, CARTGVSplitRecord* best, SIZE_t* starts, SIZE_t* ends, SIZE_t n_leaves, int group, SIZE_t* sorted_obs):

        cdef unsigned char* splt_tree

        ser_splitting_tree = None

        if current_proxy_improvement > best_proxy_improvement[0]:

          best_proxy_improvement[0] = current_proxy_improvement
          best_splitting_tree = self.splitting_tree
          ser_splitting_tree = pickle.dumps(best_splitting_tree,0)
          splt_tree = <unsigned char*>malloc(sys.getsizeof(ser_splitting_tree)*sizeof(unsigned char))
          memcpy(splt_tree,<unsigned char*>ser_splitting_tree,sys.getsizeof(ser_splitting_tree)*sizeof(unsigned char))

          best[0].splitting_tree = <unsigned char*>malloc(sys.getsizeof(ser_splitting_tree)*sizeof(unsigned char))
          memcpy(best[0].splitting_tree, splt_tree, sys.getsizeof(ser_splitting_tree)*sizeof(unsigned char))
          best[0].starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
          best[0].ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
          memcpy(best[0].starts, starts, n_leaves*sizeof(SIZE_t))
          memcpy(best[0].ends, ends, n_leaves*sizeof(SIZE_t))
          best[0].n_childs = n_leaves
          best[0].group = group
          for l in range(self.splitting_tree_builder.splitter.n_samples):
            sorted_obs[l] = self.splitting_tree_builder.splitter.samples[l]

          free(splt_tree)

        return 0

    cdef int node_split(self, double impurity, CARTGVSplitRecord* split, SIZE_t* n_constant_features, int parent_start, int parent_end):

        cdef SIZE_t n_visited_grouped_features = 0                      # The number of group visited
        cdef SIZE_t mgroup = self.mgroup                                # The max number of group we will visit

        cdef SIZE_t start = self.start                                  # The start of the node in the sample array
        cdef SIZE_t end = self.end
        cdef SIZE_t* sorted_obs                                         # The sorted observation of the node

        cdef SIZE_t* current_obs
        cdef SIZE_t* current_samples

        cdef int[:,:] groups = self.groups                              # The different groups
        cdef int[:] group                                               # The selected group
        cdef int[:] len_groups = self.len_groups                        # The length of each group
        cdef int len_group

        cdef np.ndarray groups_id = np.arange(groups.shape[0])
        cdef np.ndarray groups_taken = np.full(mgroup, np.inf, dtype=int)

        cdef int n_leaves
        cdef Node* sorted_leaves                                        # The sorted leaves of the splitting tree
        cdef SIZE_t n_samples                                           # The number of observations
        cdef SIZE_t* starts                                             # The start of each leaves of the splitting tree
        cdef SIZE_t* ends

        cdef CARTGVSplitRecord best                                     # The best and current split (splitting tree)
        cdef double current_proxy_improvement = -INFINITY               # The improvement in impurity for the current split
        cdef double best_proxy_improvement = -INFINITY                  # The improvement in impurity of the best split

        cdef np.ndarray Xf

        _init_split(&best)

        sorted_obs = <SIZE_t*> malloc(self.n_samples*sizeof(SIZE_t))

        while(n_visited_grouped_features < mgroup):

            f_j = np.random.choice(np.setdiff1d(groups_id,groups_taken),1)[0]

            groups_taken[n_visited_grouped_features] = f_j

            n_visited_grouped_features += 1

            group = groups[f_j]
            len_group = len_groups[f_j]

            Xf = self.group_sample(group, len_group, start, end)

            self.criterion.reset()

            y = np.zeros((end-start,self.y.shape[1])) # Récupère la shape correcte des données

            incr = 0
            for i in range(start,end):
                for j in range(self.y.shape[1]):
                    y[incr][j] = self.y[self.samples[i],j]
                incr+=1

            self.reset_scikit_learn_instances(y, f_j, len_group)

            self.splitting_tree_construction(Xf, y)

            n_leaves = self.splitting_tree.n_leaves

            sorted_leaves = <Node*> malloc(n_leaves*sizeof(Node))

            self.get_splitting_tree_leaves(&sorted_leaves)

            n_samples = self.splitting_tree_builder.splitter.n_samples
            starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
            ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))

            self.get_splitting_tree_leaves_pos(&starts, &ends, sorted_leaves, n_leaves, n_samples)

            current_obs = <SIZE_t*> malloc(self.splitting_tree_builder.splitter.n_samples*sizeof(SIZE_t))
            for l in range(self.splitting_tree_builder.splitter.n_samples):
                current_obs[l] = self.splitting_tree_builder.splitter.samples[l]

            current_samples = <SIZE_t*> malloc(self.splitting_tree_builder.splitter.n_samples * sizeof(SIZE_t))
            incr = 0
            for incr in range(self.splitting_tree_builder.splitter.n_samples):
                current_samples[incr] = self.samples[parent_start+current_obs[incr]]

            self.criterion.samples = current_samples
            self.criterion.n_samples = self.splitting_tree_builder.splitter.n_samples

            self.criterion.update(starts, ends, n_leaves)

            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

            self.switch_best_splitting_tree(current_proxy_improvement, &best_proxy_improvement, &best, starts, ends, n_leaves, f_j, sorted_obs)

            free(current_obs)
            free(current_samples)
            free(sorted_leaves)
            free(starts)
            free(ends)

        self.criterion.samples = self.samples

        sorted_samples = []
        incr = 0
        for incr in range(self.splitting_tree_builder.splitter.n_samples):
            sorted_samples.append(self.samples[parent_start+sorted_obs[incr]])

        incr = 0
        for k in range(parent_start,parent_end):
            self.samples[k] = sorted_samples[incr]
            incr+=1

        for s in range(best.n_childs):
            best.starts[s]+=parent_start
        for e in range(best.n_childs):
            best.ends[e]+=parent_start

        self.criterion.reset()

        n_leaves = pickle.loads(best.splitting_tree).n_leaves

        self.criterion.update(best.starts,best.ends,n_leaves)

        best.impurity_childs = <double*> malloc(n_leaves * sizeof(double))

        self.criterion.children_impurity(&best.impurity_childs)

        best.improvement = self.criterion.impurity_improvement(impurity, best.impurity_childs)

        split[0] = best # best if freed later as we free split in the build method of CARTGVTreeBuilder

        free(sorted_obs)

        return 0


    ################################# TEST ##################################

    cpdef np.ndarray test_group_sample(self, int[:] group, int len_group, int start, int end):
        return self.group_sample(group,len_group,start,end)

    cpdef int test_reset_scikit_learn_instances(self, np.ndarray y, int group, int len_group):
        return self.reset_scikit_learn_instances(y,group,len_group)

    cpdef int test_splitting_tree_construction(self, np.ndarray Xf, np.ndarray y):
        return self.splitting_tree_construction(Xf,y)

    cpdef int test_get_splitting_tree_leaves(self):

        cdef Node* sorted_leaves = <Node*> malloc(self.splitting_tree.n_leaves*sizeof(Node))
        cdef int res = self.get_splitting_tree_leaves(&sorted_leaves)

        return 0

    cpdef int test_get_splitting_tree_leaves_samples_and_pos(self):

        cdef Node* sorted_leaves = <Node*> malloc(self.splitting_tree.n_leaves*sizeof(Node))
        self.get_splitting_tree_leaves(&sorted_leaves)

        cdef SIZE_t n_samples = self.splitting_tree_builder.splitter.n_samples
        cdef SIZE_t* starts = <SIZE_t*> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t))
        cdef SIZE_t* ends = <SIZE_t*> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t))

        res = self.get_splitting_tree_leaves_pos(&starts, &ends, sorted_leaves, self.splitting_tree.n_leaves, n_samples)

    cpdef int test_switch_best_splitting_tree(self):
        cdef double current_proxy_improvement, best_proxy_improvement
        current_proxy_improvement = 1.0
        best_proxy_improvement = 0.0
        cdef CARTGVSplitRecord best
        cdef SIZE_t* sorted_obs

        cdef Node* sorted_leaves = <Node*> malloc(self.splitting_tree.n_leaves*sizeof(Node))
        self.get_splitting_tree_leaves(&sorted_leaves)

        cdef SIZE_t n_samples = self.splitting_tree_builder.splitter.n_samples
        cdef SIZE_t* starts = <SIZE_t*> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t))
        cdef SIZE_t* ends = <SIZE_t*> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t))

        self.get_splitting_tree_leaves_pos(&starts, &ends, sorted_leaves, self.splitting_tree.n_leaves, n_samples)

        res = self.switch_best_splitting_tree(current_proxy_improvement, &best_proxy_improvement, &best, starts, ends, self.splitting_tree.n_leaves,0, sorted_obs)
        return 0

    cpdef int test_node_split(self):

        cdef impurity = np.inf
        cdef CARTGVSplitRecord split
        cdef SIZE_t n_constant_features
        cdef int parent_start = 0
        cdef int parent_end = 10 #TODO refaire le test

        self.node_split(impurity, &split, &n_constant_features, parent_start, parent_end)

        return 0