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
np.import_array()

from sklearn.tree._utils cimport log, rand_int, rand_uniform, RAND_R_MAX, safe_realloc
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.tree._tree import BestFirstTreeBuilder, DepthFirstTreeBuilder
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._criterion import Gini
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
                  int mgroup):
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
            self.mvar = np.repeat(mvar,n_groups) #TODO cas d'erreur lorsque mvar est supérieur à l'un des groupes
        elif isinstance(mvar, list):
            self.mvar = np.array(mvar) #TODO cas d'erreur taille liste inférieure à n_groups
        elif isinstance(mvar,np.ndarray) and mvar.ndim == 1:
            self.mvar = mvar #TODO cas d'erreur taille tableau inférieure à n_groups
        elif isinstance(mvar, str):
            self.mvar = mvar  # We need to know the size of each group for this one
        else:
            self.mvar = mvar #TODO error case, mvar does't correpsond to anything


#        self.mvar = mvar
        self.mgroup = mgroup
#        self.min_impurity_decrease = min_impurity_decrease
#        self.min_impurity_split = min_impurity_split

        ## ACCESS TO FIELDS TO CHECK CORRUPTION ##
        # NO PROBLEM
#        print(self.criterion)
#        print(np.asarray(<SIZE_t[:self.n_samples]>self.samples))
#        print(self.n_samples)
#        print(np.asarray(<SIZE_t[:self.n_features]>self.features))
#        print(self.n_features)
#        print(np.asarray(<DOUBLE_t[:self.n_samples]>self.sample_weight))
#        print(self.max_grouped_features)
#        print(self.min_samples_leaf)
#        print(self.min_weight_leaf)
#        print(self.random_state)
#        print(self.groups)
#        print(self.n_groups)
#        print(self.len_groups)
#        print(self.splitting_tree_builder)
#        print(self.splitting_tree)

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
                  DOUBLE_t* sample_weight, object groups) except -1:

#        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
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

        for k in range(n_groups):
          self.len_groups[k] = len(groups[k])

        if isinstance(self.mvar,str):
            if self.mvar == "root":
                self.mvar = np.sqrt(self.len_groups)
            elif self.mvar == "third":
                self.mvar = np.divide(self.len_groups,3)

        ## ACCESS TO FIELDS TO CHECK CORRUPTION ##
        # NO PROBLEM
#        print(self.rand_r_state)
#        print(np.asarray(<SIZE_t[:self.n_samples]>self.samples))
#        print(self.n_samples)
#        print(self.weighted_n_samples)
#        print(np.asarray(<SIZE_t[:self.n_features]>self.features))
#        print(self.n_features)
#        print(self.y)
#        if sample_weight != NULL:
#            print(np.asarray(<DOUBLE_t[:self.n_samples]>self.sample_weight))
#        print(self.groups)
#        print(self.n_groups)
#        print(self.len_groups)

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:

            self.start = start
            self.end = end

#            with gil:
#                print("### Y ###")
#                print(np.asarray(self.y[:5]))
#                print("### SAMPLES ###")
#                print(np.asarray(<SIZE_t[:self.n_samples]>self.samples))

            self.criterion.init(self.y,
                                self.sample_weight,
                                self.weighted_n_samples,
                                self.samples,
                                self.n_samples,
                                start,
                                end)

            weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

            ## ACCESS TO FIELDS TO CHECK CORRUPTION ##
            # NO PROBLEM
#            with gil:
#                print(self.start)
#                print(self.end)
#                print(self.criterion)
#                print(self.criterion.weighted_n_node_samples)

            return 0

    cdef np.ndarray group_sample(self, int[:] group, int len_group, int start, int end):
        pass

    cdef int reset_scikit_learn_instances(self, np.ndarray y, int group, int len_group):
        pass

    cdef int splitting_tree_construction(self, np.ndarray Xf, np.ndarray y):
        pass

    cdef int get_splitting_tree_leaves(self, Node** sorted_leaves):
        pass

    cdef int get_splitting_tree_leaves_samples_and_pos(self, SIZE_t** starts, SIZE_t** ends, Node* sorted_leaves, SIZE_t n_leaves, SIZE_t*** samples_leaves, SIZE_t n_samples):
        pass

    cdef int switch_best_splitting_tree(self, double current_proxy_improvement, double* best_proxy_improvement, CARTGVSplitRecord* best, CARTGVSplitRecord* current, SIZE_t* starts, SIZE_t* ends, SIZE_t n_leaves, int group, SIZE_t* sorted_obs):
        pass

    cdef int node_split(self, double impurity, CARTGVSplitRecord* split, SIZE_t* n_constant_features, int parent_start, int parent_end):
        pass

    cdef void node_value(self, double* dest) nogil:
        self.criterion.node_value(dest)
        #NO PROBLEM
#        with  gil:
#            print(dest[0])

    cdef double node_impurity(self) nogil:
        cdef double node_impurity = self.criterion.node_impurity()
        #NO PROBLEM
#        with gil:
#            print(node_impurity)
        return node_impurity

    ################################# TEST ##################################

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

#    cdef const DTYPE_t[:,:] X # Si définie ici, il n'est pas accessible pour les tests.

#    cdef SIZE_t n_total_samples

    cdef int init(self,
                object X,
                const DOUBLE_t[:, ::1] y,
                DOUBLE_t* sample_weight,
                object groups) except -1:

        CARTGVSplitter.init(self,X,y,sample_weight,groups)

        self.X = X

        ## ACCESS TO FIELDS TO CHECK CORRUPTION ##
        # NO PROBLEM
#        print(self.rand_r_state)
#        print(np.asarray(<SIZE_t[:self.n_samples]>self.samples))
#        print(self.n_samples)
#        print(self.weighted_n_samples)
#        print(np.asarray(<SIZE_t[:self.n_features]>self.features))
#        print(self.n_features)
#        print(np.asarray(self.X))
#        print(np.asarray(self.y))
#        if sample_weight != NULL:
#            print(np.asarray(<DOUBLE_t[:self.n_samples]>self.sample_weight))
#        print(np.asarray(self.groups))
#        print(self.n_groups)
#        print(np.asarray(self.len_groups))

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

#        print("Xf")
#        print(Xf)
#        print("X")
#        print(np.asarray(self.X))
#        print("Samples")
#        print(np.asarray(<SIZE_t[:self.n_samples]>self.samples))
#        print("group")
#        print(np.asarray(group))
#        print("start")
#        print(start)
#        print("end")
#        print(end)
        ## ACCESS TO FIELDS TO CHECK CORRUPTION ##
        # NO PROBLEM
#        print(Xf)

#        Xf.sort(axis=0)
#        print("Xf sorted")
#        print(Xf)

        return Xf

    cdef int reset_scikit_learn_instances(self, np.ndarray y, int group, int len_group):
        cdef SIZE_t k

        cdef SIZE_t n_outputs = y.shape[1]
        classes = []
        n_classes = []

        for k in range(n_outputs):
          classes_k = np.unique(y[:,k])
          classes.append(classes_k)
          n_classes.append(classes_k.shape[0])

        n_classes = np.array(n_classes, dtype=np.intp)

        self.splitting_tree = Tree(len_group, n_classes, n_outputs)

        ## ACCESS TO FIELDS TO CHECK CORRUPTION ##
        # NO PROBLEM
#        print(self.splitting_tree)

        cdef int max_features = self.mvar[group]
#        cdef SIZE_t max_leaf_nodes = -1 #self.X.shape[0]
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t max_depth = self.max_depth
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split
        cdef object random_state = check_random_state(self.random_state)

        cdef Criterion criterion = Gini(n_outputs, n_classes)

        ## ACCESS TO FIELDS TO CHECK CORRUPTION ##
        # NO PROBLEM
#        print(criterion)

        cdef Splitter splitter = BestSplitter(criterion, max_features, min_samples_leaf, min_weight_leaf, random_state)

        ## ACCESS TO FIELDS TO CHECK CORRUPTION ##
        # NO PROBLEM
#        print(splitter)

        cdef TreeBuilder depthFirstTreeBuilder = DepthFirstTreeBuilder(splitter, min_samples_split, min_samples_leaf, min_weight_leaf, max_depth, min_impurity_decrease, min_impurity_split)

        ## ACCESS TO FIELDS TO CHECK CORRUPTION ##

#        print(depthFirstTreeBuilder)

        self.splitting_tree_builder = depthFirstTreeBuilder

        return 0

    cdef int splitting_tree_construction(self, np.ndarray Xf, np.ndarray y):

        self.splitting_tree_builder.build(self.splitting_tree, Xf, y)

#        clf = DecisionTreeClassifier()

#        print(np.asarray(self.splitting_tree.value))
#        clf.tree_ = self.splitting_tree
#        plot_tree(clf)
#        plt.show()

        ## ACCESS TO FIELDS TO CHECK CORRUPTION ##
        # NO PROBLEM
#        print(self.splitting_tree)
#        print(Xf)
#        print(y)

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

    cdef int get_splitting_tree_leaves_samples_and_pos(self, SIZE_t** starts, SIZE_t** ends, Node* sorted_leaves, SIZE_t n_leaves, SIZE_t*** samples_leaves, SIZE_t n_samples):
        cdef SIZE_t previous_pos = 0
        cdef SIZE_t* tmp_sorted_sample = self.splitting_tree_builder.splitter.samples

        for j in range(n_leaves):
          if(previous_pos + sorted_leaves[j].n_node_samples <= n_samples):
            incr = 0
            for m in range(previous_pos,previous_pos+sorted_leaves[j].n_node_samples):
              samples_leaves[0][j][incr] = tmp_sorted_sample[m]
              incr += 1

          starts[0][j] = previous_pos
          ends[0][j] = previous_pos + sorted_leaves[j].n_node_samples
          previous_pos += sorted_leaves[j].n_node_samples

        return 0

    cdef int switch_best_splitting_tree(self, double current_proxy_improvement, double* best_proxy_improvement, CARTGVSplitRecord* best, CARTGVSplitRecord* current, SIZE_t* starts, SIZE_t* ends, SIZE_t n_leaves, int group, SIZE_t* sorted_obs):
        ser_splitting_tree = None

#        print("### PROXY ###")
#        print(current_proxy_improvement)
#        print(best_proxy_improvement[0])
        if current_proxy_improvement > best_proxy_improvement[0]:
          best_proxy_improvement[0] = current_proxy_improvement
          best_splitting_tree = self.splitting_tree
          ser_splitting_tree = pickle.dumps(best_splitting_tree,0)
          current[0].splitting_tree = <unsigned char*>malloc(sys.getsizeof(ser_splitting_tree)*sizeof(unsigned char))
          current[0].splitting_tree = ser_splitting_tree
          current[0].starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
          current[0].ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
          current[0].starts = starts
          current[0].ends = ends
          current[0].n_childs = n_leaves
          current[0].group = group
          best[0].splitting_tree = <unsigned char*>malloc(sys.getsizeof(ser_splitting_tree)*sizeof(unsigned char))
          for k in range(sys.getsizeof(ser_splitting_tree)):
            best[0].splitting_tree[k] = current[0].splitting_tree[k]
          best[0].starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
          best[0].ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
          for i in range(n_leaves):
            best[0].starts[i] = current[0].starts[i]
          for j in range(n_leaves):
            best[0].ends[j] = current[0].ends[j]
          best[0].n_childs = current[0].n_childs
          best[0].group = current[0].group
          for l in range(self.n_samples):
            sorted_obs[l] = self.splitting_tree_builder.splitter.samples[l]


        return 0

    cdef int node_split(self, double impurity, CARTGVSplitRecord* split, SIZE_t* n_constant_features, int parent_start, int parent_end):

        cdef SIZE_t n_visited_grouped_features = 0                      # The number of group visited
        cdef SIZE_t mgroup = self.mgroup    # The max number of group we will visit

        cdef SIZE_t start = self.start                                  # The start of the node in the sample array
        cdef SIZE_t end = self.end
        cdef SIZE_t* sorted_obs                                         # The sorted observation of the node

        cdef SIZE_t* current_obs
        cdef SIZE_t* current_samples

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

        cdef int[:] group_to_select = np.arange(len(groups))

        sorted_obs = <SIZE_t*> malloc(self.n_samples*sizeof(SIZE_t))

        current_obs = <SIZE_t*> malloc(self.n_samples*sizeof(SIZE_t))

        while(n_visited_grouped_features < mgroup):
#            print("############### LOOP " + str(n_visited_grouped_features) + " ###################")

            f_j = group_to_select[n_visited_grouped_features]
            #TODO utiliser n_visited_grouped_features ?
            #TODO Mais si max_grouped_features < nb_group alors on ne prendra que les n_visited_grouped_features premier
#            f_j = np.random.randint(0,max_grouped_features)
            #TODO Faire une différenciation entre un arbre CARTGV et un arbre CARTGV pour RFGV

            n_visited_grouped_features += 1

            print("## GROUP " + str(f_j) + " ##") #TODO Ne visite pas tout les groupes !!!

            group = groups[f_j]
            len_group = len_groups[f_j]

            Xf = self.group_sample(group, len_group, start, end)

            self.criterion.reset()

            y = np.empty((end-start,self.y.shape[1])) # Récupère la shape correcte des données
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

            samples_leaves = <SIZE_t**> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t*))
            for j in range(self.splitting_tree.n_leaves):
                samples_leaves[j] = <SIZE_t*> malloc(sorted_leaves[j].n_node_samples *sizeof(SIZE_t))
            n_samples = self.splitting_tree_builder.splitter.n_samples
            starts = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))
            ends = <SIZE_t*> malloc(n_leaves * sizeof(SIZE_t))

            self.get_splitting_tree_leaves_samples_and_pos(&starts, &ends, sorted_leaves, n_leaves, &samples_leaves, n_samples)

            for l in range(self.n_samples):
                current_obs[l] = self.splitting_tree_builder.splitter.samples[l]

            current_samples = <SIZE_t*> malloc(self.splitting_tree_builder.splitter.n_samples * sizeof(SIZE_t))
            incr = 0
            for incr in range(self.splitting_tree_builder.splitter.n_samples):
                current_samples[incr] = self.samples[parent_start+current_obs[incr]]

            self.criterion.samples = current_samples
            self.criterion.n_samples = self.splitting_tree_builder.splitter.n_samples

            self.criterion.update(starts, ends, n_leaves)

            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

#            print("Group : " + str(f_j))
#            print("Improvement : " + str(current_proxy_improvement))
#            print("Best improvement : " + str(best_proxy_improvement))

            self.switch_best_splitting_tree(current_proxy_improvement, &best_proxy_improvement, &best, &current, starts, ends, n_leaves, f_j, sorted_obs)

#        print("### SORTED OBS ###")
#        print(np.asarray(<SIZE_t[:self.splitting_tree_builder.splitter.n_samples]>sorted_obs))

#        self.samples = <SIZE_t*> malloc(self.splitting_tree_builder.splitter.n_samples*sizeof(SIZE_t))
#        self.n_samples = self.splitting_tree_builder.splitter.n_samples

        self.criterion.samples = self.samples

        sorted_samples = []
        incr = 0
        for incr in range(self.splitting_tree_builder.splitter.n_samples):
            sorted_samples.append(self.samples[parent_start+sorted_obs[incr]])

        incr = 0
        for k in range(parent_start,parent_end):
            self.samples[k] = sorted_samples[incr]
            incr+=1

        print("BEST STARTS AND ENDS")
        print(np.asarray(<SIZE_t[:best.n_childs]>best.starts))
        print(np.asarray(<SIZE_t[:best.n_childs]>best.ends))

        for s in range(best.n_childs):
            best.starts[s]+=parent_start #TODO necessary ? It is
        for e in range(best.n_childs):
            best.ends[e]+=parent_start #TODO necessary ? It is

        print("BEST STARTS AND ENDS CHANGED")
        print(np.asarray(<SIZE_t[:best.n_childs]>best.starts))
        print(np.asarray(<SIZE_t[:best.n_childs]>best.ends))
        print(parent_start)

        self.criterion.reset()

#        print("### THRESHOLD ###")
#        print(best_splitting_tree.threshold)

        n_leaves = pickle.loads(best.splitting_tree).n_leaves

        self.criterion.update(best.starts,best.ends,n_leaves)

        best.impurity_childs = <double*> malloc(n_leaves * sizeof(double))

        self.criterion.children_impurity(&best.impurity_childs) #TODO Une potentielle erreur/valeurs erronées dans cette fonction

        best.improvement = self.criterion.impurity_improvement(impurity, best.impurity_childs)

#        print(" ## BEST ##")
#        print(best.improvement)
#        print(best.n_childs)
#        print(np.asarray(<double[:best.n_childs]>best.impurity_childs))
#        print("Group : " + str(best.group))
#        print(np.asarray(<SIZE_t[:best.n_childs]>best.starts))
#        print(np.asarray(<SIZE_t[:best.n_childs]>best.ends))
#        print(best.splitting_tree)

        clf = DecisionTreeClassifier()

        clf.tree_ = pickle.loads(best.splitting_tree)
        plot_tree(clf)
        plt.show()

        split[0] = best



#        split[0].splitting_tree = <unsigned char*>malloc(sys.getsizeof(best.splitting_tree)*sizeof(unsigned char))
#        for k in range(sys.getsizeof(best.splitting_tree)):
#            split[0].splitting_tree[k] = best.splitting_tree[k]
#        split[0].starts = <SIZE_t*> malloc(best.n_childs * sizeof(SIZE_t))
#        split[0].ends = <SIZE_t*> malloc(best.n_childs * sizeof(SIZE_t))
#        for l in range(best.n_childs):
#            split[0].starts[l] = best.starts[l]
#        for m in range(best.n_childs):
#            split[0].ends[m] = best.ends[m]
#        split[0].n_childs = best.n_childs
#        split[0].group = best.group
#        split[0].improvement = best.improvement
#        split[0].impurity_childs = <double*> malloc(best.n_childs * sizeof(double))
#        for n in range(best.n_childs):
#            split[0].impurity_childs[n] = best.impurity_childs[n]

#        print("## SPLIT ##")
#        print(split[0].improvement)
#        print(split[0].n_childs)
#        print(np.asarray(<double[:split[0].n_childs]>split[0].impurity_childs))
#        print(np.asarray(<SIZE_t[:split[0].n_childs]>split[0].starts))
#        print(np.asarray(<SIZE_t[:split[0].n_childs]>split[0].ends))
#        print(split[0].splitting_tree)

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

        # NO PROBLEM
#        print("############################# GET SPLITTING TREE LEAVES ###########################")
#        for i in range(self.splitting_tree.n_leaves):
#            print("### LEAF " + str(i) + " ###")
#            print(sorted_leaves[i].left_child)
#            print(sorted_leaves[i].right_child)
#            print(sorted_leaves[i].feature)
#            print(sorted_leaves[i].threshold)
#            print(sorted_leaves[i].impurity)
#            print(sorted_leaves[i].n_node_samples)
#            print(sorted_leaves[i].weighted_n_node_samples)

        return 0

    cpdef int test_get_splitting_tree_leaves_samples_and_pos(self):

        cdef Node* sorted_leaves = <Node*> malloc(self.splitting_tree.n_leaves*sizeof(Node))
        self.get_splitting_tree_leaves(&sorted_leaves)

        cdef SIZE_t** samples_leaves = <SIZE_t**> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t*)) #* sizeof(SIZE_t*)
        for j in range(self.splitting_tree.n_leaves):
            samples_leaves[j] = <SIZE_t*> malloc(sorted_leaves[j].n_node_samples *sizeof(SIZE_t)) #*sizeof(SIZE_t)

        cdef SIZE_t n_samples = self.splitting_tree_builder.splitter.n_samples
        cdef SIZE_t* starts = <SIZE_t*> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t))
        cdef SIZE_t* ends = <SIZE_t*> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t))

        res = self.get_splitting_tree_leaves_samples_and_pos(&starts, &ends, sorted_leaves, self.splitting_tree.n_leaves, &samples_leaves, n_samples)

        #NO PROBLEM
#        print("################################ TEST GET SPLITTING TREE LEAVES SAMPLES AND POS #######################################")
#        print(n_samples)
#        for i in range(self.splitting_tree.n_leaves):
#            print("### LEAF " + str(i) + " ###")
#            print(starts[i])
#            print(ends[i])
#            print(n_samples)
#            print(np.asarray(<SIZE_t[:ends[i]-starts[i]]>samples_leaves[i]))
#            print(ends[i] - starts[i])
#            print(len(np.asarray(<SIZE_t[:ends[i]-starts[i]]>samples_leaves[i])))

    cpdef int test_switch_best_splitting_tree(self):
        cdef double current_proxy_improvement, best_proxy_improvement
        current_proxy_improvement = 1.0
        best_proxy_improvement = 0.0
        cdef CARTGVSplitRecord best, current
        cdef SIZE_t* sorted_obs

        cdef Node* sorted_leaves = <Node*> malloc(self.splitting_tree.n_leaves*sizeof(Node))
        self.get_splitting_tree_leaves(&sorted_leaves)

        cdef SIZE_t** samples_leaves = <SIZE_t**> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t*)) #* sizeof(SIZE_t*)
        for j in range(self.splitting_tree.n_leaves):
            samples_leaves[j] = <SIZE_t*> malloc(sorted_leaves[j].n_node_samples * sizeof(SIZE_t)) #*sizeof(SIZE_t)

        cdef SIZE_t n_samples = self.splitting_tree_builder.splitter.n_samples
        cdef SIZE_t* starts = <SIZE_t*> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t))
        cdef SIZE_t* ends = <SIZE_t*> malloc(self.splitting_tree.n_leaves * sizeof(SIZE_t))

        self.get_splitting_tree_leaves_samples_and_pos(&starts, &ends, sorted_leaves, self.splitting_tree.n_leaves, &samples_leaves, n_samples)

        res = self.switch_best_splitting_tree(current_proxy_improvement, &best_proxy_improvement, &best, &current, starts, ends, self.splitting_tree.n_leaves,0, sorted_obs)
#TODO Faire une boucle pour test
#        print("##### CURRENT #####")
#        print(current_proxy_improvement)
#        print(current.improvement)
#        print(current.n_childs)
#        print(np.asarray(<double[:current.n_childs]>current.impurity_childs))
#        print(np.asarray(<SIZE_t[:current.n_childs]>current.starts))
#        print(np.asarray(<SIZE_t[:current.n_childs]>current.ends))
#        print(current.splitting_tree)
#
#        print("##### BEST #####")
#        print(best_proxy_improvement)
#        print(best.improvement)
#        print(best.n_childs)
#        print(np.asarray(<double[:best.n_childs]>best.impurity_childs))
#        print(np.asarray(<SIZE_t[:best.n_childs]>best.starts))
#        print(np.asarray(<SIZE_t[:best.n_childs]>best.ends))
#        print(best.splitting_tree)
#
#        print("#### SORTED SAMPLES ####")
#        print(np.asarray(<SIZE_t[:self.splitting_tree_builder.splitter.n_samples]> sorted_obs))

        return 0

    cpdef int test_node_split(self):

        cdef impurity = np.inf
        cdef CARTGVSplitRecord split
        cdef SIZE_t n_constant_features
        cdef int parent_start = 0
        cdef int parent_end = 10 #TODO refaire le test

        self.node_split(impurity, &split, &n_constant_features, parent_start, parent_end)

#        print("################################ TEST NODE SPLIT #######################################")
#        print(split.n_childs)
#        print(split.splitting_tree)
#        print(split.improvement)
#        print(split.group)
#        print(np.asarray(<double[:split.n_childs]>split.impurity_childs))
#        print(np.asarray(<SIZE_t[:split.n_childs]>split.starts))
#        print(np.asarray(<SIZE_t[:split.n_childs]>split.ends))