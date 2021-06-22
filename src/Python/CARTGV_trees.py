import numbers
import warnings
import copy
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.tree._classes import BaseDecisionTree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import clone
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier
from sklearn.base import MultiOutputMixin
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args

from CARTGV import CARTGVTree, CARTGVTreeBuilder
from CARTGV import CARTGVSplitter, BaseDenseCARTGVSplitter, BestCARTGVSplitter
from CARTGV import CARTGVCriterion, CARTGVClassificationCriterion, CARTGVGini, CARTGVMSE

from sklearn.tree import _tree, _splitter, _criterion

__all__ = ["DecisionCARTGVTreeClassifier"]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {"gini": CARTGVGini}
CRITERIA_REG = {"mse": CARTGVMSE}

DENSE_SPLITTERS = {"best": BestCARTGVSplitter}

class DecisionCARTGVTree(BaseDecisionTree):

    def __init__(self, *,
                 criterion,
                 splitter,
                 max_depth,
                 max_depth_splitting_tree,
                 min_samples_split,
                 min_samples_leaf,
                 min_weight_fraction_leaf,
                 max_features,
                 mvar,
                 mgroup,
                 pen,
                 random_state,
                 max_leaf_nodes,
                 min_impurity_decrease,
                 min_impurity_split,
                 class_weight=None,
                 ccp_alpha):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)

        self.max_depth_splitting_tree = max_depth_splitting_tree
        self.mvar = mvar
        self.mgroup = mgroup
        self.pen = pen

    def fit(self, X, y, groups, sample_weight=None, check_input=True):

        random_state = check_random_state(self.random_state)

        if self.ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be greater than or equal to 0")

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(X, y,
                                       validate_separately=(check_X_params,
                                                            check_y_params))

            #TODO Check groups

            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError("Some value(s) of y are negative which is"
                                     " not allowed for Poisson regression.")
                if np.sum(y) <= 0:
                    raise ValueError("Sum of y is not positive which is "
                                     "necessary for Poisson regression.")

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        self.n_features_in_ = self.n_features_
        is_classification = is_classifier(self)

        len_groups = np.array([len(group) for group in groups])
        groups = np.array(list(map(lambda group: np.pad(group,(0,max(len_groups) - len(group)), constant_values=-1), groups)))

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k],
                                                       return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original)

            self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features. "
                                 "Allowed string values are 'auto', "
                                 "'sqrt' or 'log2'.")
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, numbers.Integral):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either None "
                              "or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        min_impurity_split = self.min_impurity_split
        if min_impurity_split is not None:
            warnings.warn(
                "The min_impurity_split parameter is deprecated. Its default "
                "value has changed from 1e-7 to 0 in version 0.23, and it "
                "will be removed in 1.0 (renaming of 0.25). Use the "
                "min_impurity_decrease parameter instead.",
                FutureWarning
            )

            if min_impurity_split < 0.:
                raise ValueError("min_impurity_split must be greater than "
                                 "or equal to 0")
        else:
            min_impurity_split = 0

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, CARTGVCriterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
                                                         self.n_classes_)
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_,
                                                         n_samples)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        # TODO mettre un paramètre dans le constructeur ou le fit pour le critère de l'arbre de coupure ?
        split_criterion =""
        if(isinstance(criterion, CARTGVGini)):
            split_criterion = "gini"
        elif(isinstance(criterion,CARTGVMSE)):
            split_criterion = "mse"

        # SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        if(type(self.splitter) is tuple):
            self.splitter = self.splitter[0]

        splitter = self.splitter

        if not isinstance(self.splitter, CARTGVSplitter):
            splitter = DENSE_SPLITTERS[self.splitter](criterion,
                                                len(groups),
                                                min_samples_leaf,
                                                min_weight_leaf,
                                                random_state,
                                                self.max_depth_splitting_tree,
                                                0.0,
                                                0.0,
                                                self.mvar,
                                                self.mgroup,
                                                split_criterion)

        if is_classifier(self):
            self.tree_ = CARTGVTree(len(groups), len_groups, groups, self.n_features_,
                              self.n_classes_, self.n_outputs_)
        else:
            self.tree_ = CARTGVTree(len(groups), len_groups, groups, self.n_features_,
                              # TODO: tree should't need this in this case
                              np.array([1] * self.n_outputs_, dtype=np.intp),
                              self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0: # Was DepthFirstTreeBuilder
            builder = CARTGVTreeBuilder(splitter, min_samples_split,
                                            min_samples_leaf,
                                            min_weight_leaf,
                                            max_depth,
                                            self.min_impurity_decrease,
                                            min_impurity_split)
        else: # Was BestFirstTreeBuilder
            builder = CARTGVTreeBuilder(splitter, min_samples_split,
                                           min_samples_leaf,
                                           min_weight_leaf,
                                           max_depth,
                                           max_leaf_nodes,
                                           self.min_impurity_decrease,
                                           min_impurity_split)


        builder.build(self.tree_, X, y, groups, len_groups, self.pen, sample_weight)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        # self._prune_tree()

        return self

class DecisionCARTGVTreeClassifier(DecisionCARTGVTree,DecisionTreeClassifier):

    def __init__(self, *,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 max_depth_splitting_tree=2,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 mvar = None,
                 mgroup=None,
                 pen=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 ccp_alpha=0.0):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            max_depth_splitting_tree=max_depth_splitting_tree,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            mvar=mvar,
            mgroup=mgroup,
            pen=pen,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)

    def fit(self, X, y, groups, sample_weight=None, check_input=True):
        super().fit(
            X, y,
            groups,
            sample_weight=sample_weight,
            check_input=check_input)
        return self

class DecisionCARTGVTreeRegressor(DecisionCARTGVTree, RegressorMixin):

    def __init__(self, *,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 max_depth_splitting_tree=2,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 mvar=None,
                 mgroup=None,
                 pen=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 ccp_alpha=0.0):

        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            max_depth_splitting_tree=max_depth_splitting_tree,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            mvar=mvar,
            mgroup=mgroup,
            pen=pen,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)

    def fit(self, X, y, groups, sample_weight=None, check_input=True):
        super().fit(
            X, y,
            groups,
            sample_weight=sample_weight,
            check_input=check_input)
        return self
