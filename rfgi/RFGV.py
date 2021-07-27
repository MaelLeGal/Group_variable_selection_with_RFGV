import numbers
from warnings import catch_warnings, simplefilter, warn
import threading

from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack
from joblib import Parallel
import time
import pickle

from sklearn.base import ClassifierMixin, RegressorMixin, MultiOutputMixin
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from CARTGV_trees import (DecisionCARTGVTreeClassifier, DecisionCARTGVTreeRegressor)
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.ensemble._forest import BaseForest, ForestClassifier, ForestRegressor
from sklearn.utils.fixes import delayed
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.metrics import check_scoring
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from joblib import wrap_non_picklable_objects




__all__ = ["RFGVClassifier",
           "RFGVRegressor"]

MAX_INT = np.iinfo(np.int32).max


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.
    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.
    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    # print("Start get n samples bootstrap")
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return round(n_samples * max_samples)

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to _parallel_build_trees function.
    params random_state : An int, the seed used to set the randomness
    params n_samples : An int, the number of samples
    params n_samples_bootstrap : An int, the number of samples taken for the bootstrap sample

    outputs : An array of indices, used to get the in-bag sample to create a tree
    """
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to forest._set_oob_score function.
    params random_state : An int, the seed used to set the randomness
    params n_samples : An int, the number of samples
    params n_samples_bootstrap : An int, the number of samples taken for the bootstrap sample and therefore the unsampled indices

    outputs : An array of indices, used to get the out-of-bag sample of a tree
    """
    sample_indices = _generate_sample_indices(random_state, n_samples,
                                              n_samples_bootstrap)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]
    return unsampled_indices


def _parallel_build_trees(tree, forest, X, y, groups, sample_weight, tree_idx, n_trees,
                          verbose=0, class_weight=None,
                          n_samples_bootstrap=None):
    """
    Private function used to fit a single tree in parallel.
    params tree : A instance of the DecisionCARTGVTree, the tree that will be built
    params forest : An instance of the RFGVBaseForest, the forest the tree will be a part of
    params X : A numpy array or pandas DataFrame, The datas use to build the tree
    params y : A numpy array or pandas DataFrame, the responses of the datas
    params groups : A 2D array containing arrays for each group containing the index of the group variables
    params sample_weight : An array, The weight of the samples
    params tree_idx : An int, the id of the tree
    params n_trees : An int, the number of trees that will be built
    params verbose : An int, verbose > 1 will print the advancement of the building process
    params class_weight : A dict, list of dict, or "balanced", The class associated weights
    params n_samples_boostrap : An int, the number of samples to take in the bootstrap sample

    outputs : An instance of DecisionCARTGVTree, used to create the tree in the forest
    """
    # print("Start parallel_build_trees")
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if forest.random == False:
        forest.bootstrap = False

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(tree.random_state, n_samples,
                                           n_samples_bootstrap)

        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == 'subsample':
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y,
                                                            indices=indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y,
                                                        indices=indices)
        # start = time.time()
        tree.fit(X, y, groups, sample_weight=curr_sample_weight, check_input=False)
        # end = time.time()
        # print("Temps construction arbre : " + str(end-start) + " secondes")
    else:
        # start = time.time()
        tree.fit(X, y, groups, sample_weight=sample_weight, check_input=False)
        # end = time.time()
        # print("Temps construction arbre : " + str(end - start) + " secondes")
    # print("End of tree construction n°"+str(tree_idx+1))
    # print(str(((tree_idx+1)/n_trees)*100)+"% Done")
    return tree

def _set_random_states(estimator, random_state=None, random=True):
    """Set fixed random_state parameters for an estimator.
    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.
    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.
    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:
        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    # random_state = check_random_state(random_state) #TODO remettre
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == 'random_state' or key.endswith('__random_state'):
            if random:
                random_state = check_random_state(random_state)
                to_set[key] = random_state.randint(np.iinfo(np.int32).max) #TODO remettre
            else:
                to_set[key] = random_state

    if to_set:
        estimator.set_params(**to_set)


class RFGVBaseForest():
    """
        Base class for forests of trees.
        Warning: This class should not be used directly. Use derived classes
        instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100, *,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 ib_score=False,
                 group_importance=None,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 max_samples=None,
                 random=True):
        """
        The constructor of the RFGVBaseForest
        params base_estimator : A DecisionCARTGVTree, a template of the trees
        params n_estimators : An int, the number of trees that will be created
        params estimator_params : A tuple containing the parameters for the base estimator construction
        params bootstrap : A boolean, If true the forest will use the bootstrap method
        params oob_score : A boolean, If true the forest will compute the out-of-bag score
        params ib_score : A boolean, If true, the forest will compute the in-bag score
        params group_importance : Not used.
        params n_jobs : An int, the number of process, threads that will be used to create the forest
        params random_state : An int, the seed use to fix the randomness
        params verbose : An int, If verbose > 0 It will print the stepts of the forest
        params warm_start : A boolean, Reuse the solution of the previous fit and add more estimators
        params class_weight : A dict, list of dict, or "balanced", The class associated weights
        params max_samples : An int, The number of samples to draw from the data to train each estimator
        params random : A boolean, Remove the randomness in the forest

        outputs : An instance of RFGVBaseForest
        """

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.ib_score = ib_score
        self.group_importance=group_importance
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples
        self.random = random

    def fit(self, X, y, groups, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )
        # print(X)
        # print(y)
        X, y = self._validate_data(X, y, multi_output=True,
                                   accept_sparse="csc", dtype=DTYPE)

        #TODO Check groups
        # print(X)
        # print(y)

        self.X = X
        # self.y = y
        self.groups = groups

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        self.sample_weight = sample_weight

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        self.y = y
        # print(y)
        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0],
            max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            if self.random :
                trees = [self._make_estimator(append=False,
                                              random_state=random_state) #TODO remettre juste random_state
                         for i in range(n_more_estimators)]
            else:
                trees = [self._make_estimator(append=False,
                                              random_state=self.random_state)
                         for i in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_build_trees)(
                    t, self, X, y, groups, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        if self.ib_score:
            self._set_ib_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        # if self.group_importance != None:
        #         self.importances = self.permutation_importance(X, y, groups, self.group_importance, random_state=self.random_state)

        return self

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p)
                                for p in self.estimator_params})

        if random_state is not None:
            _set_random_states(estimator, random_state, self.random)
        if append:
            self.estimators_.append(estimator)

        return estimator

    @abstractmethod
    def _set_ib_score(self,X,y):
        """
        Compute in bag predictions and score.

        outputs : A float, the in-bag score of the forest
        """

    def _weights_scorer(self, scorer, estimator, X, y, sample_weight):
        """
        Compute the score of the estimator based on the datas and responses
        params scorer : A scoring function
        params estimator : An estimator, it needs to have a scorring function
        params X : An array or matrix, the testing data
        params y : An array or matrix, the testing responses
        params sample_weight : An array, the weight of each sample in the testing data

        outputs : A float, the tree score of one of the tree (estimator)
        """
        # print(self)
        # print(estimator)
        if sample_weight is not None:
            return scorer(estimator, X, y, sample_weight)
        return scorer(estimator, X, y)

    # @wrap_non_picklable_objects
    def _permutation_importance_Breiman(self, X, y, sample_weight, group_idx, random_state, n_repeats, scorer, estimator):
        """
        Compute the group importance by permutation following the Breiman method
        params X: An array or matrix, the out-of-bag data of the estimator
        params y: An array of matrix, the out-of-bag responses associated to the out-of-bag data of the estimator
        params sample_weight: An array, the weight of each sample in the oob sample
        params group_idx: An int, the index of the group being tested
        params random_state: An int, the seed to fix the randomness
        params n_repeats : An int, the number of times the process will be done
        params scorer : A scoring function, A scoring function most of the time the one from the estimator
        params estimator: An estimator, A tree of the forest

        outputs : An array of float, the scores of the estimator in the n_rounds
        """

        random_state = check_random_state(random_state)

        # if not hasattr(X,"iloc"):
        #     X = pd.DataFrame(X)

        # Swap the values of the group variables between observations while keeping the link between the variables of the group
        X_permuted = X.copy()
        scores = np.zeros(n_repeats)
        shuffling_idx = np.arange(X.shape[0])
        for n_round in range(n_repeats):
            random_state.shuffle(shuffling_idx)
            col = X_permuted.iloc[shuffling_idx, group_idx]
            col.index = X_permuted.index
            X_permuted.iloc[:, group_idx] = col

            feature_score = self._weights_scorer(
                scorer, estimator, X_permuted, y, sample_weight
            )

            scores[n_round] = feature_score

        return scores


    def _permutation_importance_Ishwaran(self,X_permuted,sample_weight, n_repeats, scorer, random_state, estimator):
        """
        Compute the group importance by permutation following the Ishwaran method
        params X_permuted: An array or matrix, the out-of-bag data of the estimator already permuted
        params sample_weight: An array, the weight of each sample in the oob sample
        params n_repeats : An int, the number of times the process will be done
        params scorer : A scoring function, A scoring function most of the time the one from the estimator
        params random_state: An int, the seed to fix the randomness
        params estimator: An estimator, A tree of the forest

        outputs : An array of float, the scores of the estimator in the n_round
        """
        n_samples = self.y.shape[0]

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, self.max_samples
        )

        unsampled_indices = _generate_unsampled_indices(random_state, n_samples,
                                                        n_samples_bootstrap)  # was estimators.random_state

        if hasattr(X_permuted, "iloc"):
            X_permuted_oob = X_permuted.iloc[unsampled_indices]
        else:
            X_permuted_oob = X_permuted[unsampled_indices]

        if hasattr(self.y, "iloc"):
            y_oob = self.y.iloc[unsampled_indices]
        else:
            y_oob = self.y[unsampled_indices]

        scores = np.zeros(n_repeats)

        for n_round in range(n_repeats):

            feature_score_permuted = self._weights_scorer(
                scorer, estimator, X_permuted_oob, y_oob, sample_weight
            )

            scores[n_round] = feature_score_permuted

        return scores



    def _permutation_importance(self, importance = "breiman", n_jobs = 1, n_repeats = 5):
        """
        Compute the permutation importances of groups. Will use either the breiman or ishwaran method.
        params importance : A string, either "breiman" or "ishwaran", select the method that will be used to compute the importances
        params n_jobs : An int, the number of processes or threads used to compute (Can't be above 1 at the moment)
        params n_repeats : An int, the number of time the process will be done

        outputs : A dictionary containing the mean, deviation and all the values of the importances
        """

        n_samples = self.y.shape[0]

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, self.max_samples
        )

        if not hasattr(self.X,"iloc"):
            X = pd.DataFrame(self.X)

        scores = []
        if importance == "breiman":

            # Loop on all trees of the forest
            for estimator in self.estimators_:

                # Get the estimator oob sample
                unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples_bootstrap) #was estimators.random_state

                # if hasattr(self.X,"iloc"):
                #     X_oob = self.X.iloc[unsampled_indices]
                # else:
                #     X_oob = self.X[unsampled_indices]
                #
                X_oob = X.iloc[unsampled_indices]
                if hasattr(self.y,"iloc"):
                    y_oob = self.y.iloc[unsampled_indices]
                else:
                    y_oob = self.y[unsampled_indices]

                # Check if the estimator has a correct score function
                scorer = check_scoring(estimator, scoring=None) #was self instead of estimator
                baseline_score = self._weights_scorer(scorer, estimator, self.X, self.y, None) #TODO Est ce que je dois mettre les oob ? Même résultats

                # try:
                #     print("Pickle")
                #     est = pickle.dumps(estimator)
                #     # print(est)
                #     pickle.loads(est)
                #     print("Estimator is pickleable")
                # except:
                #     print("Estimator not pickleable")
                #     ValueError("Estimator not pickleable")

                # start = time.time()
                #Launch the function _permutation_importance_Breiman on each group with the oob sample
                score = Parallel(n_jobs=n_jobs)(delayed(self._permutation_importance_Breiman)(
                    X_oob, y_oob, self.sample_weight, group_idx, estimator.random_state, n_repeats, scorer, estimator
                ) for group_idx in self.groups)
                # end = time.time()
                # print("Time score  : " + str(end-start))

                scores.append(score)


        elif importance == "ishwaran":

            random_state = check_random_state(self.random_state)

            if not hasattr(self.X, "iloc"):
                X = pd.DataFrame(self.X)

            # start = time.time()

            # Create a copy of the original datas for each group with the group swapped
            X_permuted_array = []
            shuffling_idx = np.arange(X.shape[0])
            for group_idx in self.groups:
                X_permuted = X.copy()
                random_state.shuffle(shuffling_idx)
                cols = X_permuted.iloc[shuffling_idx, group_idx]
                cols.index = X_permuted.index
                X_permuted.iloc[:, group_idx] = cols
                X_permuted_array.append(X_permuted)
            # end = time.time()
            # print("Time Permutation Ishwaran : " + str(end-start))

            # Loop on all trees of the forest
            for estimator in self.estimators_:

                scorer = check_scoring(estimator, scoring=None)  # was self instead of estimator
                baseline_score = self._weights_scorer(scorer, estimator, self.X, self.y, None)

                # Launch the _permutation_importance_Ishwaran function for each swapped sample of observations
                # to compute the oob error for the swapped group
                score = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(self._permutation_importance_Ishwaran)(
                    X_permuted, self.sample_weight, n_repeats, scorer, estimator.random_state, estimator
                ) for X_permuted in X_permuted_array)

                # OOB ERROR

                # Get the oob sample
                unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples,
                                                                n_samples_bootstrap)  # was estimators.random_state

                if hasattr(X, "iloc"):
                    X_oob = X.iloc[unsampled_indices]
                else:
                    X_oob = X[unsampled_indices]

                if hasattr(self.y, "iloc"):
                    y_oob = self.y.iloc[unsampled_indices]
                else:
                    y_oob = self.y[unsampled_indices]

                # Compute the oob errors
                oob_errors = []
                for n_round in range(n_repeats):
                    oob_error = estimator.score(X_oob, y_oob)
                    oob_errors.append(oob_error)

                scores.append(score)

        else:
            print("Does not exist")
            return -1

        importances = baseline_score - np.mean(scores, axis=0)

        return Bunch(importances_mean=np.mean(importances, axis=1),
                     importances_std=np.std(importances, axis=1),
                     importances=importances)


class RFGVClassifier(RFGVBaseForest, ForestClassifier):
    """
    Classification forest for group variables
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100, *,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 max_depth_splitting_tree=2,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 mvar="root",
                 mgroup=None,
                 pen=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 ib_score=False,
                 group_importance=None,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 random=True):
        """
        The constructor of the RFGVClassifier
        params n_estimators : An int, the number of trees that will be created
        params criterion : A string, the criterion name that will be used to construct the forest
        params splitter : A string, the splitter name that will be used to construct the forest
        max_depth : An int, the maximum depth of the trees
        max_depth_splitting_tree : An int, the maximal depth of the splitting trees
        params min_samples_split : An int, the minimal number of samples in a node needed to split it
        params min_samples_leaf : An int, the minimal number under which the nodes are considered leaves
        params min_weight_fraction_leaf : An int, the minimal weigth in a node under which it is considered a leaf
        params max_features : An int, Not used anymore
        params mvar : An string or array/list, The number of variable that will be used to construct the splitting trees for each group
        params mgroup : An int, the number of group that will be tested to find the best splitting tree
        params pen : A function with one parameter or a string, The penality function on the group size for the impurity calculation
        params random_state : An int, the seed to fix the randomness
        params max_leaf_nodes : An int, the maximum number of leaf the tree will be restricted to
        params min_impurity_decrease : A float, The value under which the decrease in impurity of a split need to be to split a node
        params min_impurity_split : A float, The minimal value of impurity under which the node is considered a leaf
        params bootstrap : A boolean, If true the forest will use the bootstrap method
        params oob_score : A boolean, If true the forest will compute the out-of-bag score
        params ib_score : A boolean, If true, the forest will compute the in-bag score
        params group_importance : Not used.
        params n_jobs : An int, the number of process, threads that will be used to create the forest
        params random_state : An int, the seed use to fix the randomness
        params verbose : An int, If verbose > ... It will print the stepts of the forest
        params verbose : An int, If verbose > 0 It will print the stepts of the forest
        params warm_start : A boolean, Reuse the solution of the previous fit and add more estimators
        params class_weight : A dict, list of dict, or "balanced", The class associated weights
        params max_samples : An int, The number of samples to draw from the data to train each estimator
        params random : A boolean, Remove the randomness in the forest
        params ccp_alpha : A non-negative float, the complexity parameter used for tree pruning

        outputs : An instance of RFGVClassifier
        """
        super().__init__(
            base_estimator=DecisionCARTGVTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "splitter", "max_depth", "max_depth_splitting_tree",
                              "min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "mvar", "mgroup", "pen", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state", "ccp_alpha"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            ib_score=ib_score,
            group_importance=group_importance,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
            random=random)

        self.criterion = criterion
        self.splitter = splitter,
        self.max_depth = max_depth
        self.max_depth_splitting_tree = max_depth_splitting_tree
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.mvar = mvar
        self.mgroup = mgroup
        self.pen = pen
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha

    def _set_ib_score(self, X, y):
        """
        Compute the in-bag score
        params X : An array or matrix, the data used to construct the forest
        params y : An array or matrix, the responses used to construct the forest

        outputs : A float, the in-bag score of the forest
        """
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        ib_decision_function = []
        ib_score = 0.0
        predictions = [np.zeros((n_samples, n_classes_[k]))
                       for k in range(self.n_outputs_)]

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, self.max_samples
        )

        for estimator in self.estimators_:
            sampled_indices = _generate_sample_indices(estimator.random_state, n_samples, n_samples_bootstrap)
            p_estimator = estimator.predict_proba(X[sampled_indices, :],
                                                  check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][sampled_indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn("Some inputs do not have OOB scores. "
                     "This probably means too few trees were used "
                     "to compute any reliable oob estimates.")

            decision = (predictions[k] /
                        predictions[k].sum(axis=1)[:, np.newaxis])
            ib_decision_function.append(decision)
            ib_score += np.mean(y[:, k] ==
                                 np.argmax(predictions[k], axis=1), axis=0)

        if self.n_outputs_ == 1:
            self.ib_decision_function_ = ib_decision_function[0]
        else:
            self.ib_decision_function_ = ib_decision_function

        self.ib_score_ = ib_score / self.n_outputs_

class RFGVRegressor(RFGVBaseForest, ForestRegressor):
    """
        Regression forest for group variables
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100, *,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 max_depth_splitting_tree=2,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 mvar="third",
                 mgroup=None,
                 pen=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 ib_score=False,
                 group_importance=None,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None,
                 random=True):
        """
        The constructor of the RFGVRegressor
        params n_estimators : An int, the number of trees that will be created
        params criterion : A string, the criterion name that will be used to construct the forest
        params splitter : A string, the splitter name that will be used to construct the forest
        max_depth : An int, the maximum depth of the trees
        max_depth_splitting_tree : An int, the maximal depth of the splitting trees
        params min_samples_split : An int, the minimal number of samples in a node needed to split it
        params min_samples_leaf : An int, the minimal number under which the nodes are considered leaves
        params min_weight_fraction_leaf : An int, the minimal weigth in a node under which it is considered a leaf
        params max_features : An int, Not used anymore
        params mvar : An string or array/list, The number of variable that will be used to construct the splitting trees for each group
        params mgroup : An int, the number of group that will be tested to find the best splitting tree
        params pen : A function with one parameter or a string, The penality function on the group size for the impurity calculation
        params random_state : An int, the seed to fix the randomness
        params max_leaf_nodes : An int, the maximum number of leaf the tree will be restricted to
        params min_impurity_decrease : A float, The value under which the decrease in impurity of a split need to be to split a node
        params min_impurity_split : A float, The minimal value of impurity under which the node is considered a leaf
        params bootstrap : A boolean, If true the forest will use the bootstrap method
        params oob_score : A boolean, If true the forest will compute the out-of-bag score
        params ib_score : A boolean, If true, the forest will compute the in-bag score
        params group_importance : Not used.
        params n_jobs : An int, the number of process, threads that will be used to create the forest
        params random_state : An int, the seed use to fix the randomness
        params verbose : An int, If verbose > 0 It will print the stepts of the forest
        params warm_start : A boolean, Reuse the solution of the previous fit and add more estimators
        params class_weight : A dict, list of dict, or "balanced", The class associated weights
        params max_samples : An int, The number of samples to draw from the data to train each estimator
        params random : A boolean, Remove the randomness in the forest
        params ccp_alpha : A non-negative float, the complexity parameter used for tree pruning

        outputs : An instance of RFGVRegressor
        """
        super().__init__(
            base_estimator=DecisionCARTGVTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "splitter", "max_depth", "max_depth_splitting_tree",
                              "min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "mvar", "mgroup", "pen", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state", "ccp_alpha"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            ib_score=ib_score,
            group_importance=group_importance,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
            random=random)

        self.criterion = criterion
        self.splitter = splitter,
        self.max_depth = max_depth
        self.max_depth_splitting_tree = max_depth_splitting_tree
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.mvar = mvar
        self.mgroup = mgroup
        self.pen = pen
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha

    def _set_ib_score(self, X, y):
        """
        Compute in-bag scores.
        params X : An array or matrix, the data used to construct the forest
        params y : An array or matrix, the responses used to construct the forest

        outputs : A float, the in-bag score of the forest
        """
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, self.max_samples
        )

        for estimator in self.estimators_:
            sampled_indices = _generate_sample_indices(
                estimator.random_state, n_samples, n_samples_bootstrap)
            p_estimator = estimator.predict(
                X[sampled_indices, :], check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = p_estimator[:, np.newaxis]

            predictions[sampled_indices, :] += p_estimator
            n_predictions[sampled_indices, :] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.ib_prediction_ = predictions

        if self.n_outputs_ == 1:
            self.ib_prediction_ = \
                self.ib_prediction_.reshape((n_samples,))

        self.ib_score_ = 0.0

        for k in range(self.n_outputs_):
            self.ib_score_ += r2_score(y[:, k],
                                        predictions[:, k])

        self.ib_score_ /= self.n_outputs_