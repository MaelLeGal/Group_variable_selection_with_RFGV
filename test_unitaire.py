import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pickle
import sys

from CARTGV import CARTGVSplitter
from CARTGV import CARTGVGini
from CARTGV import CARTGVTree, CARTGVTreeBuilder

from sklearn.utils.validation import check_random_state
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.tree._tree import DepthFirstTreeBuilder, BestFirstTreeBuilder, Tree
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._criterion import Gini, Entropy

from scipy.sparse import issparse
from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE


# tree = CARTGVTree(n_grouped_features, n_classes, n_outputs)

# builder = CARTGVTreeBuilder(splitter, min_samples_split,
#                             min_samples_leaf, min_weight_leaf,
#                             max_depth, mgroup, mvar,
#                             min_impurity_decrease, min_impurity_split)

class CARTGVCriterionTest(unittest.TestCase):

    def test_init(self):
        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])

        n_samples, n_features = X.shape
        n_grouped_features = 2
        y = np.atleast_1d(y)
        max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = 0  # (0.25 * n_samples)
        random_state = check_random_state(2457)
        max_depth = 3
        mgroup = 1
        mvar = 10
        min_impurity_decrease = 0.1
        min_impurity_split = 0.0

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        n_outputs = y.shape[1]

        y = np.copy(y)

        classes = []
        n_classes = []

        y_encoded = np.zeros(y.shape, dtype=np.float64)
        for k in range(n_outputs):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        y = y_encoded

        n_classes = np.array(n_classes, dtype=np.intp)

        criterion = CARTGVGini(n_outputs, n_classes)

        sample_weight = None
        weighted_n_samples = X.shape[0]
        samples = np.arange(X.shape[0], dtype=np.intp)
        start = np.intp(0)
        end = np.intp(X.shape[0] - 1)
        criterion.test_init(y, sample_weight, weighted_n_samples, samples, start, end)

        self.assertEqual(np.array(criterion.y).all(), y.all())
        self.assertEqual(criterion.sample_weight, sample_weight)
        # self.assertEqual(criterion.starts, [start]) #Identique en Cython mais je n'arrive pas à obtenir criterion.starts correctement avec le bon type
        # self.assertEqual(criterion.ends, [end]) #Identique en Cython mais je n'arrive pas à obtenir criterion.ends correctement avec le bon type
        self.assertEqual(criterion.n_node_samples, end - start)
        self.assertEqual(criterion.weighted_n_samples, weighted_n_samples)
        self.assertEqual(criterion.weighted_n_node_samples, np.sum(np.ones(X.shape[0] - 1)))

    def test_criterion_reset(self):
        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        y = np.atleast_1d(y)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        n_outputs = y.shape[1]

        y = np.copy(y)

        classes = []
        n_classes = []

        y_encoded = np.zeros(y.shape, dtype=np.float64)
        for k in range(n_outputs):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        y = y_encoded

        n_classes = np.array(n_classes, dtype=np.intp)

        criterion = CARTGVGini(n_outputs, n_classes)

        sample_weight = None
        weighted_n_samples = X.shape[0]
        samples = np.arange(X.shape[0], dtype=np.intp)
        start = np.intp(0)
        end = np.intp(X.shape[0] - 1)
        criterion.test_init(y, sample_weight, weighted_n_samples, samples, start, end)
        criterion.test_reset()

class CARTGVSplitterTest(unittest.TestCase):

    def test_init(self):

        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])

        n_samples, n_features = X.shape
        n_grouped_features = 2
        y = np.atleast_1d(y)
        max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = 0  # (0.25 * n_samples)
        random_state = check_random_state(2457)
        max_depth = 3
        mgroup = 1
        mvar = 10
        min_impurity_decrease = 0.1
        min_impurity_split = 0.0

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        n_outputs = y.shape[1]

        y = np.copy(y)

        classes = []
        n_classes = []

        y_encoded = np.zeros(y.shape, dtype=np.float64)
        for k in range(n_outputs):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        y = y_encoded

        n_classes = np.array(n_classes, dtype=np.intp)

        criterion = CARTGVGini(n_outputs, n_classes)

        splitter = CARTGVSplitter(criterion, max_grouped_features, len(groups),
                                  min_samples_leaf, min_weight_leaf,
                                  random_state)

        sample_weight = None
        res = splitter.test_init(X, y, sample_weight, groups)
        self.assertEqual(res, 0)
        self.assertIsNotNone(splitter.splitting_tree_builder)
        self.assertIsNotNone(splitter.splitting_tree)
        self.assertEqual(splitter.n_groups, 5)
        self.assertTrue((np.asarray(splitter.len_groups) == [5, 5, 5, 5, 5]).all())
        self.assertTrue(X.all().all() == splitter.X.all().all())
        self.assertTrue(y.all() == np.asarray(splitter.y).all())

    def test_node_reset(self):

        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])

        n_samples, n_features = X.shape
        n_grouped_features = 2
        y = np.atleast_1d(y)
        max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = 0  # (0.25 * n_samples)
        random_state = check_random_state(2457)
        max_depth = 3
        mgroup = 1
        mvar = 10
        min_impurity_decrease = 0.1
        min_impurity_split = 0.0

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        n_outputs = y.shape[1]

        y = np.copy(y)

        classes = []
        n_classes = []

        y_encoded = np.zeros(y.shape, dtype=np.float64)
        for k in range(n_outputs):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        y = y_encoded

        n_classes = np.array(n_classes, dtype=np.intp)

        criterion = CARTGVGini(n_outputs, n_classes)

        splitter = CARTGVSplitter(criterion, max_grouped_features, len(groups),
                                  min_samples_leaf, min_weight_leaf,
                                  random_state)

        sample_weight = None
        splitter.test_init(X, y, sample_weight, groups)

        start = 0
        end = n_samples - 1
        weighted_n_node_samples = 0
        res = splitter.test_node_reset(start, end, weighted_n_node_samples)
        self.assertEqual(res, 0)
        self.assertEqual(splitter.start, start)
        self.assertEqual(splitter.end, end)
        self.assertIsNotNone(splitter.criterion)

    def test_node_split(self):
        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])

        n_samples, n_features = X.shape
        n_grouped_features = 2
        y = np.atleast_1d(y)
        max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = (0.25 * n_samples)
        random_state = check_random_state(2457)
        max_depth = 3
        mgroup = 1
        mvar = 10
        min_impurity_decrease = 0.1
        min_impurity_split = 0.0

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        n_outputs = y.shape[1]

        y = np.copy(y)

        classes = []
        n_classes = []

        y_encoded = np.zeros(y.shape, dtype=np.float64)
        for k in range(n_outputs):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        y = y_encoded

        n_classes = np.array(n_classes, dtype=np.intp)

        criterion = CARTGVGini(n_outputs, n_classes)

        splitter = CARTGVSplitter(criterion, max_grouped_features, len(groups),
                                  min_samples_leaf, min_weight_leaf,
                                  random_state)

        sample_weight = None
        X, y, sample_weight = self._check_input(np.array(X), y, sample_weight)
        splitter.test_init(X, y, sample_weight, groups)

        start = 0
        end = n_samples
        weighted_n_node_samples = 0
        splitter.test_node_reset(start, end, weighted_n_node_samples)

        splitter.test_node_split(np.inf, 0)

    def test_one_split(self):
        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])

        n_samples, n_features = X.shape
        n_grouped_features = 2
        y = np.atleast_1d(y)
        max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
        max_features = len(groups[0])
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = 0
        random_state = check_random_state(2457)
        max_depth = 3
        mgroup = 1
        mvar = 10
        min_impurity_decrease = 0.1
        min_impurity_split = 0.0

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        n_outputs = y.shape[1]

        y = np.copy(y)

        classes = []
        n_classes = []

        y_encoded = np.zeros(y.shape, dtype=np.float64)
        for k in range(n_outputs):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        y = y_encoded

        n_classes = np.array(n_classes, dtype=np.intp)

        criterion = CARTGVGini(n_outputs, n_classes)

        splitter = CARTGVSplitter(criterion, max_grouped_features, len(groups),
                                  min_samples_leaf, min_weight_leaf,
                                  random_state)

        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
                                     max_leaf_nodes=X.shape[0])
        tree = clf.fit(X.iloc[:, groups[0]], y)
        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        plot_tree(tree, ax=ax[0])
        # plt.show()

        sample_weight = None
        X, y, sample_weight = self._check_input(np.array(X.iloc[:, groups[0]]), y, sample_weight)
        splitter.test_init(X, y, sample_weight, groups)

        start = 0
        end = n_samples
        weighted_n_node_samples = 0
        splitter.test_node_reset(start, end, weighted_n_node_samples)

        splitter.test_one_split(np.inf, 0)

        self.assertIsNotNone(splitter.splitting_tree)
        self.assertIsNotNone(splitter.criterion)
        self.assertIsNotNone(splitter.splitting_tree_builder)
        self.assertTrue(splitter.splitting_tree.node_count > 1)

        clf2 = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
                                      max_leaf_nodes=X.shape[0])
        clf2.tree_ = splitter.splitting_tree
        plot_tree(clf2, ax=ax[1])
        plt.title("current")
        plt.show()

        self.assertEqual(splitter.splitting_tree.n_features, clf.tree_.n_features)
        self.assertEqual(splitter.splitting_tree.n_classes, clf.tree_.n_classes)
        self.assertEqual(splitter.splitting_tree.n_outputs, clf.tree_.n_outputs)
        self.assertEqual(splitter.splitting_tree.max_n_classes, clf.tree_.max_n_classes)
        self.assertEqual(splitter.splitting_tree.max_depth, clf.tree_.max_depth)
        self.assertEqual(splitter.splitting_tree.node_count, clf.tree_.node_count)
        self.assertEqual(splitter.splitting_tree.n_leaves, clf.tree_.n_leaves)
        self.assertEqual(splitter.splitting_tree.value.all(), clf.tree_.value.all())

    def test_n_split(self):
        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])

        n_samples, n_features = X.shape
        n_grouped_features = 2
        y = np.atleast_1d(y)
        max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
        max_features = len(groups[0])
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = 0
        random_state = check_random_state(2457)
        max_depth = 3
        mgroup = 1
        mvar = 10
        min_impurity_decrease = 0.1
        min_impurity_split = 0.0

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        n_outputs = y.shape[1]

        y = np.copy(y)

        classes = []
        n_classes = []

        y_encoded = np.zeros(y.shape, dtype=np.float64)
        for k in range(n_outputs):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        y = y_encoded

        n_classes = np.array(n_classes, dtype=np.intp)

        criterion = CARTGVGini(n_outputs, n_classes)

        splitter = CARTGVSplitter(criterion, max_grouped_features, len(groups),
                                  min_samples_leaf, min_weight_leaf,
                                  random_state)

        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
                                     max_leaf_nodes=X.shape[0])
        tree = clf.fit(X.iloc[:, groups[0]], y)
        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        plot_tree(tree, ax=ax[0])
        # plt.show()

        sample_weight = None
        X, y, sample_weight = self._check_input(np.array(X.iloc[:, groups[0]]), y, sample_weight)
        splitter.test_init(X, y, sample_weight, groups)

        start = 0
        end = n_samples
        weighted_n_node_samples = 0
        splitter.test_node_reset(start, end, weighted_n_node_samples)

        splitter.test_n_split(np.inf, 0, 4, 4 - 1)

        self.assertIsNotNone(splitter.splitting_tree)
        self.assertIsNotNone(splitter.criterion)
        self.assertIsNotNone(splitter.splitting_tree_builder)
        self.assertTrue(splitter.splitting_tree.node_count > 1)

        clf2 = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
                                      max_leaf_nodes=X.shape[0])
        clf2.tree_ = splitter.splitting_tree
        plot_tree(clf2, ax=ax[1])
        plt.title("current")
        plt.show()

        self.assertEqual(splitter.splitting_tree.n_features, clf.tree_.n_features)
        self.assertEqual(splitter.splitting_tree.n_classes, clf.tree_.n_classes)
        self.assertEqual(splitter.splitting_tree.n_outputs, clf.tree_.n_outputs)
        self.assertEqual(splitter.splitting_tree.max_n_classes, clf.tree_.max_n_classes)
        self.assertEqual(splitter.splitting_tree.max_depth, clf.tree_.max_depth)
        self.assertEqual(splitter.splitting_tree.node_count, clf.tree_.node_count)
        self.assertEqual(splitter.splitting_tree.n_leaves, clf.tree_.n_leaves)
        self.assertEqual(splitter.splitting_tree.value.all(), clf.tree_.value.all())

    def test_splitting_tree_serialization(self):

        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])
        y = np.atleast_1d(y)

        random_state = check_random_state(2457)
        max_depth = 3

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        n_outputs = y.shape[1]

        y = np.copy(y)

        classes = []
        n_classes = []

        y_encoded = np.zeros(y.shape, dtype=np.float64)
        for k in range(n_outputs):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        y = y_encoded

        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
                                     max_leaf_nodes=X.shape[0])
        tree = clf.fit(X.iloc[:, groups[0]], y)

        splitting_tree_serialized = pickle.dumps(clf.tree_)
        splitting_tree_unserialized = pickle.loads(splitting_tree_serialized)

        self.assertEqual(splitting_tree_unserialized.n_features, clf.tree_.n_features)
        self.assertEqual(splitting_tree_unserialized.n_classes, clf.tree_.n_classes)
        self.assertEqual(splitting_tree_unserialized.n_outputs, clf.tree_.n_outputs)
        self.assertEqual(splitting_tree_unserialized.max_n_classes, clf.tree_.max_n_classes)
        self.assertEqual(splitting_tree_unserialized.max_depth, clf.tree_.max_depth)
        self.assertEqual(splitting_tree_unserialized.node_count, clf.tree_.node_count)
        self.assertEqual(splitting_tree_unserialized.n_leaves, clf.tree_.n_leaves)
        self.assertEqual(splitting_tree_unserialized.value.all(), clf.tree_.value.all())

    def test_splitting_tree_into_struct(self):
        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])

        n_samples, n_features = X.shape
        n_grouped_features = 2
        y = np.atleast_1d(y)
        max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
        max_features = len(groups[0])
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = 0
        random_state = check_random_state(2457)
        max_depth = 3
        mgroup = 1
        mvar = 10
        min_impurity_decrease = 0.1
        min_impurity_split = 0.0

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        n_outputs = y.shape[1]

        y = np.copy(y)

        classes = []
        n_classes = []

        y_encoded = np.zeros(y.shape, dtype=np.float64)
        for k in range(n_outputs):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        y = y_encoded

        n_classes = np.array(n_classes, dtype=np.intp)

        criterion = CARTGVGini(n_outputs, n_classes)

        splitter = CARTGVSplitter(criterion, max_grouped_features, len(groups),
                                  min_samples_leaf, min_weight_leaf,
                                  random_state)

        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
                                     max_leaf_nodes=X.shape[0])
        tree = clf.fit(X.iloc[:, groups[0]], y)

        splitting_tree = splitter.test_splitting_tree_into_struct(clf.tree_)

        splitting_tree_unserialized = pickle.loads(splitting_tree)

        self.assertEqual(splitting_tree_unserialized.n_features, clf.tree_.n_features)
        self.assertEqual(splitting_tree_unserialized.n_classes, clf.tree_.n_classes)
        self.assertEqual(splitting_tree_unserialized.n_outputs, clf.tree_.n_outputs)
        self.assertEqual(splitting_tree_unserialized.max_n_classes, clf.tree_.max_n_classes)
        self.assertEqual(splitting_tree_unserialized.max_depth, clf.tree_.max_depth)
        self.assertEqual(splitting_tree_unserialized.node_count, clf.tree_.node_count)
        self.assertEqual(splitting_tree_unserialized.n_leaves, clf.tree_.n_leaves)
        self.assertEqual(splitting_tree_unserialized.value.all(), clf.tree_.value.all())

    def test_splitting_tree_construction(self):
        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        y = y.to_numpy()
        y = np.ndarray((y.shape[0], 1), buffer=y, dtype=np.intp)

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])

        f_j = 0

        n_outputs = y.shape[1]
        n_features = X.shape[1]
        classes = []
        n_classes = []

        for k in range(n_outputs):
            classes_k = np.unique(y[:, k])
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

        n_classes = np.array(n_classes, dtype=np.intp)

        # Reset the splitting tree for the next loop iteration
        splitting_tree = Tree(n_features, n_classes, n_outputs)

        max_features = len(groups[f_j])
        max_leaf_nodes = X.shape[0]
        min_samples_leaf = 1
        min_samples_split = 2
        min_weight_leaf = 0.0
        max_depth = 3
        min_impurity_decrease = 0
        min_impurity_split = 0
        random_state = check_random_state(2547)
        max_grouped_features = max([len(groups[i]) for i in range(len(groups))])

        # Create the Criterion, Splitter et TreeBuilder for the splitting tree
        criterion = Gini(n_outputs, n_classes)
        splitter = BestSplitter(criterion, max_features, min_samples_leaf, min_weight_leaf, random_state)

        splitting_tree_builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                                       min_samples_leaf,
                                                       min_weight_leaf,
                                                       max_depth,
                                                       min_impurity_decrease,
                                                       min_impurity_split)

        cartgvcriterion = CARTGVGini(n_outputs, n_classes)

        cartgvsplitter = CARTGVSplitter(cartgvcriterion, max_grouped_features, len(groups),
                                  min_samples_leaf, min_weight_leaf,
                                  random_state)

        sample_weight = None
        X, y, sample_weight = self._check_input(np.array(X.iloc[:, groups[0]]), y, sample_weight)

        cartgvsplitter.test_init(X,y,sample_weight,groups)

        cartgvsplitter.splitting_tree = splitting_tree
        cartgvsplitter.splitting_tree_builder = splitting_tree_builder

        group = groups[0]
        len_groups = np.array([len(group) for group in groups])
        len_group = len(group)
        start = 0
        end = X.shape[0]

        Xf = cartgvsplitter.group_sample(group, len_group, start, end)

        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
                                         max_leaf_nodes=X.shape[0])

        fig, ax = plt.subplots(2, 4, figsize=(16, 9))

        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
                                     max_leaf_nodes=X.shape[0])
        tree = clf.fit(X, y)
        plot_tree(tree, ax=ax[1][3])
        plt.title("Reference")

        for i in range(7):
            print("################# Splitting tree Construction " + str(i) + " ####################")
            cartgvsplitter.splitting_tree_construction(Xf, y)
            clf.tree_ = cartgvsplitter.splitting_tree
            plot_tree(clf, ax=ax[i//4][i%4])
            if(i%2 == 0):
                cartgvsplitter.reset_scikit_learn_instances(y, len_groups)
        plt.show()

    # def test_best_node_split(self):
    #     df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)
    #
    #     train = df.loc[df['Type'] == 'train']
    #
    #     X = train.iloc[:, 2:]
    #
    #     y = train['Y']
    #
    #     g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
    #     g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
    #     g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
    #     g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
    #     g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]
    #
    #     groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])
    #
    #     n_samples, n_features = X.shape
    #     n_grouped_features = 2
    #     y = np.atleast_1d(y)
    #     max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
    #     min_samples_leaf = 1
    #     min_samples_split = 2
    #     min_weight_leaf = 0.0
    #     random_state = check_random_state(2457)
    #     max_depth = 3
    #     mgroup = 1
    #     mvar = 10
    #     min_impurity_decrease = 0.1
    #     min_impurity_split = 0.0
    #
    #     if y.ndim == 1:
    #         y = np.reshape(y, (-1, 1))
    #
    #     n_outputs = y.shape[1]
    #
    #     y = np.copy(y)
    #
    #     classes = []
    #     n_classes = []
    #
    #     y_encoded = np.zeros(y.shape, dtype=np.float64)
    #     for k in range(n_outputs):
    #         classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
    #         classes.append(classes_k)
    #         n_classes.append(classes_k.shape[0])
    #
    #     y = y_encoded
    #
    #     n_classes = np.array(n_classes, dtype=np.intp)
    #
    #     criterion = CARTGVGini(n_outputs, n_classes)
    #
    #     splitter = CARTGVSplitter(criterion, max_grouped_features, len(groups),
    #                               min_samples_leaf, min_weight_leaf,
    #                               random_state)
    #
    #     sample_weight = None
    #     X, y, sample_weight = self._check_input(np.array(X), y, sample_weight)
    #     splitter.test_init(X, y, sample_weight, groups)
    #
    #     start = 0
    #     end = n_samples
    #     weighted_n_node_samples = 0
    #     splitter.test_node_reset(start, end, weighted_n_node_samples)
    #
    #     best = splitter.test_best_node_split(np.inf, 0) #tree1, tree2,
    #
    #     clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
    #                                  max_leaf_nodes=X.shape[0])
    #
    #     # tree1 = pickle.loads(tree1)
    #     # tree2 = pickle.loads(tree2)
    #     # best = pickle.loads(best)
    #     # print(tree1)
    #     # print(tree2)
    #     print(best)
    #
    #     # clf.tree_ = tree1
    #     # print("plt subplot")
    #     # fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    #     # print("plot tree")
    #     # plot_tree(clf, ax=ax[0])
    #     # # plot_tree(clf)
    #     # # plt.show()
    #     # plt.title("current")
    #     #
    #     # print("tree1 done")
    #     #
    #     # tree2 = pickle.loads(tree2)
    #     # clf.tree_ = tree2
    #     # plot_tree(clf, ax=ax[1])
    #     # # plot_tree(clf)
    #     # # plt.show()
    #     # plt.title("current")
    #     #
    #     # print("tree2 done")
    #     #
    #     best = pickle.loads(best)
    #     print(best)
    #     print(best.node_count)
    #     clf.tree_ = best
    #     # plot_tree(clf)
    #     # plot_tree(clf, ax=ax[2])
    #     # plt.title("current")
    #     # plt.show()
    #
    #     print("best tree done")
    #     return 0

    def test_node_samples_construction(self):

        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        samples = np.arange(X.shape[0])

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])
        len_groups = np.array([len(g1_idx), len(g2_idx), len(g3_idx), len(g4_idx), len(g5_idx)])

        start = 0
        end = 334

        f_j = 0  # np.random.randint(0,max_grouped_features)           # Select a group at random

        group = groups[f_j]
        len_group = len_groups[f_j]

        Xf = np.empty((end - start, len_group))  # Récupère la shape correcte des données
        X = X.to_numpy()

        # Take the observations columns of group f_j between indexes start and end
        for i in range(start, end):
            for l in range(len_group):
                Xf[i][l] = X[samples[i], group[l]]

        self.assertEqual(Xf[start][0], X[samples[start], [0]])
        self.assertEqual(Xf[end-start-1][0], X[samples[end-start-1], [0]])


    def test_sklearn_tree_build(self):

        df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

        train = df.loc[df['Type'] == 'train']

        X = train.iloc[:, 2:]

        y = train['Y']

        y = y.to_numpy()
        y = np.ndarray((y.shape[0], 1), buffer=y, dtype=np.intp)
        print(X.to_numpy().shape)

        g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
        g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
        g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
        g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
        g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

        groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])

        for i in range(2):

            f_j = 0

            n_outputs = y.shape[1]
            n_features = X.shape[1]
            classes = []
            n_classes = []

            for k in range(n_outputs):
                classes_k = np.unique(y[:,k])
                classes.append(classes_k)
                n_classes.append(classes_k.shape[0])

            n_classes = np.array(n_classes, dtype=np.intp)

            # Reset the splitting tree for the next loop iteration
            splitting_tree = Tree(n_features, n_classes, n_outputs)

            max_features = len(groups[f_j])
            max_leaf_nodes = X.shape[0]
            min_samples_leaf = 1
            min_samples_split = 2
            min_weight_leaf = 0.0
            max_depth = 3
            min_impurity_decrease = 0
            min_impurity_split = 0
            random_state = check_random_state(2547)

            # Create the Criterion, Splitter et TreeBuilder for the splitting tree
            criterion = Gini(n_outputs, n_classes)
            splitter = BestSplitter(criterion, max_features, min_samples_leaf, min_weight_leaf, random_state)

            splitting_tree_builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                                                min_samples_leaf,
                                                                min_weight_leaf,
                                                                max_depth,
                                                                min_impurity_decrease,
                                                                min_impurity_split)

            splitting_tree_builder.build(splitting_tree, np.asarray(X), y, None)

            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
                                         max_leaf_nodes=X.shape[0])
            clf.tree_ = splitting_tree
            plot_tree(clf)
            plt.show()

            # Necessary loop to get the number of leaves as self.splitting_tree.n_leaves crash the program (the np.sum in the splitting_tree.n_leaves property has an error)
            n_leaves = 0
            for i in range(len(splitting_tree.children_left)):
                if splitting_tree.children_left[i] == -1 and splitting_tree.children_right[i] == -1:
                    n_leaves += 1

            n_nodes = splitting_tree.node_count

            self.assertEqual(n_nodes, 13)
            self.assertEqual(n_leaves, 7)

    # def test_sklearn_builder_field(self):
    #     df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)
    #
    #     train = df.loc[df['Type'] == 'train']
    #
    #     X = train.iloc[:, 2:]
    #
    #     y = train['Y']
    #
    #     g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
    #     g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
    #     g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
    #     g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
    #     g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]
    #
    #     groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])
    #
    #     y = np.atleast_1d(y)
    #     max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
    #     min_samples_leaf = 1
    #     min_weight_leaf = 0
    #     random_state = check_random_state(2457)
    #
    #
    #     if y.ndim == 1:
    #         y = np.reshape(y, (-1, 1))
    #
    #     n_outputs = y.shape[1]
    #
    #     y = np.copy(y)
    #
    #     classes = []
    #     n_classes = []
    #
    #     y_encoded = np.zeros(y.shape, dtype=np.float64)
    #     for k in range(n_outputs):
    #         classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
    #         classes.append(classes_k)
    #         n_classes.append(classes_k.shape[0])
    #
    #     y = y_encoded
    #
    #     n_classes = np.array(n_classes, dtype=np.intp)
    #
    #     criterion = CARTGVGini(n_outputs, n_classes)
    #
    #     splitter = CARTGVSplitter(criterion, max_grouped_features, len(groups),
    #                               min_samples_leaf, min_weight_leaf,
    #                               random_state)
    #
    #     sample_weight = None
    #     X, y, sample_weight = self._check_input(np.array(X), y, sample_weight)
    #     splitter.test_init(X, y, sample_weight, groups)
    #
    #     start = 0
    #     end = X.shape[0]
    #     weighted_n_node_samples = 0
    #     splitter.test_node_reset(start, end, weighted_n_node_samples)
    #
    #     splitter.test_sklearn_builder_field()

    def _check_input(self, X, y, sample_weight):
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


# class CARTGVTreeTest(unittest.TestCase):

if __name__ == '__main__':
    unittest.main()
