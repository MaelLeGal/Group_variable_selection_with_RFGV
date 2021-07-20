import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, fetch_california_housing, load_boston, load_diabetes
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree._tree import Tree

from CARTGV_trees import DecisionCARTGVTreeClassifier, DecisionCARTGVTreeRegressor
from RFGV import RFGVClassifier, RFGVRegressor, _get_n_samples_bootstrap, _generate_sample_indices, _generate_unsampled_indices

# df = pd.read_csv('../data/training.csv', sep=",")
# df_test = pd.read_csv('../data/testing.csv', sep=",")

# df = df[df['class'].isin(['grass ', 'building ']) ]
# df_test = df_test[df_test['class'].isin(['grass ', 'building '])]

# X = df.iloc[:, 1:]

# y = df['class'][:]

# X_test = df_test.iloc[:, 1:]
# y_test = df_test['class'][:]

# g0_idx = [col for col in range(len(X.columns)) if ('_40' not in X.columns[col] and '_60' not in X.columns[col] and '_80' not in X.columns[col] and '_100' not in X.columns[col] and '_120' not in X.columns[col] and '_140' not in X.columns[col])]
# g0_idx = np.array([col for col in range(20)])
# g1_idx = np.array([col for col in range(len(X.columns)) if '_40' in X.columns[col]])
# g2_idx = np.array([col for col in range(len(X.columns)) if '_60' in X.columns[col]])
# g3_idx = np.array([col for col in range(len(X.columns)) if '_80' in X.columns[col]])
# g4_idx = np.array([col for col in range(len(X.columns)) if '_100' in X.columns[col]])
# g5_idx = np.array([col for col in range(len(X.columns)) if '_120' in X.columns[col]])
# g6_idx = np.array([col for col in range(len(X.columns)) if '_140' in X.columns[col]])
# g7_idx = np.hstack([[col for col in range(len(X.columns)) if 'Bright' in X.columns[col]], [col for col in range(len(X.columns)) if 'Mean_' in X.columns[col]], [col for col in range(len(X.columns)) if 'NDVI' in X.columns[col]]])
# g8_idx = np.hstack([[col for col in range(len(X.columns)) if 'SD_' in X.columns[col]], [col for col in range(len(X.columns)) if 'GLCM' in X.columns[col]]])
# g9_idx = np.array([X.columns.get_loc("BrdIndx"),X.columns.get_loc("Area"),X.columns.get_loc("Round"),X.columns.get_loc("Compact"),X.columns.get_loc("ShpIndx"),X.columns.get_loc("LW"),X.columns.get_loc("Rect"),X.columns.get_loc("Dens"),X.columns.get_loc("Assym"),X.columns.get_loc("BordLngth")])

X_reg, y_reg = fetch_california_housing(return_X_y=True)
X_reg, y_reg = X_reg[:30], y_reg[:30]
X_reg, X_reg_test, y_reg, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3)

RANDOM_STATE = 2547

class MyTestCase(unittest.TestCase):


    def test_predict_train_CART_classif_multi_titanic(self):

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=1664816)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, group_importance="Breiman", verbose=0, random=False,
                              max_features=None)  # verbose >1 for text
        rfgv.fit(X.iloc[:n_obs], y.iloc[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        print("CART train predict classif multi-classes")
        np.testing.assert_array_equal(rfgv.predict(X), cartgvtree.predict(X))
        np.testing.assert_array_equal(cartgvtree.predict(X), clf.predict(X))
        np.testing.assert_array_equal(rfgv.predict(X), clf.predict(X))

    def test_score_train_CART_classif_multi_titanic(self):

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=1664816)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, group_importance="Breiman", verbose=0, random=False,
                              max_features=None)  # verbose >1 for text
        rfgv.fit(X.iloc[:n_obs], y.iloc[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        print("CART train score classif multi-classes")
        np.testing.assert_array_equal(rfgv.score(X,y), cartgvtree.score(X,y))
        np.testing.assert_array_equal(cartgvtree.score(X,y), clf.score(X,y))
        np.testing.assert_array_equal(rfgv.score(X,y), clf.score(X,y))

    def test_predict_test_CART_classif_multi_titanic(self):

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=1664816)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups), max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, group_importance="Breiman", verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X.iloc[:n_obs], y.iloc[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE, max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        print("CART test predict classif multi-classes")
        np.testing.assert_array_equal(rfgv.predict(X_test),cartgvtree.predict(X_test))
        np.testing.assert_array_equal(cartgvtree.predict(X_test),clf.predict(X_test))
        np.testing.assert_array_equal(rfgv.predict(X_test), clf.predict(X_test))

        # np.testing.assert_array_almost_equal(rfgv.predict(X_test),clf.predict(X_test))
        # np.testing.assert_array_almost_equal(cartgvtree.predict(X_test), clf.predict(X_test))

    def test_score_test_CART_classif_multi_titanic(self):

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=1664816)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        print("CART test score classif multi-classes scores")

        self.assertEqual(rfgv.score(X_test, y_test), cartgvtree.score(X_test, y_test))
        self.assertEqual(rfgv.score(X_test, y_test), clf.score(X_test,y_test))
        self.assertEqual(cartgvtree.score(X_test,y_test), clf.score(X_test,y_test))

        # self.assertAlmostEqual(rfgv.score(X_test, y_test), clf.score(X_test,y_test), delta=0.1)
        # self.assertAlmostEqual(cartgvtree.score(X_test,y_test), clf.score(X_test,y_test), delta=0.1)

    def test_predict_train_CART_classif_bin_titanic(self):

        # df = pd.read_csv('../data/training.csv', sep=",")
        # df_test = pd.read_csv('../data/testing.csv', sep=",")
        #
        # df = df[df['class'].isin(['grass ', 'building ']) ]
        # df_test = df_test[df_test['class'].isin(['grass ', 'building '])]
        #
        # X = df.iloc[:, 1:]
        #
        # y = df['class']
        #
        # X_test = df_test.iloc[:, 1:]
        # y_test = df_test['class']

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        df = df[df['pclass'].isin(['2nd', '3rd'])]

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups), max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE, max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        print("CART train predict classif binaire")
        np.testing.assert_array_equal(rfgv.predict(X),cartgvtree.predict(X))
        np.testing.assert_array_equal(cartgvtree.predict(X),clf.predict(X))
        np.testing.assert_array_equal(rfgv.predict(X), clf.predict(X))

        # np.testing.assert_array_almost_equal(rfgv.predict(X_test),clf.predict(X_test))
        # np.testing.assert_array_almost_equal(cartgvtree.predict(X_test), clf.predict(X_test))

    def test_score_train_CART_classif_bin_titanic(self):

        # df = pd.read_csv('../data/training.csv', sep=",")
        # df_test = pd.read_csv('../data/testing.csv', sep=",")
        #
        # df = df[df['class'].isin(['grass ', 'building '])]
        # df_test = df_test[df_test['class'].isin(['grass ', 'building '])]
        #
        # X = df.iloc[:, 1:]
        #
        # y = df['class']
        #
        # X_test = df_test.iloc[:, 1:]
        # y_test = df_test['class']

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        df = df[df['pclass'].isin(['2nd', '3rd'])]

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        print("CART train score classif binaire")

        self.assertEqual(rfgv.score(X, y), cartgvtree.score(X, y))
        self.assertEqual(rfgv.score(X, y), clf.score(X,y))
        self.assertEqual(cartgvtree.score(X,y), clf.score(X,y))

        # self.assertAlmostEqual(rfgv.score(X_test, y_test), clf.score(X_test,y_test), delta=0.1)
        # self.assertAlmostEqual(cartgvtree.score(X_test,y_test), clf.score(X_test,y_test), delta=0.1)

    def test_predict_test_CART_classif_bin_titanic(self):

        # df = pd.read_csv('../data/training.csv', sep=",")
        # df_test = pd.read_csv('../data/testing.csv', sep=",")
        #
        # df = df[df['class'].isin(['grass ', 'building ']) ]
        # df_test = df_test[df_test['class'].isin(['grass ', 'building '])]
        #
        # X = df.iloc[:, 1:]
        #
        # y = df['class']
        #
        # X_test = df_test.iloc[:, 1:]
        # y_test = df_test['class']

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        df = df[df['pclass'].isin(['2nd', '3rd'])]

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups), max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE, max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        print("CART test predict classif binaire")
        np.testing.assert_array_equal(rfgv.predict(X_test),cartgvtree.predict(X_test))
        np.testing.assert_array_equal(cartgvtree.predict(X_test),clf.predict(X_test))
        np.testing.assert_array_equal(rfgv.predict(X_test), clf.predict(X_test))

        # np.testing.assert_array_almost_equal(rfgv.predict(X_test),clf.predict(X_test))
        # np.testing.assert_array_almost_equal(cartgvtree.predict(X_test), clf.predict(X_test))

    def test_score_test_CART_classif_bin_titanic(self):

        # df = pd.read_csv('../data/training.csv', sep=",")
        # df_test = pd.read_csv('../data/testing.csv', sep=",")
        #
        # df = df[df['class'].isin(['grass ', 'building '])]
        # df_test = df_test[df_test['class'].isin(['grass ', 'building '])]
        #
        # X = df.iloc[:, 1:]
        #
        # y = df['class']
        #
        # X_test = df_test.iloc[:, 1:]
        # y_test = df_test['class']

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        df = df[df['pclass'].isin(['2nd', '3rd'])]

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        print("CART test score classif binaire")

        self.assertEqual(rfgv.score(X_test, y_test), cartgvtree.score(X_test, y_test))
        self.assertEqual(rfgv.score(X_test, y_test), clf.score(X_test,y_test))
        self.assertEqual(cartgvtree.score(X_test,y_test), clf.score(X_test,y_test))

        # self.assertAlmostEqual(rfgv.score(X_test, y_test), clf.score(X_test,y_test), delta=0.1)
        # self.assertAlmostEqual(cartgvtree.score(X_test,y_test), clf.score(X_test,y_test), delta=0.1)

    def test_predict_train_CART_regression_california_housing(self):

        X_reg, y_reg = fetch_california_housing(return_X_y=True)
        X_reg, y_reg = X_reg[:300], y_reg[:300]
        X_reg, X_reg_test, y_reg, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X_reg.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y_reg.shape[0]
        rfgv = RFGVRegressor(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, group_importance="Breiman", verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        clf = DecisionTreeRegressor(random_state=RANDOM_STATE)
        clf.fit(X_reg[:n_obs], y_reg[:n_obs])

        print("CART train predict regression")
        np.testing.assert_array_equal(rfgv.predict(X_reg), cartgvtree.predict(X_reg))
        np.testing.assert_array_equal(cartgvtree.predict(X_reg), clf.predict(X_reg))
        np.testing.assert_array_equal(rfgv.predict(X_reg), clf.predict(X_reg))
        # np.testing.assert_array_equal(clf.predict(X_reg_test), clf2.predict(X_reg_test))

    def test_score_train_CART_regression_california_housing(self):

        X_reg, y_reg = fetch_california_housing(return_X_y=True)
        X_reg, y_reg = X_reg[:300], y_reg[:300]
        X_reg, X_reg_test, y_reg, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X_reg.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y_reg.shape[0]
        rfgv = RFGVRegressor(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        clf = DecisionTreeRegressor(random_state=RANDOM_STATE)
        clf.fit(X_reg[:n_obs], y_reg[:n_obs])

        print("CART train score regression")

        self.assertEqual(rfgv.score(X_reg, y_reg), cartgvtree.score(X_reg, y_reg))
        self.assertEqual(rfgv.score(X_reg, y_reg), clf.score(X_reg,y_reg))
        self.assertEqual(cartgvtree.score(X_reg,y_reg), clf.score(X_reg,y_reg))

    def test_predict_test_CART_regression_california_housing(self):

        X_reg, y_reg = fetch_california_housing(return_X_y=True)
        X_reg, y_reg = X_reg[:300], y_reg[:300]
        X_reg, X_reg_test, y_reg, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X_reg.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y_reg.shape[0]
        rfgv = RFGVRegressor(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, group_importance="Breiman", verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        clf = DecisionTreeRegressor(random_state=RANDOM_STATE)
        clf.fit(X_reg[:n_obs], y_reg[:n_obs])

        print("CART test predict regression")
        cpred = cartgvtree.predict(X_reg_test)
        spred = clf.predict(X_reg_test)

        obs = np.where(cpred != spred)[0]

        if obs.size > 0:
            print(cartgvtree.score(X_reg_test[obs], y_reg_test[obs]))
            print(clf.score(X_reg_test[obs], y_reg_test[obs]))

        np.testing.assert_array_equal(rfgv.predict(X_reg_test), cartgvtree.predict(X_reg_test))
        np.testing.assert_array_equal(cartgvtree.predict(X_reg_test), clf.predict(X_reg_test))
        np.testing.assert_array_equal(rfgv.predict(X_reg_test), clf.predict(X_reg_test))
        # np.testing.assert_array_equal(clf.predict(X_reg_test), clf2.predict(X_reg_test))

    def test_score_test_CART_regression_california_housing(self):

        X_reg, y_reg = fetch_california_housing(return_X_y=True)
        X_reg, y_reg = X_reg[:300], y_reg[:300]
        X_reg, X_reg_test, y_reg, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X_reg.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y_reg.shape[0]
        rfgv = RFGVRegressor(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        clf = DecisionTreeRegressor(random_state=RANDOM_STATE)
        clf.fit(X_reg[:n_obs], y_reg[:n_obs])

        print("CART test score regression")

        self.assertEqual(rfgv.score(X_reg_test, y_reg_test), cartgvtree.score(X_reg_test, y_reg_test))
        self.assertEqual(rfgv.score(X_reg_test, y_reg_test), clf.score(X_reg_test,y_reg_test))
        self.assertEqual(cartgvtree.score(X_reg_test,y_reg_test), clf.score(X_reg_test,y_reg_test))


    def test_predict_train_CARTGV_classif_multi_titanic(self):

        # groups = np.array([g0_idx, g1_idx, g2_idx, g3_idx, g4_idx, g5_idx, g6_idx, g7_idx, g8_idx, g9_idx],
        #                   dtype=object)

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = np.array([[1,2],[0]], dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2,
                              random_state=RANDOM_STATE, group_importance="Breiman",
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        print("CARTGV train predict classif multi-classes")
        np.testing.assert_array_equal(rfgv.predict(X), cartgvtree.predict(X))

    def test_score_train_CARTGV_classif_multi_titanic(self):

        # groups = np.array([g0_idx, g1_idx, g2_idx, g3_idx, g4_idx, g5_idx, g6_idx, g7_idx, g8_idx, g9_idx],
        #                   dtype=object)

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = np.array([[1, 2], [0]], dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2, random_state=RANDOM_STATE,
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        print("CARTGV train score classif multi-classes")
        self.assertEqual(rfgv.score(X, y), cartgvtree.score(X, y))


    def test_predict_test_CARTGV_classif_multi_titanic(self):

        # groups = np.array([g0_idx, g1_idx, g2_idx, g3_idx, g4_idx, g5_idx, g6_idx, g7_idx, g8_idx, g9_idx],
        #                   dtype=object)

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = np.array([[1,2],[0]], dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2,
                              random_state=RANDOM_STATE, group_importance="Breiman",
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        print("CARTGV test predict classif multi-classes")
        np.testing.assert_array_equal(rfgv.predict(X_test), cartgvtree.predict(X_test))

    def test_score_test_CARTGV_classif_multi_titanic(self):

        # groups = np.array([g0_idx, g1_idx, g2_idx, g3_idx, g4_idx, g5_idx, g6_idx, g7_idx, g8_idx, g9_idx],
        #                   dtype=object)

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = np.array([[1, 2], [0]], dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2, random_state=RANDOM_STATE,
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        print("CARTGV train score classif multi-classes")
        self.assertEqual(rfgv.score(X_test, y_test), cartgvtree.score(X_test, y_test))


    def test_predict_train_CARTGV_classif_bin_titanic(self):

        # df = pd.read_csv('../data/training.csv', sep=",")
        # df_test = pd.read_csv('../data/testing.csv', sep=",")
        #
        # df = df[df['class'].isin(['grass ', 'building '])]
        # df_test = df_test[df_test['class'].isin(['grass ', 'building '])]
        #
        # X = df.iloc[:, 1:]
        #
        # y = df['class']
        #
        # X_test = df_test.iloc[:, 1:]
        # y_test = df_test['class']

        # groups = np.array([g0_idx, g1_idx, g2_idx, g3_idx, g4_idx, g5_idx, g6_idx, g7_idx, g8_idx, g9_idx],
        #                   dtype=object)

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        df = df[df['pclass'].isin(['2nd', '3rd'])]

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = np.array([[1, 2], [0]], dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2,
                              random_state=RANDOM_STATE, group_importance="Breiman",
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        print("CARTGV train predict classif binaire")
        np.testing.assert_array_equal(rfgv.predict(X), cartgvtree.predict(X))

    def test_score_train_CARTGV_classif_bin_titanic(self):

        # df = pd.read_csv('../data/training.csv', sep=",")
        # df_test = pd.read_csv('../data/testing.csv', sep=",")
        #
        # df = df[df['class'].isin(['grass ', 'building '])]
        # df_test = df_test[df_test['class'].isin(['grass ', 'building '])]
        #
        # X = df.iloc[:, 1:]
        #
        # y = df['class']
        #
        # X_test = df_test.iloc[:, 1:]
        # y_test = df_test['class']
        #
        # groups = np.array([g0_idx, g1_idx, g2_idx, g3_idx, g4_idx, g5_idx, g6_idx, g7_idx, g8_idx, g9_idx],
        #                   dtype=object)

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        df = df[df['pclass'].isin(['2nd', '3rd'])]

        # print(df.head(10))

        X = df.iloc[:, 4:]
        y = df['pclass']

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = np.array([[1, 2], [0]], dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2, random_state=RANDOM_STATE,
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        print("CARTGV train score classif binaire")
        self.assertEqual(rfgv.score(X, y), cartgvtree.score(X, y))

    def test_predict_test_CARTGV_classif_bin_titanic(self):

        # df = pd.read_csv('../data/training.csv', sep=",")
        # df_test = pd.read_csv('../data/testing.csv', sep=",")
        #
        # df = df[df['class'].isin(['grass ', 'building '])]
        # df_test = df_test[df_test['class'].isin(['grass ', 'building '])]
        #
        # X = df.iloc[:, 1:]
        #
        # y = df['class']
        #
        # X_test = df_test.iloc[:, 1:]
        # y_test = df_test['class']

        # groups = np.array([g0_idx, g1_idx, g2_idx, g3_idx, g4_idx, g5_idx, g6_idx, g7_idx, g8_idx, g9_idx],
        #                   dtype=object)

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        df = df[df['pclass'].isin(['2nd', '3rd'])]

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = np.array([[1, 2], [0]], dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2,
                              random_state=RANDOM_STATE, group_importance="Breiman",
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        print("CARTGV test predict classif binaire")
        np.testing.assert_array_equal(rfgv.predict(X_test), cartgvtree.predict(X_test))

    def test_score_test_CARTGV_classif_bin_titanic(self):

        # df = pd.read_csv('../data/training.csv', sep=",")
        # df_test = pd.read_csv('../data/testing.csv', sep=",")
        #
        # df = df[df['class'].isin(['grass ', 'building '])]
        # df_test = df_test[df_test['class'].isin(['grass ', 'building '])]
        #
        # X = df.iloc[:, 1:]
        #
        # y = df['class']
        #
        # X_test = df_test.iloc[:, 1:]
        # y_test = df_test['class']
        #
        # groups = np.array([g0_idx, g1_idx, g2_idx, g3_idx, g4_idx, g5_idx, g6_idx, g7_idx, g8_idx, g9_idx],
        #                   dtype=object)

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        df = df[df['pclass'].isin(['2nd', '3rd'])]

        # print(df.head(10))

        X = df.iloc[:, 4:]
        y = df['pclass']

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = np.array([[1, 2], [0]], dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2, random_state=RANDOM_STATE,
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        print("CARTGV test score classif binaire")
        self.assertEqual(rfgv.score(X_test, y_test), cartgvtree.score(X_test, y_test))

    def test_predict_train_CARTGV_regression_california_housing(self):

        X_reg, y_reg = fetch_california_housing(return_X_y=True)
        X_reg, y_reg = X_reg[:300], y_reg[:300]
        X_reg, X_reg_test, y_reg, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.5, random_state=6761474)

        np.random.seed(RANDOM_STATE)
        random_groups = [np.random.choice(8, np.random.randint(1, 5), replace=False) for i in
                         range(np.random.randint(3, 5))]
        groups = np.array(random_groups, dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y_reg.shape[0]
        rfgv = RFGVRegressor(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2,
                              random_state=RANDOM_STATE, group_importance="Breiman",
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        print("CARTGV train predict régression")
        np.testing.assert_array_equal(rfgv.predict(X_reg), cartgvtree.predict(X_reg))

    def test_score_train_CARTGV_regression_california_housing(self):

        X_reg, y_reg = fetch_california_housing(return_X_y=True)
        X_reg, y_reg = X_reg[:300], y_reg[:300]
        X_reg, X_reg_test, y_reg, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.5, random_state=6761474)

        np.random.seed(RANDOM_STATE)
        random_groups = [np.random.choice(8, np.random.randint(1, 5), replace=False) for i in
                         range(np.random.randint(3, 5))]
        groups = np.array(random_groups, dtype=object)
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y_reg.shape[0]
        rfgv = RFGVRegressor(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2, random_state=RANDOM_STATE,
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        print("CARTGV train score regression")
        self.assertEqual(rfgv.score(X_reg, y_reg), cartgvtree.score(X_reg, y_reg))

    def test_predict_test_CARTGV_regression_california_housing(self):

        X_reg, y_reg = fetch_california_housing(return_X_y=True)
        X_reg, y_reg = X_reg[:300], y_reg[:300]
        X_reg, X_reg_test, y_reg, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.5, random_state=6761474)

        np.random.seed(RANDOM_STATE)
        random_groups = [np.random.choice(8, np.random.randint(1, 5), replace=False) for i in
                         range(np.random.randint(3, 5))]
        groups = np.array(random_groups, dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y_reg.shape[0]
        rfgv = RFGVRegressor(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2,
                              random_state=RANDOM_STATE, group_importance="Breiman",
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        print("CARTGV test predict regression")
        np.testing.assert_array_equal(rfgv.predict(X_reg_test), cartgvtree.predict(X_reg_test))

    def test_score_test_CARTGV_regression_california_housing(self):

        X_reg, y_reg = fetch_california_housing(return_X_y=True)
        X_reg, y_reg = X_reg[:300], y_reg[:300]
        X_reg, X_reg_test, y_reg, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.5, random_state=6761474)

        np.random.seed(RANDOM_STATE)
        random_groups = [np.random.choice(8, np.random.randint(1, 5), replace=False) for i in
                         range(np.random.randint(3, 5))]
        groups = np.array(random_groups, dtype=object)
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y_reg.shape[0]
        rfgv = RFGVRegressor(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=2, random_state=RANDOM_STATE,
                              verbose=0,
                              random=False)  # verbose >1 for text
        rfgv.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=2)
        cartgvtree.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        print("CARTGV test score regression")
        self.assertEqual(rfgv.score(X_reg_test, y_reg_test), cartgvtree.score(X_reg_test, y_reg_test))

    def test_prediction_rfgv_multi_classes_titanic(self):
        # groups = np.array([g0_idx, g1_idx, g2_idx, g3_idx, g4_idx, g5_idx, g6_idx, g7_idx, g8_idx, g9_idx], dtype=object)

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = np.array([[1, 2], [0]], dtype=object)

        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        n_obs_test = y_test.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar="root", mgroup=len(groups),
                              max_depth_splitting_tree=2,
                              random_state=RANDOM_STATE, group_importance="Breiman", verbose=0)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)


        predictions = np.zeros((n_obs_test,n_estimators), dtype=int)
        for i in range(len(rfgv.estimators_)):

            res = np.array(rfgv.estimators_[i].predict(X_test.iloc[:n_obs_test]), dtype=int)
            for j in range(n_obs_test):
                predictions[j,i] = res[j]

        res_estimators = []
        for i in range(n_obs_test):
            res_estimators.append(rfgv.classes_[np.bincount(predictions[i]).argmax()])

        res = rfgv.predict(X_test.iloc[:n_obs_test])

        np.testing.assert_array_equal(res,res_estimators)


    def test_tree_comp_CART_classif_bin_titanic(self):

        print("CART tree comparaison classif binaire")

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        df = df[df['pclass'].isin(['2nd', '3rd'])]

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])


        # print(cartgvtree.tree_.n_leaves)
        # print(clf.tree_.n_leaves)
        # print(cartgvtree.tree_.max_depth)
        # print(clf.tree_.max_depth)
        # print(cartgvtree.tree_.node_count)
        # print(clf.tree_.node_count)
        # print(cartgvtree.tree_.nodes_impurities, clf.tree_.impurity)
        # print(set(cartgvtree.tree_.value.flat))
        # print(set(clf.tree_.value.flat))
        # print(cartgvtree.tree_.nodes_n_node_samples)
        # print(clf.tree_.n_node_samples)
        # print(clf.tree_.feature)
        # print(cartgvtree.tree_.nodes_group)

        # fig, ax = plt.subplots(1, figsize=(16,9))
        # plot_tree(clf, ax=ax)
        # plt.show()

        # np.testing.assert_array_equal(cartgvtree.tree_.value,clf.tree_.value) # Same arrays but different order
        self.assertCountEqual(cartgvtree.tree_.nodes_group, clf.tree_.feature)
        self.assertCountEqual(cartgvtree.tree_.nodes_n_node_samples, clf.tree_.n_node_samples)
        self.assertSetEqual(set(cartgvtree.tree_.nodes_n_node_samples),set(clf.tree_.n_node_samples))
        self.assertCountEqual(cartgvtree.tree_.value.flat,clf.tree_.value.flat)
        self.assertSetEqual(set(cartgvtree.tree_.value.flat),set(clf.tree_.value.flat)) # A way to compare the values but not perfect
        self.assertCountEqual(cartgvtree.tree_.nodes_impurities, clf.tree_.impurity)
        self.assertSetEqual(set(cartgvtree.tree_.nodes_impurities), set(clf.tree_.impurity))
        self.assertEqual(cartgvtree.tree_.node_count, clf.tree_.node_count)
        self.assertEqual(cartgvtree.tree_.n_leaves,clf.tree_.n_leaves)
        self.assertEqual(cartgvtree.tree_.max_depth, clf.tree_.max_depth)

    def test_tree_comp_CART_classif_multi_titanic(self):

        print("CART tree comparaison classif multi-classes")

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        X = df.iloc[:, 4:]
        y = df['pclass'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])


        # print(cartgvtree.tree_.n_leaves)
        # print(clf.tree_.n_leaves)
        # print(cartgvtree.tree_.max_depth)
        # print(clf.tree_.max_depth)
        # print(cartgvtree.tree_.node_count)
        # print(clf.tree_.node_count)
        # print(cartgvtree.tree_.nodes_impurities, clf.tree_.impurity)
        # print(set(cartgvtree.tree_.value.flat))
        # print(set(clf.tree_.value.flat))
        # print(cartgvtree.tree_.nodes_n_node_samples)
        # print(clf.tree_.n_node_samples)
        # print(clf.tree_.feature)
        # print(cartgvtree.tree_.nodes_group)

        # fig, ax = plt.subplots(1, figsize=(16,9))
        # plot_tree(clf, ax=ax)
        # plt.show()

        # np.testing.assert_array_equal(cartgvtree.tree_.value,clf.tree_.value) # Same arrays but different order
        self.assertCountEqual(cartgvtree.tree_.nodes_group, clf.tree_.feature)
        self.assertCountEqual(cartgvtree.tree_.nodes_n_node_samples, clf.tree_.n_node_samples)
        self.assertSetEqual(set(cartgvtree.tree_.nodes_n_node_samples),set(clf.tree_.n_node_samples))
        self.assertCountEqual(cartgvtree.tree_.value.flat,clf.tree_.value.flat)
        self.assertSetEqual(set(cartgvtree.tree_.value.flat),set(clf.tree_.value.flat)) # A way to compare the values but not perfect
        self.assertCountEqual(cartgvtree.tree_.nodes_impurities, clf.tree_.impurity)
        self.assertSetEqual(set(cartgvtree.tree_.nodes_impurities), set(clf.tree_.impurity))
        self.assertEqual(cartgvtree.tree_.node_count, clf.tree_.node_count)
        self.assertEqual(cartgvtree.tree_.n_leaves,clf.tree_.n_leaves)
        self.assertEqual(cartgvtree.tree_.max_depth, clf.tree_.max_depth)

    def test_tree_comp_CART_regression_california_housing(self):

        X_reg, y_reg = fetch_california_housing(return_X_y=True)
        X_reg, y_reg = X_reg[:300], y_reg[:300]
        X_reg, X_reg_test, y_reg, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X_reg.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y_reg.shape[0]
        rfgv = RFGVRegressor(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                             max_depth_splitting_tree=1, random_state=RANDOM_STATE, verbose=0,
                             random=False)  # verbose >1 for text
        rfgv.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                 max_depth_splitting_tree=1)
        cartgvtree.fit(X_reg[:n_obs], y_reg[:n_obs], groups)

        clf = DecisionTreeRegressor(random_state=RANDOM_STATE)
        clf.fit(X_reg[:n_obs], y_reg[:n_obs])

        print("CART tree comparaison regression")

        self.assertCountEqual(cartgvtree.tree_.nodes_group, clf.tree_.feature)
        self.assertCountEqual(cartgvtree.tree_.nodes_n_node_samples, clf.tree_.n_node_samples)
        self.assertSetEqual(set(cartgvtree.tree_.nodes_n_node_samples), set(clf.tree_.n_node_samples))
        self.assertCountEqual(cartgvtree.tree_.value.flat, clf.tree_.value.flat)
        self.assertSetEqual(set(cartgvtree.tree_.value.flat), set(clf.tree_.value.flat))  # A way to compare the values but not perfect
        self.assertCountEqual(cartgvtree.tree_.nodes_impurities, clf.tree_.impurity)
        self.assertSetEqual(set(cartgvtree.tree_.nodes_impurities), set(clf.tree_.impurity))
        self.assertEqual(cartgvtree.tree_.node_count, clf.tree_.node_count)
        self.assertEqual(cartgvtree.tree_.n_leaves, clf.tree_.n_leaves)
        self.assertEqual(cartgvtree.tree_.max_depth, clf.tree_.max_depth)

    def test_obs_follow(self):

        print("CART observation follow")

        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        df = df[df['pclass'].isin(['2nd', '3rd'])]

        X = df.iloc[:330, 4:]
        y = df['pclass'][:330]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        cpred = cartgvtree.predict(X_test)
        spred = clf.predict(X_test)

        # np.testing.assert_array_equal(cpred, spred)
        obs = np.where(cpred != spred)[0]
        print(obs)
        if obs.size == 0:
            obs = 0
        else:
            obs = obs[0]
        print(obs)

        # obs = np.random.randint(0,n_obs)

        print(X_test.iloc[obs])

        node = cartgvtree.apply(X_test.iloc[obs].to_numpy().reshape(1,-1))
        pred = cartgvtree.predict(X_test.iloc[obs].to_numpy().reshape(1,-1))
        spred_obs = clf.predict(X_test.iloc[obs].to_numpy().reshape(1,-1))

        print(pred)
        print(spred_obs)

        split_tree = DecisionTreeClassifier()
        split_tree.tree_ = cartgvtree.tree_.nodes_splitting_trees[int(cartgvtree.tree_.nodes_parent[node[0]])]
        split_tree.n_outputs_ = 1
        split_tree.classes_ = np.array(['2nd', '3rd'])

        pred_split = split_tree.predict(X_test.iloc[obs].to_numpy().reshape(1,-1))

        print(pred_split)

        # fig, ax = plt.subplots(1, figsize=(16, 9))
        # plot_tree(split_tree, ax=ax)
        # plt.show()
        #
        # fig, ax = plt.subplots(1, figsize=(16, 9))
        # plot_tree(clf, ax=ax)
        # plt.show()

        self.assertEqual(pred,pred_split)

    def test_one_important_variable(self):
        df = pd.read_csv('../data/titanic.csv')

        df.dropna(inplace=True)

        le = LabelEncoder()

        df = df.apply(le.fit_transform)

        print(df.head(10))

        # df = df[df['pclass'].isin(['2nd', '3rd'])]
        np.random.seed(6761474)
        uniform1 = np.random.uniform(-10, -3, df.shape[0])
        np.random.seed(6761474)
        uniform2 = np.random.uniform(-50, -26, df.shape[0])
        df['uniform1'] = uniform1
        df['uniform2'] = uniform2

        X = df.iloc[:,[1,3,4,7,8]] #1,3,4

        y = df['survived'][:]

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        print(X[:10])

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        print("CART test predict and score One important variable")

        # print(cartgvtree.tree_.n_leaves)
        # print(clf.tree_.n_leaves)
        # print(cartgvtree.tree_.max_depth)
        # print(clf.tree_.max_depth)
        # print(cartgvtree.tree_.node_count)
        # print(clf.tree_.node_count)
        # print(cartgvtree.tree_.nodes_impurities, clf.tree_.impurity)
        # print(set(cartgvtree.tree_.value.flat))
        # print(set(clf.tree_.value.flat))
        # print(cartgvtree.tree_.nodes_n_node_samples)
        # print(clf.tree_.n_node_samples)
        # print(clf.tree_.feature)
        # print(cartgvtree.tree_.nodes_group)

        cpred = cartgvtree.predict(X_test)
        spred = clf.predict(X_test)

        obs = np.where(cpred != spred)[0]

        if obs.size > 0:

            print(cartgvtree.score(X_test.to_numpy()[obs],y_test.to_numpy()[obs]))
            print(clf.score(X_test.to_numpy()[obs],y_test.to_numpy()[obs]))



        self.assertCountEqual(cartgvtree.tree_.nodes_group, clf.tree_.feature)
        self.assertCountEqual(cartgvtree.tree_.nodes_n_node_samples, clf.tree_.n_node_samples)
        self.assertSetEqual(set(cartgvtree.tree_.nodes_n_node_samples), set(clf.tree_.n_node_samples))
        self.assertCountEqual(cartgvtree.tree_.value.flat, clf.tree_.value.flat)
        self.assertSetEqual(set(cartgvtree.tree_.value.flat),set(clf.tree_.value.flat))  # A way to compare the values but not perfect
        self.assertCountEqual(cartgvtree.tree_.nodes_impurities, clf.tree_.impurity)
        self.assertSetEqual(set(cartgvtree.tree_.nodes_impurities), set(clf.tree_.impurity))
        self.assertEqual(cartgvtree.tree_.node_count, clf.tree_.node_count)
        self.assertEqual(cartgvtree.tree_.n_leaves, clf.tree_.n_leaves)
        self.assertEqual(cartgvtree.tree_.max_depth, clf.tree_.max_depth)


        self.assertEqual(rfgv.score(X_test, y_test), cartgvtree.score(X_test, y_test))
        self.assertEqual(cartgvtree.score(X_test, y_test), clf.score(X_test, y_test))
        self.assertEqual(rfgv.score(X_test, y_test), clf.score(X_test, y_test))

        np.testing.assert_array_equal(rfgv.predict(X_test), cartgvtree.predict(X_test))
        np.testing.assert_array_equal(cartgvtree.predict(X_test), clf.predict(X_test))
        np.testing.assert_array_equal(rfgv.predict(X_test), clf.predict(X_test))

    def test_score_diff_scikit_CARTGV(self):

        df = pd.read_csv('../data/training.csv', sep=",")
        df_test = pd.read_csv('../data/testing.csv', sep=",")

        # df = df[df['class'].isin(['grass ', 'building ']) ]
        # df_test = df_test[df_test['class'].isin(['grass ', 'building '])]

        X = df.iloc[:, 1:]

        y = df['class']

        X_test = df_test.iloc[:, 1:]
        y_test = df_test['class']

        # df = pd.read_csv('../data/titanic.csv')
        #
        # df.dropna(inplace=True)
        #
        # le = LabelEncoder()
        #
        # df = df.apply(le.fit_transform)
        #
        # X = df.iloc[:, 4:]
        # y = df['survived'][:]

        # X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        cpred = cartgvtree.predict(X_test)
        spred = clf.predict(X_test)

        obs = np.where(cpred != spred)[0]

        print("Score on diff between sklearn and CARTGV")

        if obs.size > 0:

            print(cartgvtree.score(X_test.to_numpy()[obs],y_test.to_numpy()[obs]))
            print(clf.score(X_test.to_numpy()[obs],y_test.to_numpy()[obs]))

        self.assertEqual(cartgvtree.score(X_test.to_numpy()[obs],y_test.to_numpy()[obs]),clf.score(X_test.to_numpy()[obs],y_test.to_numpy()[obs]))
        np.testing.assert_array_equal(cpred,spred)

    def test_predict_CART_bin_fict_data(self):

        n = 1000
        p = 10
        y = np.random.binomial(1,0.5,n)
        y = [-1 if y[i] == 0 else y[i] for i in range(n)]
        X = np.zeros((n,p))
        for i in range(n):
            u = np.random.uniform(1)
            if u > 0.3:
                for j in range(3):
                    X[i,j] = y[i] * np.random.normal(j, 1 ,1)

                for j in range(4,6):
                    X[i, j] = y[i] * np.random.normal(0, 1, 1)

                if (p>6):
                    X[i,7:p] = np.random.normal(0,p-6,1)
            else:
                for j in range(3):
                    X[i, j] = y[i] * np.random.normal(0, 1, 1)

                for j in range(4,6):
                    X[i, j] = y[i] * np.random.normal(j-3, 1, 1)

                if (p > 6):
                    X[i, 7:p] = np.random.normal(0, p-6, 1)

        X = np.array(X)
        y = np.array(y)

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        print(X)

        groups = [[i] for i in range(X.shape[1])]
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 1
        n_obs = y.shape[0]
        rfgv = RFGVClassifier(n_jobs=1, n_estimators=n_estimators, mvar=len_groups, mgroup=len(groups),
                              max_depth_splitting_tree=1,
                              random_state=RANDOM_STATE, verbose=0, random=False)  # verbose >1 for text
        rfgv.fit(X[:n_obs], y[:n_obs], groups)

        cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups, mgroup=len(groups), random_state=RANDOM_STATE,
                                                  max_depth_splitting_tree=1)
        cartgvtree.fit(X[:n_obs], y[:n_obs], groups)

        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        clf.fit(X[:n_obs], y[:n_obs])

        cpred = cartgvtree.predict(X_test)
        spred = clf.predict(X_test)

        obs = np.where(cpred != spred)[0]

        print("CART fictive dataset predict compare")

        if obs.size > 0:
            print(cartgvtree.score(X_test[obs], y_test[obs]))
            print(clf.score(X_test[obs], y_test[obs]))

            self.assertEqual(cartgvtree.score(X_test[obs], y_test[obs]),
                             clf.score(X_test[obs], y_test[obs]))

        print(cartgvtree.tree_.n_leaves)
        print(clf.tree_.n_leaves)
        print(cartgvtree.tree_.max_depth)
        print(clf.tree_.max_depth)
        print(cartgvtree.tree_.node_count)
        print(clf.tree_.node_count)
        print(cartgvtree.tree_.nodes_impurities, clf.tree_.impurity)
        print(set(cartgvtree.tree_.value.flat))
        print(set(clf.tree_.value.flat))
        print(cartgvtree.tree_.nodes_n_node_samples)
        print(clf.tree_.n_node_samples)
        print(clf.tree_.feature)
        print(cartgvtree.tree_.nodes_group)

        self.assertCountEqual(cartgvtree.tree_.nodes_group, clf.tree_.feature)
        self.assertCountEqual(cartgvtree.tree_.nodes_n_node_samples, clf.tree_.n_node_samples)
        self.assertSetEqual(set(cartgvtree.tree_.nodes_n_node_samples), set(clf.tree_.n_node_samples))
        self.assertCountEqual(cartgvtree.tree_.value.flat, clf.tree_.value.flat)
        self.assertSetEqual(set(cartgvtree.tree_.value.flat),
                            set(clf.tree_.value.flat))  # A way to compare the values but not perfect
        self.assertCountEqual(cartgvtree.tree_.nodes_impurities, clf.tree_.impurity)
        self.assertSetEqual(set(cartgvtree.tree_.nodes_impurities), set(clf.tree_.impurity))
        self.assertEqual(cartgvtree.tree_.node_count, clf.tree_.node_count)
        self.assertEqual(cartgvtree.tree_.n_leaves, clf.tree_.n_leaves)
        self.assertEqual(cartgvtree.tree_.max_depth, clf.tree_.max_depth)

        self.assertEqual(rfgv.score(X_test, y_test), cartgvtree.score(X_test, y_test))
        self.assertEqual(cartgvtree.score(X_test, y_test), clf.score(X_test, y_test))
        self.assertEqual(rfgv.score(X_test, y_test), clf.score(X_test, y_test))

        np.testing.assert_array_equal(rfgv.predict(X_test), cartgvtree.predict(X_test))
        np.testing.assert_array_equal(cartgvtree.predict(X_test), clf.predict(X_test))
        np.testing.assert_array_equal(rfgv.predict(X_test), clf.predict(X_test))

    def _importance_variable(self):

        importances = np.empty((6,5))
        sk_importances = np.empty((6,5))

        for k in range(1):
            n = 500
            p = 6
            y = np.random.binomial(1, 0.5, n)
            y = [-1 if y[i] == 0 else y[i] for i in range(n)]
            X = np.zeros((n, p))
            for i in range(n):
                u = np.random.uniform(size=1)
                if u > 0.3:
                    for j in range(3):
                        X[i, j] = y[i] * np.random.normal(j, 1, 1)

                    for j in range(3, 6):
                        X[i, j] = y[i] * np.random.normal(0, 1, 1)

                    if (p > 6):
                        X[i, 7:p] = np.random.normal(0, p-6, 1)
                else:
                    for j in range(3):
                        X[i, j] = y[i] * np.random.normal(0, 1, 1)

                    for j in range(3, 6):
                        X[i, j] = y[i] * np.random.normal(j - 3, 1, 1)

                    if (p > 6):
                        X[i, 7:p] = np.random.normal(0, p-6, 1)

            X = np.array(X)
            y = np.array(y)

            X = X/np.linalg.norm(X,axis=0)

            X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

            # print(np.histogram(X, density = True))

            print(X)

            groups = [[i] for i in range(X.shape[1])]
            print(groups)
            len_groups = np.array([len(group) for group in groups])

            n_estimators = 500
            n_obs = y.shape[0]
            rfgv = RFGVClassifier(n_jobs=4, n_estimators=n_estimators, mvar=1, mgroup=np.sqrt(len(groups)),
                                  max_depth_splitting_tree=1,
                                  random_state=RANDOM_STATE, verbose=0)  # verbose >1 for text
            rfgv.fit(X[:n_obs], y[:n_obs], groups)
            importance = rfgv._permutation_importance(importance="breiman", n_jobs=1, n_repeats=5)
            print(importance)
            # importances.append(importance["importances"])
            if k == 0:
                importances = importance["importances"]
            else:
                print(importances)
                print(importance["importances"])
                importances = np.concatenate((importances, importance["importances"]), axis=1)

            rf = RandomForestClassifier(n_jobs=4, n_estimators=n_estimators, max_features="sqrt",
                                  random_state=RANDOM_STATE, verbose=0)
            rf.fit(X[:n_obs], y[:n_obs])

            sk_importance = permutation_importance(rf,X_test[:n_obs], y_test[:n_obs], n_jobs=1, n_repeats=5)
            print(sk_importance)
            # sk_importances.append(sk_importance["importances"])
            if k == 0:
                sk_importances = sk_importance["importances"]
            else:
                sk_importances = np.concatenate((sk_importances,sk_importance["importances"]), axis=1)

        print(sk_importances)

        plt.boxplot(np.transpose(sk_importances))
        plt.show()

        print(importances)

        plt.boxplot(np.transpose(importances))
        plt.show()
        # print(rfgv._permutation_importance(importance="ishwaran", n_jobs=1, n_repeats=5))

    def _pickling(self):

        n = 20
        p = 6
        y = np.random.binomial(1, 0.5, n)
        y = [-1 if y[i] == 0 else y[i] for i in range(n)]
        X = np.zeros((n, p))
        for i in range(n):
            u = np.random.uniform(size=1)
            if u > 0.3:
                for j in range(3):
                    X[i, j] = y[i] * np.random.normal(j, 1, 1)

                for j in range(3, 6):
                    X[i, j] = y[i] * np.random.normal(0, 1, 1)

                if (p > 6):
                    X[i, 7:p] = np.random.normal(0, p - 6, 1)
            else:
                for j in range(3):
                    X[i, j] = y[i] * np.random.normal(0, 1, 1)

                for j in range(3, 6):
                    X[i, j] = y[i] * np.random.normal(j - 3, 1, 1)

                if (p > 6):
                    X[i, 7:p] = np.random.normal(0, p - 6, 1)

        X = np.array(X)
        y = np.array(y)

        X = X / np.linalg.norm(X, axis=0)

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X.shape[1])]
        print(groups)
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 5
        n_obs = y.shape[0]

        clf = DecisionTreeClassifier()
        clf.fit(X[:n_obs], y[:n_obs])

        print(clf.__getstate__())
        print("###############################################################")
        print(clf.tree_.__getstate__())

        ctgvcf = DecisionCARTGVTreeClassifier(max_depth_splitting_tree=1)
        ctgvcf.fit(X[:n_obs], y[:n_obs],groups)

        print("###############################################################")
        print(ctgvcf.__getstate__())
        print("###############################################################")
        print(ctgvcf.tree_.__getstate__())

        # rf = RandomForestClassifier(n_jobs=4, n_estimators=n_estimators, max_features="sqrt",
        #                             random_state=RANDOM_STATE, verbose=0)
        # rf.fit(X[:n_obs], y[:n_obs])
        #
        #
        # rfgv = RFGVClassifier(n_jobs=4, n_estimators=n_estimators, mvar=1, mgroup=np.sqrt(len(groups)),
        #                       max_depth_splitting_tree=1,
        #                       random_state=RANDOM_STATE, verbose=0)  # verbose >1 for text
        # rfgv.fit(X[:n_obs], y[:n_obs], groups)

    def test_sobol_indice(self):

        n = 200
        p = 6
        y = np.random.binomial(1, 0.5, n)
        y = [-1 if y[i] == 0 else y[i] for i in range(n)]
        X = np.zeros((n, p))
        for i in range(n):
            u = np.random.uniform(size=1)
            if u > 0.3:
                for j in range(3):
                    X[i, j] = y[i] * np.random.normal(j, 1, 1)

                for j in range(3, 6):
                    X[i, j] = y[i] * np.random.normal(0, 1, 1)

                if (p > 6):
                    X[i, 7:p] = np.random.normal(0, p - 6, 1)
            else:
                for j in range(3):
                    X[i, j] = y[i] * np.random.normal(0, 1, 1)

                for j in range(3, 6):
                    X[i, j] = y[i] * np.random.normal(j - 3, 1, 1)

                if (p > 6):
                    X[i, 7:p] = np.random.normal(0, p - 6, 1)

        X = np.array(X)
        y = np.array(y)

        X = X / np.linalg.norm(X, axis=0)

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        groups = [[i] for i in range(X.shape[1])]
        print(groups)
        len_groups = np.array([len(group) for group in groups])

        n_estimators = 5
        n_obs = y.shape[0]

        ctgvcf = DecisionCARTGVTreeClassifier(max_depth_splitting_tree=1)
        ctgvcf.fit(X[:n_obs], y[:n_obs], groups)

        print("Sobol indice")
        print(ctgvcf.tree_.nodes_group)

        _get_n_samples_bootstrap, _generate_sample_indices, _generate_unsampled_indices

        n_samples = y.shape[0]

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, None
        )

        sampled_indices = _generate_sample_indices(ctgvcf.random_state, n_samples, n_samples_bootstrap)
        unsampled_indices = _generate_unsampled_indices(ctgvcf.random_state, n_samples, n_samples_bootstrap)

        print(ctgvcf.tree_.sobol_indice(np.array(X, dtype=np.float32),5,sampled_indices,unsampled_indices))

    def test_sobol_example(self):
        n = 50
        p = 2
        y = np.random.binomial(1, 0.5, n)
        y = [-1 if y[i] == 0 else y[i] for i in range(n)]
        X = np.zeros((n, p))
        for i in range(n):
            u = np.random.uniform(size=1)
            if u > 0.3:
                for j in range(2):
                    X[i, j] = y[i] * np.random.normal(j, 1, 1)

                # for j in range(3, 6):
                #     X[i, j] = y[i] * np.random.normal(0, 1, 1)
                #
                # if (p > 6):
                #     X[i, 7:p] = np.random.normal(0, p - 6, 1)
            else:
                for j in range(2):
                    X[i, j] = y[i] * np.random.normal(0, 1, 1)

                # for j in range(3, 6):
                #     X[i, j] = y[i] * np.random.normal(j - 3, 1, 1)
                #
                # if (p > 6):
                #     X[i, 7:p] = np.random.normal(0, p - 6, 1)

        X = np.array(X)
        y = np.array(y)

        X = X / np.linalg.norm(X, axis=0)

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, random_state=6761474)

        print(X)
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        plot_tree(clf)
        plt.show()

if __name__ == '__main__':
    unittest.main()
