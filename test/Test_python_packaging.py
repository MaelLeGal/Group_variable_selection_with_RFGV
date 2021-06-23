import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, fetch_california_housing, load_boston, load_diabetes

from CARTGV_trees import DecisionCARTGVTreeClassifier, DecisionCARTGVTreeRegressor
from RFGV import RFGVClassifier, RFGVRegressor

df = pd.read_csv('../data/training.csv', sep=",")
df_test = pd.read_csv('../data/testing.csv', sep=",")

X = df.iloc[:, 1:]

y = df['class']

X_test = df_test.iloc[:, 1:]
y_test = df_test['class']

g0_idx = [col for col in range(len(X.columns)) if ('_40' not in X.columns[col] and '_60' not in X.columns[col] and '_80' not in X.columns[col] and '_100' not in X.columns[col] and '_120' not in X.columns[col] and '_140' not in X.columns[col])]
g0_idx = np.array([col for col in range(20)])
g1_idx = np.array([col for col in range(len(X.columns)) if '_40' in X.columns[col]])
g2_idx = np.array([col for col in range(len(X.columns)) if '_60' in X.columns[col]])
g3_idx = np.array([col for col in range(len(X.columns)) if '_80' in X.columns[col]])
g4_idx = np.array([col for col in range(len(X.columns)) if '_100' in X.columns[col]])
g5_idx = np.array([col for col in range(len(X.columns)) if '_120' in X.columns[col]])
g6_idx = np.array([col for col in range(len(X.columns)) if '_140' in X.columns[col]])
g7_idx = np.hstack([[col for col in range(len(X.columns)) if 'Bright' in X.columns[col]], [col for col in range(len(X.columns)) if 'Mean_' in X.columns[col]], [col for col in range(len(X.columns)) if 'NDVI' in X.columns[col]]])
g8_idx = np.hstack([[col for col in range(len(X.columns)) if 'SD_' in X.columns[col]], [col for col in range(len(X.columns)) if 'GLCM' in X.columns[col]]])
g9_idx = np.array([X.columns.get_loc("BrdIndx"),X.columns.get_loc("Area"),X.columns.get_loc("Round"),X.columns.get_loc("Compact"),X.columns.get_loc("ShpIndx"),X.columns.get_loc("LW"),X.columns.get_loc("Rect"),X.columns.get_loc("Dens"),X.columns.get_loc("Assym"),X.columns.get_loc("BordLngth")])

groups = np.array([g0_idx, g1_idx, g2_idx, g3_idx, g4_idx, g5_idx, g6_idx, g7_idx, g8_idx, g9_idx], dtype=object)
len_groups = np.array([len(group) for group in groups])

cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups,mgroup=len(groups), random_state=2547)

cartgvtree.fit(X,y,groups)
print(cartgvtree.score(X_test,y_test))
print(pd.DataFrame(cartgvtree.predict(X_test)))
print(cartgvtree.get_n_leaves())
print(cartgvtree.get_depth())
print(cartgvtree.apply(X_test.to_numpy()[0].reshape(1,-1)))
print(cartgvtree.predict_proba(X_test))
print(cartgvtree.predict_log_proba(X_test))

n_estimators = 100
rfgv = RFGVClassifier(n_jobs=8, n_estimators=n_estimators, mvar="root", mgroup=len(groups)/3, random_state=2547, oob_score=True, ib_score=True, verbose=2)
start = time.time()
rfgv.fit(X, y, groups)
end = time.time()
print("Temps construction RFGV Classification avec " + str(n_estimators) + " arbres : " + str(end-start) + " secondes")
print("Classification Score " + str(rfgv.score(X_test, y_test)))
print("Classification OOB score : " + str(rfgv.oob_score_))
print("Classification IB score : " + str(rfgv.ib_score_))

# clf = DecisionTreeClassifier()
# clf.tree_ = cartgvtree.tree_.nodes_splitting_trees[0]
# fig, ax = plt.subplots(1,figsize=(16,9))
# plot_tree(clf,ax=ax)
# plt.show()

datas = fetch_california_housing(return_X_y=True)
# datas = load_boston(return_X_y=True)
# datas = load_diabetes(return_X_y=True)
X = datas[0]
y = datas[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(X_train.shape)
groups = np.array([[2,3,4],[6,7],[0],[4,7,5],[1,3]], dtype=object) # Group for california
# groups = np.array([[0,1,2,7,10,11,12],[5,6,9,12],[3,4,8]], dtype=object) # Group boston
# groups = np.array([[0,1,2],[2,3,9],[4,7,8,9],[4,5,8],[4,6,8]]) # Group diabete
len_groups = np.array([len(group) for group in groups])
groups = np.array(list(map(lambda group: np.pad(group, (0, max(len_groups) - len(group)), constant_values=-1), groups)))

cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=2547, max_depth_splitting_tree=2)
start = time.time()
cartgvtree.fit(X_train, y_train, groups)
end = time.time()
print("Temps Construction Tree Regression : " + str(end-start))
print(cartgvtree.score(X_test, y_test))
print(pd.DataFrame(cartgvtree.predict(X_test)))
print(cartgvtree.get_n_leaves())
print(cartgvtree.get_depth())
print(cartgvtree.apply(X_test[0].reshape(1,-1)))

n_estimators = 100
rfgv = RFGVRegressor(n_jobs=8, n_estimators=n_estimators, mvar="third", mgroup=len(groups)/3, random_state=2547, max_depth=2, max_depth_splitting_tree=2, verbose=2, oob_score=True, ib_score=True)
start = time.time()
rfgv.fit(X_train, y_train, groups)
end = time.time()
print("Temps construction RFGV Regression avec " + str(n_estimators) + " arbres : " + str(end-start) + " secondes")
print("Regression Score " + str(rfgv.score(X_test, y_test)))
print("Regression OOB score : " + str(rfgv.oob_score_))
print("Regression IB score : " + str(rfgv.ib_score_))

# clf = DecisionTreeRegressor()
# clf.tree_ = cartgvtree.tree_.nodes_splitting_trees[20]
# fig, ax = plt.subplots(1,figsize=(16,9))
# plot_tree(clf,ax=ax)
# plt.show()