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

# from CARTGV_trees import DecisionCARTGVTreeClassifier, DecisionCARTGVTreeRegressor
# from RFGV import RFGVClassifier, RFGVRegressor, _get_n_samples_bootstrap, _generate_sample_indices, _generate_unsampled_indices

from rfgi.CARTGV_trees import DecisionCARTGVTreeClassifier, DecisionCARTGVTreeRegressor
from rfgi.RFGV import RFGVClassifier, RFGVRegressor, _get_n_samples_bootstrap, _generate_sample_indices, _generate_unsampled_indices

df = pd.read_csv('../data/training.csv', sep=",")
df_test = pd.read_csv('../data/testing.csv', sep=",")

df = df[df['class'].isin(['grass ', 'building ']) ]
df_test = df_test[df_test['class'].isin(['grass ', 'building ']) ]

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

# groups = [[i] for i in range(X.shape[1])] # group for CART
groups = [[0], [1]]
len_groups = np.array([len(group) for group in groups])

cartgvtree = DecisionCARTGVTreeClassifier(mvar=len_groups,mgroup=len(groups), random_state=2547, max_depth_splitting_tree=2)

cartgvtree.fit(X.iloc[:,:2],y,groups)
# file = open("cartgvtree_size.txt", "wb")
# pickle.dump(obj=cartgvtree, file=file)
print("Score prédiction multi-classes CARTGV (CART) : " + str(cartgvtree.score(X_test.iloc[:,:2],y_test)))
print(pd.DataFrame(cartgvtree.predict(X_test.iloc[:,:2])))
print(cartgvtree.get_n_leaves())
print(cartgvtree.get_depth())
print(cartgvtree.apply(X_test.to_numpy()[0][:2].reshape(1,-1)))
print(cartgvtree.predict_proba(X_test.iloc[:,:2]))
print(cartgvtree.predict_log_proba(X_test.iloc[:,:2]))

# importance = permutation_importance(cartgvtree,X,y,random_state=2547)
# print(importance.importances_mean)
# print(importance.importances_std)
# print(importance.importances)
# print(len(importance.importances_mean))
# print(X.shape)


clf = DecisionTreeClassifier(random_state=2547)
clf.fit(X,y)
print(pd.DataFrame(clf.predict(X_test)))
print("Score prédiction multi-classes Scikit-learn : " + str(clf.score(X_test,y_test)))

df1 = pd.DataFrame(cartgvtree.predict(X_test.iloc[:,:2]))
df2 = pd.DataFrame(clf.predict(X_test))
diff = pd.concat([df1,df2]).drop_duplicates(keep=False)
print(diff)
print(diff.shape)
print(diff.index)
indexes = diff.index
print(indexes)
if indexes.size > 0:
    print(clf.decision_path(X_test[np.array(indexes)]))
# fig, ax = plt.subplots(1,figsize=(16,9))
# plot_tree(clf,ax=ax)
# plt.show()

# print(len_groups)
# from sklearn.utils import check_random_state
# random_state = check_random_state(2547)
# X_permuted = np.array(X).copy()
# print(X_permuted)
# shuffling_idx = np.arange(X.shape[0])
# print(shuffling_idx)
# random_state.shuffle(shuffling_idx)
# print(shuffling_idx)
#
# X_permuted = pd.DataFrame(X_permuted)
#
# print(X_permuted.iloc[73, g9_idx])
# col = X_permuted.iloc[shuffling_idx, g9_idx] # For pandas DataFrame
# print(col)
# col.index = X_permuted.index
# X_permuted.iloc[:, g9_idx] = col
# print(np.array(X_permuted))

# X_permuted[:, g9_idx] = X_permuted[shuffling_idx, g9_idx] # For numpy array
# X_permuted = np.take_along_axis(X_permuted[:, g9_idx],1)
# print(X_permuted[:,g9_idx])
# print(X_permuted.shape)
# print(X_permuted[shuffling_idx,g9_idx])

# X_permuted = np.put_along_axis(X_permuted[:,g9_idx],shuffling_idx,X_permuted[shuffling_idx, g9_idx], axis=1)
# print(X_permuted[:, g9_idx])
n_estimators = 5
n_obs = y.shape[0]
rfgv = RFGVClassifier(n_jobs=4, n_estimators=n_estimators, mvar="root", mgroup=1, max_depth_splitting_tree=2, random_state=2547, group_importance="Breiman", verbose=0) #verbose >1 for text
start = time.time()
rfgv.fit(X.iloc[:n_obs], y[:n_obs], groups)
end = time.time()
print("Temps construction RFGV Classification avec " + str(n_estimators) + " arbres : " + str(end-start) + " secondes")
# print("Classification Score " + str(rfgv.score(X_test, y_test)))
# print("Classification OOB score : " + str(rfgv.oob_score_))
# print("Classification IB score : " + str(rfgv.ib_score_))

n_samples_bootstrap = _get_n_samples_bootstrap(y.shape[0], None)
sample_indices = _generate_sample_indices(2547, y.shape[0], n_samples_bootstrap)
unsampled_indices = _generate_unsampled_indices(2547, y.shape[0], n_samples_bootstrap)

print(np.bincount(sample_indices))
print(np.bincount(unsampled_indices))
print(np.where(np.bincount(sample_indices) != 0))
print(np.where(np.bincount(unsampled_indices) != 0))
# print(rfgv.importances)
# print(rfgv.permutation_importance(X.iloc[:n_obs,:2], y[:n_obs], groups=groups, importance="Breiman", random_state=2547))
# print(rfgv.permutation_importance(cartgvtree,X.iloc[:n_obs,:2], y[:n_obs], random_state=2547, n_repeats=1))
print(rfgv._permutation_importance(importance="breiman", n_jobs=1, n_repeats=5))
# print(rfgv._permutation_importance(importance="ishwaran", n_jobs=1, n_repeats=5))

rfc = RandomForestClassifier(n_jobs=1, n_estimators=n_estimators, random_state=2547)
rfc.fit(X.iloc[:n_obs,:2], y[:n_obs])
# print(permutation_importance(rfc, X.iloc[:n_obs,:2], y[:n_obs], random_state=2547))


# clf = DecisionTreeClassifier()
# clf.tree_ = cartgvtree.tree_.nodes_splitting_trees[0]
# fig, ax = plt.subplots(1,figsize=(16,9))
# plot_tree(clf,ax=ax)
# plt.show()

# datas = fetch_california_housing(return_X_y=True)
# datas = load_boston(return_X_y=True)
# datas = load_diabetes(return_X_y=True)
# X = datas[0]
# y = datas[1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(X_train.shape)
# groups = np.array([[2,3,4],[6,7],[0],[4,7,5],[1,3]], dtype=object) # Group for california
# groups = np.array([[0,1,2,7,10,11,12],[5,6,9,12],[3,4,8]], dtype=object) # Group boston
# groups = np.array([[0,1,2],[2,3,9],[4,7,8,9],[4,5,8],[4,6,8]]) # Group diabete
# groups = [[i] for i in range(X.shape[1])] # Group for CART
# len_groups = np.array([len(group) for group in groups])
# groups = np.array(list(map(lambda group: np.pad(group, (0, max(len_groups) - len(group)), constant_values=-1), groups)))

# cartgvtree = DecisionCARTGVTreeRegressor(mvar=len_groups, mgroup=len(groups), random_state=2547, max_depth_splitting_tree=1)
# start = time.time()
# cartgvtree.fit(X_train, y_train, groups)
# end = time.time()
# print("Temps Construction Tree Regression : " + str(end-start))
# print("Score prédiction régression CARTGV (CART) : " + str(cartgvtree.score(X_test, y_test)))
# print(pd.DataFrame(cartgvtree.predict(X_test)))
# print(cartgvtree.get_n_leaves())
# print(cartgvtree.get_depth())
# print(cartgvtree.apply(X_test[0].reshape(1,-1)))

# dtr = DecisionTreeRegressor(random_state=2547)
# dtr.fit(X_train,y_train)
# print("Score prédiction régression Scikit-learn : " + str(dtr.score(X_test, y_test)))

# df1 = pd.DataFrame(cartgvtree.predict(X_test))
# df2 = pd.DataFrame(dtr.predict(X_test))
# diff = pd.concat([df1,df2]).drop_duplicates(keep=False)
# print(diff)
# print(diff.shape)
# print(diff.index)
# indexes = diff.index
# print(indexes)

# print(dtr.decision_path(X_test[np.array(indexes)]))
# fig, ax = plt.subplots(1,figsize=(16,9))
# plot_tree(dtr,ax=ax)
# plt.show()

# n_estimators = 100
# rfgv = RFGVRegressor(n_jobs=8, n_estimators=n_estimators, mvar="third", mgroup=len(groups)/3, random_state=2547, max_depth=2, max_depth_splitting_tree=2, verbose=2, oob_score=True, ib_score=True)
# start = time.time()
# rfgv.fit(X_train, y_train, groups)
# end = time.time()
# print("Temps construction RFGV Regression avec " + str(n_estimators) + " arbres : " + str(end-start) + " secondes")
# print("Regression Score " + str(rfgv.score(X_test, y_test)))
# print("Regression OOB score : " + str(rfgv.oob_score_))
# print("Regression IB score : " + str(rfgv.ib_score_))

# clf = DecisionTreeRegressor()
# clf.tree_ = cartgvtree.tree_.nodes_splitting_trees[20]
# fig, ax = plt.subplots(1,figsize=(16,9))
# plot_tree(clf,ax=ax)
# plt.show()