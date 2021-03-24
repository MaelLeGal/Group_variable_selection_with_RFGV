import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pickle
import sys
print("Start import")

from CARTGV import CARTGVSplitter
from CARTGV import CARTGVGini
from CARTGV import CARTGVTree, CARTGVTreeBuilder

from sklearn.utils.validation import check_random_state
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.tree._tree import DepthFirstTreeBuilder, BestFirstTreeBuilder, Tree
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._criterion import Gini,Entropy

df = pd.read_csv('CARTGV/data_Mael.csv',sep=";",index_col=0) #names=("Type", "Y", "V3_G1", "V4_G1", "V5_G1", "V6_G1", "V7_G1", "V8_G2", "V9_G2", "V10_G2", "V11_G2", "V12_G2", "V13_G3", "V14_G3", "V15_G3", "V16_G3", "V17_G3", "V18_G4", "V19_G4", "V20_G4", "V21_G4", "V22_G4", "V23_G5", "V24_G5", "V25_G5", "V26_G5", "V27_G5")
print(df.shape)

train = df.loc[df['Type'] == 'train']
print(train.shape)

X = train.iloc[:,2:]
print(X.shape)

y = train['Y']
print(y.shape)

g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

groups = np.array([g1_idx,g2_idx,g3_idx,g4_idx,g5_idx])

print(groups)

print(X.iloc[:,groups[0]].shape)

clf = DecisionTreeClassifier(max_depth=3,random_state=0)
tree = clf.fit(X.iloc[:,groups[0]],y)
treeS = pickle.dumps(clf.tree_)
treeDS = pickle.loads(treeS)
print(clf.tree_)
print(treeDS)
print(tree)
plot_tree(tree)
plt.show()

# Build tree
n_samples, n_features = X.iloc[:,groups[0]].shape
n_grouped_features = 2
y = np.atleast_1d(y)
max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
max_features = max(1, int(np.sqrt(n_features))) #len(groups[0])
max_leaf_nodes = -1 #X.shape[0]
min_samples_leaf = 1
min_samples_split = 2
min_weight_leaf = (0.25 * n_samples)
random_state = check_random_state(0)
max_depth = np.iinfo(np.int32).max
mgroup = 1
mvar = 10
min_impurity_decrease = 0.
min_impurity_split = 0
sample_weight = None


if y.ndim == 1:
    y = np.reshape(y, (-1, 1))

n_outputs = y.shape[1]

y = np.copy(y)

classes = []
n_classes = []

y_encoded = np.zeros(y.shape, dtype=int)
for k in range(n_outputs):
    classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
    classes.append(classes_k)
    n_classes.append(classes_k.shape[0])

y = y_encoded

n_classes = np.array(n_classes, dtype=np.intp)

criterion = Gini(n_outputs,n_classes)
splitter = BestSplitter(criterion,max_features,min_samples_leaf,min_weight_leaf,random_state)
tree = Tree(n_features,n_classes, n_outputs)
builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                   min_samples_leaf,
                                   min_weight_leaf,
                                   max_depth,
                                   min_impurity_decrease,
                                   min_impurity_split)

print(np.array(X.iloc[:,groups[0]]).shape)
print(np.array(y).shape)
print(tree)
print(np.array(X.iloc[:,groups[0]]))
print(np.array(y))
builder.build(tree, np.array(X.iloc[:,groups[0]]), np.array(y), sample_weight)
clf.tree_ = tree
# clf._prune_tree()
plot_tree(clf)
plt.show()

# tree = pickle.loads(b'\x80\x05\x95\xef\x01\x00\x00\x00\x00\x00\x00\x8c\x12sklearn.tree._tree\x94\x8c\x04Tree\x94\x93\x94K\x19\x8c\x12numpy.core.numeric\x94\x8c\x0b_frombuffer\x94\x93\x94(\x96\x08\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x94\x8c\x05numpy\x94\x8c\x05dtype\x94\x93\x94\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94bK\x01\x85\x94\x8c\x01C\x94t\x94R\x94K\x01\x87\x94R\x94}\x94(\x8c\tmax_depth\x94K\x00\x8c\nnode_count\x94K\x00\x8c\x05nodes\x94h\x05(\x96\x00\x00\x00\x00\x00\x00\x00\x00\x94h\t\x8c\x03V56\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94N(\x8c\nleft_child\x94\x8c\x0bright_child\x94\x8c\x07feature\x94\x8c\tthreshold\x94\x8c\x08impurity\x94\x8c\x0en_node_samples\x94\x8c\x17weighted_n_node_samples\x94t\x94}\x94(h\x1eh\x0cK\x00\x86\x94h\x1fh\x0cK\x08\x86\x94h h\x0cK\x10\x86\x94h!h\t\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03h\rNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94bK\x18\x86\x94h"h,K \x86\x94h#h\x0cK(\x86\x94h$h,K0\x86\x94uK8K\x01K\x10t\x94bK\x00\x85\x94h\x10t\x94R\x94\x8c\x06values\x94h\x05(\x96\x00\x00\x00\x00\x00\x00\x00\x00\x94h,K\x00K\x01K\x02\x87\x94h\x10t\x94R\x94ub.')
# print(tree)



