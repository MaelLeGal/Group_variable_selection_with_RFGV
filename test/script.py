import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from CARTGVCriterion import CARTGVGini, CARTGVMSE
from CARTGVTree import CARTGVTree, CARTGVTreeBuilder
from CARTGVSplitter import BestCARTGVSplitter

from sklearn.utils.validation import check_random_state
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Import the data into a dataframe
df = pd.read_csv('CARTGV/data_Mael.csv', sep=";", index_col=0)

# Select only the train samples
train = df.loc[df['Type'] == 'train']

# Remove the first two columns of teh dataframe
X = train.iloc[:, 2:]

# Select the responses for the train samples
y = train['Y']

# Select the index of each column for each group
g1_idx = [col for col in range(len(X.columns)) if '_G1' in X.columns[col]]
g2_idx = [col for col in range(len(X.columns)) if '_G2' in X.columns[col]]
g3_idx = [col for col in range(len(X.columns)) if '_G3' in X.columns[col]]
g4_idx = [col for col in range(len(X.columns)) if '_G4' in X.columns[col]]
g5_idx = [col for col in range(len(X.columns)) if '_G5' in X.columns[col]]

# Represent the groups as an array containing the index of the group column/variable inside the dataframe
groups = np.array([g1_idx, g2_idx, g3_idx, g4_idx, g5_idx])

n_samples, n_features = X.shape # The number of samples, and the number of variables
n_grouped_features = 5 # The number of group ?
y = np.atleast_1d(y)
max_grouped_features = max([len(groups[i]) for i in range(len(groups))]) # The maximal number of features in the different group
min_samples_leaf = 1 # The minimum number of samples in a leaf
min_samples_split = 2 # The minimum number of samples needed to split the node
min_weight_leaf = 0 # The minimum weight in a leaf
random_state = check_random_state(2547) # The seed
max_depth = 3 # The maximal depth for the splitting_tree
mgroup = 5 # The number of group to visit
mvar = 5 # The number of variable to visit
min_impurity_decrease = 0.1 # The minimum decrease in impurity that we want to achieve after a split
min_impurity_split = 0.1 # The minimum impurity under which the node is considered a leaf

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
times = np.ndarray(1)
startLoop = time.time()
for i in range(1):
    start = time.time()
    criterion = CARTGVGini(n_outputs, n_classes)

    splitter = BestCARTGVSplitter(criterion, max_grouped_features, len(groups),
                              min_samples_leaf, min_weight_leaf,
                              random_state)

    tree = CARTGVTree(n_grouped_features, n_classes, n_outputs)

    builder = CARTGVTreeBuilder(splitter, min_samples_split,
                                min_samples_leaf, min_weight_leaf,
                                max_depth, mgroup, mvar,
                                min_impurity_decrease, min_impurity_split)

    # builder.build(tree, X.to_numpy(dtype=np.float32), y, groups, None) #X.to_numpy(dtype=np.float32)
    builder.test_build(tree, X.to_numpy(dtype=np.float32), y, groups) #X.to_numpy(dtype=np.float32)
    end = time.time()
    times[i] = end-start
print("Mean Time for 1 tree : " + str(np.mean(times)))
endLoop = time.time()
print("Time Loop : " + str(endLoop-startLoop))
clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, max_features=len(groups[0]),
                             max_leaf_nodes=X.shape[0])

for i in range(tree.node_count):
    clf.tree_ = tree.nodes_splitting_trees[i]
    if(tree.nodes_splitting_trees[i] != None):
        fig, ax = plt.subplots(1, figsize=(16, 9))
        plot_tree(clf)
        plt.show()

# print(tree.nodes_childs)
print(tree.nodes_parent)
print(tree.nodes_impurities)
print(tree.nodes_n_node_samples)
print(tree.nodes_weighted_n_node_samples)
print(tree.nodes_group)
print(tree.nodes_n_childs)

print(tree.node_count)
print(tree.max_depth)
print(tree.n_grouped_features)
print(tree.n_outputs)
print(tree.n_classes)
print(tree.max_n_classes)
print(tree.value_stride)
print(tree.capacity)
print(tree.value) #Not sure what it represent
# print(tree.nodes) #TODO find a way to make it possible