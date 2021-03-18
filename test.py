# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:35:14 2021

@author: Alphonse
"""
# import sys
# sys.path.append('./')

import numpy as np
import pandas as pd
print("Start import")

from CARTGV import CARTGVSplitter
from CARTGV import CARTGVGini
from CARTGV import CARTGVTree, CARTGVTreeBuilder

from sklearn.utils.validation import check_random_state

def fit(X, y, groups, sample_weight=None, check_input=True,
        X_idx_sorted="deprecated"):
  print("Start")
  n_samples, n_features = X.shape
  n_grouped_features = 2
  y = np.atleast_1d(y)
  max_grouped_features = max([len(groups[i]) for i in range(len(groups))])
  min_samples_leaf = 1
  min_samples_split = 2
  min_weight_leaf = (0.25 * n_samples)
  random_state = check_random_state(0)
  max_depth = 3
  mgroup = 1
  mvar = 10
  min_impurity_decrease = 0.1
  min_impurity_split = 0.0
  
  if y.ndim == 1:
    y = np.reshape(y,(-1,1))
  
  n_outputs = y.shape[1]
  
  y = np.copy(y)
  
  classes = []  
  n_classes = []
  
  y_encoded = np.zeros(y.shape, dtype=int)
  for k in range(n_outputs):
    classes_k, y_encoded[:,k] = np.unique(y[:,k], return_inverse=True)
    classes.append(classes_k)
    n_classes.append(classes_k.shape[0])
    
  y = y_encoded
  
  n_classes = np.array(n_classes, dtype=np.intp)
    
  criterion = CARTGVGini(n_outputs, n_classes)
  
  print("Criterion created")
  
  splitter = CARTGVSplitter(criterion, max_grouped_features, len(groups),
                  min_samples_leaf, min_weight_leaf,
                  random_state)
  
  print("Splitter created")
  
  tree = CARTGVTree(n_grouped_features,n_classes, n_outputs)
  
  print("Tree created")
  
  builder = CARTGVTreeBuilder(splitter, min_samples_split,
                  min_samples_leaf, min_weight_leaf,
                  max_depth, mgroup, mvar, 
                  min_impurity_decrease, min_impurity_split)
  
  print("Builder created")
  
  print("Builder launched ...")
  builder.build(tree, X, y, groups)
  
  print(tree.node_count)
  

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

fit(np.array(X),np.array(y),groups)

# g1 = [df[col] for col in df.columns if '_G1' in col]
# g1_name = [g1[col].name for col in range(len(g1))]
# g1_df = pd.DataFrame(np.array(g1).transpose(), columns=g1_name)

# g2 = [df[col] for col in df.columns if '_G2' in col]
# g2_name = [g2[col].name for col in range(len(g2))]
# g2_df = pd.DataFrame(np.array(g2).transpose(), columns=g2_name)

# g3 = [df[col] for col in df.columns if '_G3' in col]
# g3_name = [g3[col].name for col in range(len(g3))]
# g3_df = pd.DataFrame(np.array(g3).transpose(), columns=g3_name)

# g4 = [df[col] for col in df.columns if '_G4' in col]
# g4_name = [g4[col].name for col in range(len(g4))]
# g4_df = pd.DataFrame(np.array(g4).transpose(), columns=g4_name)

# g5 = [df[col] for col in df.columns if '_G5' in col]
# g5_name = [g5[col].name for col in range(len(g5))]
# g5_df = pd.DataFrame(np.array(g5).transpose(), columns=g5_name)

# print(g1_df)
# print(g2_df)
# print(g3_df)
# print(g4_df)
# print(g5_df)