# -*- coding: utf-8 -*-
"""
@author: Francisco Valente
"""

# imports
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# load data
data_training = np.load('data_training.npy')
label_training = np.load('label_training.npy')
data_evaluation =  np.load('data_evaluation.npy.npy')

# standardization
scaler = StandardScaler()
scaler.fit(data_training,label_training)
data_training = scaler.transform(data_training)
data_evaluation = scaler.transform(data_evaluation)


## correlation analysis
data_T = np.transpose(data_training)
corrs = np.corrcoef(data_T)


## Index of features
# 0-5 : degree centrality
# 6-8 : total neighbors
# 9-11 : common neighbors
# 12-14 : jaccard index
# 15-17 : simpson index 
# 18-20 : geometric index
# 21-23 : cosine index
# 24-26 : adamic-adar
# 27-29 : resource-allocation
# 30-32 : preferential attatchment
# 33-38 : average degree of neighborhood
# 39-44 : clustering coefficient


### Perform principal component analysis (PCA) on groups of correlated features

def perform_pca(set_feat, data_training, data_evaluation):
    
    feat_train = data_training[:,set_feat]
    pca = PCA.fit(feat_train)
    ev = pca.explained_variance_ratio_ # explained variance
    pc = pca.components_ # principal components
    data_pca_train = pca.transform(feat_train)[:,0] # transformed data training
    feat_eval = data_evaluation[:,set_feat]
    data_pca_eval = pca.transform(feat_eval)[:,0] # transformed data evaluation
    
    return data_pca_train, data_pca_eval, ev, pc

# GROUP 1 (degree centrality, node 1)
set_feat1 = [0,2,4]
data_train_g1, data_eval_g1, ev1, pc1 =  perform_pca(set_feat1, data_training, data_evaluation)

# GROUP 2 (degree centrality, node 2)
set_feat2 = [1,3,5]
data_train_g2, data_eval_g2, ev2, pc2 =  perform_pca(set_feat2, data_training, data_evaluation)

# GROUP 3 (total neighbors)
set_feat3 = [6,7,8]
data_train_g3, data_eval_g3, ev3, pc3 =  perform_pca(set_feat3, data_training, data_evaluation)

# GROUP 4 (common neighbors)
set_feat4 = [9,10,11]
data_train_g4, data_eval_g4, ev4, pc4 =  perform_pca(set_feat4, data_training, data_evaluation)

# GROUP 5 (jaccard, geometric and cosine index)
set_feat5 = [12,13,14,18,19,20,21,22,23]
data_train_g5, data_eval_g5, ev5, pc5 =  perform_pca(set_feat5, data_training, data_evaluation)

# GROUP 6 (simpson index)
set_feat6 = [15,16,17]
data_train_g6, data_eval_g6, ev6, pc6 =  perform_pca(set_feat6, data_training, data_evaluation)

# GROUP 7 (adamic-adar, resource allocation and preferential attatchment)
set_feat7 = [24,25,26,27,28,29,30,31,32]
data_train_g7, data_eval_g7, ev7, pc7 =  perform_pca(set_feat7, data_training, data_evaluation)

# remaining features not transformed (average degree of neighborhood, clustering coefficient) 
idx_rem = np.array((range(33, 45)))
data_train_g8 =  data_training[:,idx_rem]
data_eval_g8 =  data_evaluation[:,idx_rem]

# New data training (concatenate all groups)

aux_train = np.vstack((data_train_g1,data_train_g2,data_train_g3,data_train_g4,data_train_g5,data_train_g6,data_train_g7))
new_data_train = np.concatenate((np.transpose(aux_train),data_train_g8), axis=1)

# New data evaluation (concatenate all groups)

aux_eval = np.vstack((data_eval_g1,data_eval_g2,data_eval_g3,data_eval_g4,data_eval_g5,data_eval_g6,data_eval_g7))
new_data_eval = np.concatenate((np.transpose(aux_eval),data_eval_g8), axis=1)

# save new matrix
np.save('data_pca_training.npy', new_data_train)
np.save('data_pca_evaluation.npy', new_data_eval)

