import os
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Perform principal component analysis (PCA) on groups of correlated features
def perform_pca(set_feat, data_training, data_evaluation):
    
    feat_train = data_training[:,set_feat]
    pca = PCA()
    pca = pca.fit(feat_train)
    ev = pca.explained_variance_ratio_ # explained variance
    pc = pca.components_ # principal components
    data_pca_train = pca.transform(feat_train)[:,0] # transformed training data
    feat_eval = data_evaluation[:,set_feat]
    data_pca_eval = pca.transform(feat_eval)[:,0] # transformed evaluation data
    
    return data_pca_train, data_pca_eval, ev, pc


def pca_data(data_source):

    # load data
    folder_to_load = os.path.join(os.getcwd(),'data_extracted')
    training_features_file_name = "TrainingFeatures_" + data_source + ".npy"
    training_label_file_name = "TrainingLabel_" + data_source + ".npy"
    evaluation_features_file_name = "EvaluationFeatures_" + data_source + ".npy"
    training_features_file = os.path.join(folder_to_load, training_features_file_name)
    training_label_file = os.path.join(folder_to_load, training_label_file_name)
    evaluation_features_file = os.path.join(folder_to_load, evaluation_features_file_name)
    data_training = np.load(training_features_file)
    size_data_training = int(len(data_training)/2)
    label_training = np.concatenate((np.ones(size_data_training), np.zeros(size_data_training)))
    data_evaluation =  np.load(evaluation_features_file)
    
    # standardization
    scaler = StandardScaler()
    scaler.fit(data_training,label_training)
    data_training = scaler.transform(data_training)
    data_evaluation = scaler.transform(data_evaluation)
    
    
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
    
    # New data training (concatenate all groups)
    new_data_train = np.stack((data_train_g1,data_train_g2,data_train_g3,data_train_g4,data_train_g5,data_train_g6,data_train_g7),axis=1)
    
    # New data evaluation (concatenate all groups)
    new_data_eval = np.stack((data_eval_g1,data_eval_g2,data_eval_g3,data_eval_g4,data_eval_g5,data_eval_g6,data_eval_g7), axis=1)
    
    return new_data_train, new_data_eval, label_training
    

def get_predictions(data_train, label_train, data_eval):

    clf = RandomForestClassifier(n_jobs=6, n_estimators=1250, min_samples_leaf=5, min_samples_split= 7)
    model = clf.fit(data_train, label_train)
    eval_predictions = model.predict_proba(data_eval)[:,1]
    ordered_eval_predictions = np.flip(np.argsort(eval_predictions,axis=0))  
    
    return ordered_eval_predictions
