# -*- coding: utf-8 -*-
"""
@author: Francisco Valente
"""

# imports
import numpy as np
import pickle
import json
from sklearn.linear_model import LogisticRegression

# load files
data_train = np.load('data_pca_training.npy')
label_train = np.load('label_training.npy')
data_eval = np.load('data_pca_evaluation.npy')

# load the random search of the best prediction model (random forest)
rf_search = pickle.load(open('rf_optimization', 'rb'))
model = rf_search.best_estimator_ # get optimized RF parameters

# apply prediction model to evaluation (2020) data
eval_predictions = model.predict_proba(data_eval)[:,1]
predictions_to_submit = np.flip(np.argsort(eval_predictions,axis=0))    

# save the results as json file
submit_file="predictions_eval.json"
all_idx_list_float=list(map(float, predictions_to_submit))
with open(submit_file, "w", encoding="utf8") as json_file:
    json.dump(all_idx_list_float, json_file)
    


#### Another option (perform an ensemble of the four classifiers)

# # load models
# rf_search = pickle.load(open('rf_optimization', 'rb'))
# rf_model = rf_search.best_estimator_
# lr_search = pickle.load(open('lr_optimization', 'rb'))
# lr_model = lr_search.best_estimator_
# knn_search = pickle.load(open('knn_optimization', 'rb'))
# knn_model = knn_search.best_estimator_
# mlp_search = pickle.load(open('mlp_optimization', 'rb'))
# mlp_model = mlp_search.best_estimator_

# # apply prediction models to evaluation (2020) data
# eval_predictions_rf = rf_model.predict_proba(data_eval)[:,1]
# eval_predictions_lr = lr_model.predict_proba(data_eval)[:,1]
# eval_predictions_knn = knn_model.predict_proba(data_eval)[:,1]
# eval_predictions_mlp = mlp_model.predict_proba(data_eval)[:,1]

# # random forest and k-nearest neighbors does not produce probabilities inherently meaningful
# # so another step (calibration) is required

# def calibrate_model(label_train, predictions_train, predictions_test):
#     clf = LogisticRegression(solver='liblinear')
#     calibrator = clf.fit(predictions_train.reshape(-1,1), label_train)
#     calibrated_predictions_test = calibrator.predict_proba(predictions_test.reshape(-1,1))[:,1]
#     return calibrated_predictions_test

# train_predictions_rf = rf_model.predict_proba(data_train)[:,1]
# train_predictions_knn = knn_model.predict_proba(data_train)[:,1]

# eval_predictions_rf_calibrated = calibrate_model(label_train, train_predictions_rf, eval_predictions_rf)
# eval_predictions_knn_calibrated = calibrate_model(label_train, train_predictions_knn, eval_predictions_knn)

# ### Get ensemble predictions

# # weight is given by the mean AUC of the 5-fold cross-validation in the parameters optimization
# train_weights = [rf_search.best_score_, lr_search.best_score_, knn_search.best_score_, mlp_search.best_score_] #
# individual_predictions = [eval_predictions_rf_calibrated, eval_predictions_lr, eval_predictions_knn_calibrated, eval_predictions_mlp]
# eval_predictions_ensemble = np.average(individual_predictions, axis=0, weights=train_weights)
