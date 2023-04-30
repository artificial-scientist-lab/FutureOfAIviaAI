# -*- coding: utf-8 -*-
"""
@author: Francisco Valente
"""

# imports
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pickle


# load files
data_train = np.load('data_pca_training.npy')
label_train = np.load('label_training.npy')
data_eval = np.load('data_pca_evaluation.npy')


### Optimize hyperparameters of predicion models 

# prediction model 1 (elastic-net logistic regression)

param_grid = {'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
              'C': [0.001, 0.01, 0.1, 1, 10, 100]}

clf = LogisticRegression(penalty='elasticnet', solver='saga')
 
grid_lr = GridSearchCV(clf, param_grid, verbose=3, n_jobs=-1, cv=5, scoring='roc_auc')
grid_lr.fit(data_train, label_train)

pickle.dump(grid_lr, open('lr_optimization', 'wb'))


# prediction model 2 (random forest) >> parameters should be re-adjusted in a second round

param_grid = {'bootstrap': [True, False],
              'max_depth': [3, 5, 10, 25, 50, 75, 100, None],
              'min_samples_leaf': [2,5, 10, 20, 50,100, 300],
              'min_samples_split': [5, 10, 15, 30, 20],
              'n_estimators': [50, 100, 200, 350, 500, 750, 1000, 1500]}

clf = RandomForestClassifier(n_jobs=-1)
 
grid_rf = RandomizedSearchCV(clf, param_grid, verbose=3, n_jobs=-1, cv=5, scoring='roc_auc',  n_iter=125)
grid_rf.fit(data_train, label_train)

pickle.dump(grid_rf, open('rf_optimization', 'wb'))


# prediction model 3 (k nearest neighbors)

clf = KNeighborsClassifier()

param_grid = {'n_neighbors': [5,10,25,50,75,100,125,150,175,250,350,500],
              'weights' : ['uniform', 'distance']}

grid_knn = GridSearchCV(clf, param_grid, verbose=3, n_jobs=-1, cv=5, scoring='roc_auc')
grid_knn.fit(data_train, label_train)

pickle.dump(grid_knn, open('knn_optimization', 'wb'))


# prediction model 4 (multi layer perceptron) >> parameters should be re-adjusted in a second round

clf = MLPClassifier()

param_grid = {'hidden_layer_sizes': [(25), (50), (100), (250), (500),
                           (50,50), (100,100), (250,250), (50,150), (100,50),
                           (300,300), (500,500), (25,25),
                           (50,50,50), (50,100,50), (100,100,100), (250,250,250),
                           (300,200,100)],
                'activation': ['tanh', 'relu', 'logistic'],
                'alpha': [0.0001, 0.005, 0.01, 1],
                'learning_rate': ['constant','adaptive'],
                'max_iter': [100,200,500,1000]}

grid_mlp = RandomizedSearchCV(clf, param_grid, verbose=3, n_jobs=-1, cv=5, scoring='roc_auc', n_iter=125)
grid_mlp.fit(data_train, label_train)

pickle.dump(grid_mlp, open('mlp_optimization', 'wb'))