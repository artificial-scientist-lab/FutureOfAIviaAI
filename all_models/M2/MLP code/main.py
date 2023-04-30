#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 12 12:11:37 2021

@author: ngoc

To reproduce the final submission results. 
"""

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from pathlib import Path
from genFeatures import *

def fitSK_MLP(X,y,MLP_options,args,get_predictions=True):
  """Fiting the sklearn version of MLP.  
  """
  results = {}
  model_name = get_outfile_name(MLP_options,MLPClassifier)
  prefix = str(args)
  print('fitting model ' +model_name +'on data gen by' + prefix)
  clf = MLPClassifier(**MLP_options)
  if get_predictions:
    #just fit once on the entire available set
    X_train_all = np.vstack((X['train'],X['val']))
    y_train_all = np.concatenate((y['train'],y['val']))
    print('fitting on all available data')
    clf.fit(X_train_all,y_train_all)
    print('score on train set:' + str(roc_auc_score(y_train_all, clf.predict_proba(X_train_all)[:,1])))
    #stash the classifier
    pickle.dump(clf, open('../model_outputs/MLP_classifier.pkl','wb'))
    return clf.predict_proba(X['test'])[:,1]
  else:
    print('fitting on train')
    clf.fit(X['train'],y['train'])
    print('score on train:' + str(roc_auc_score(y['train'], clf.predict_proba(X['train'])[:,1])))
    print('score on val:' + str(roc_auc_score(y['val'], clf.predict_proba(X['val'])[:,1])))


def get_train_val_test_data(args):
  """wrapper function. 
  Get data in the train/val/test format. 
  """
  #if args['rerun']:
  print('getting data for' + str(args))
  G = get_full_IGraph(args['yr'])
  args['G'] = G    
  #gen node features at time t0 for train
  args_tmp = get_kwargs_for(args,genNodeFeatures)
  node_features_t0 = genNodeFeatures(**args_tmp)
  #gen node features at time 1 for test
  args_tmp['t0'] = 1
  node_features_1 = genNodeFeatures(**args_tmp)
  #gen edges for training 
  args_tmp = get_kwargs_for(args,genEdgesForTraining)
  args_tmp['G'] = genFollowerGraph(args['yr'])
  train_edges = genEdgesForTraining(**args_tmp)
  
  #make train/test data
  args_tmp = get_kwargs_for(args,genEdgeFeatures)
  args_tmp['train_edges'] = train_edges
  args_tmp['node_features_t0'] = node_features_t0
  args_tmp['node_features_1'] =node_features_1
  data = genEdgeFeatures(**args_tmp)
  
  X = data['X_std']
  y = data['y']
  X_test = data['X_test_std']
  y_test = data['y_test']
  #split (X,y) to create training and validation set, keep edge identities in sss to double-check reproducibility
  sss = StratifiedShuffleSplit(n_splits=1,random_state=100,train_size=0.8,test_size=0.2)
  for train_index, test_index in sss.split(X, y):
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]
            
  #make dict
  X_dict = {'train': X_train, 'val': X_val, 'test': X_test}
  y_dict = {'train': y_train, 'val': y_val, 'test': y_test}  
  
  return (X_dict,y_dict,(train_index,test_index))





if __name__ == '__main__':
  #make sure that datafiles and folder names are as expected
  for folder in ['../data/processed','../model_outputs']:
    p = Path(folder)
    if not p.exists():
      p.mkdir(parents=True)
  
  p = Path('../data/raw')
  if not p.exists():
    p.mkdir(parents=True)
    raise IOError('Raw, unzipped data files provided by the competition should be stored in ../data/raw.')

  yr ='2017'
  args = {'t0': 0.9,'tdiff_forward': 0.1,'tdiff_backward': 0.15,'fakeRatio': 1,'scaler': StandardScaler,'yr':yr,'downsample': 0.75,'seed': 506, 'uppirate': 0.07,'hoprec_tmin':0.5,'hoprec_weight':'raw'}
  MLP_options = {'hidden_layer_sizes': (13,13,13,13,13), 'activation': 'relu', 'max_iter': 100, 'random_state': 42, 'early_stopping': True, 'verbose': False}
  X_dict,y_dict,sss = get_train_val_test_data(args)  
  y_test_hat = fitSK_MLP(X_dict,y_dict,MLP_options,args,get_predictions=True)
  convert_to_submission(y_test_hat)
  
