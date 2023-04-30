#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:44:30 2021

@author: ngoc

Codes to take raw data files and generate processed data files.

Data folder: 
(root folder)/data/raw
(root folder)/data/processed
"""

import os

import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

from features import *
from features import compute_edge_features
from utils_common import *


def genNodeFeatures(G,t0=1,tdiff_backward=0.15,yr='2014'):
  """Generate node features for all nodes in the graph G up to time t0 and pickle.  
  """
  params = locals()
  out_file = get_outfile_name(params,genNodeFeatures)  
  if os.path.exists(out_file):
    print('loading node features from ' + out_file)
    return pickle.load(open(out_file,'rb'))
  #file unavailable: regenerate. 
  if t0 < 1: 
    H_t0,V = get_subgraph(G,tmax=t0,tmin=0)      
  else: 
    H_t0 = G
    V = G.vs
 
  #compute node features
  node_features = compute_node_features(H_t0,t0,tdiff_backward)
  #add in/out degrees and their differentials as features
  G_dir = genFollowerGraph(yr=yr)
  H_t0_dir = G_dir.subgraph_edges(G_dir.es.select(lambda e: min(e['times']) <=t0), delete_vertices=False)    
  H_t0_dir_small = H_t0_dir.subgraph(V.indices)
  node_features_inout = compute_node_inout_degree(H_t0_dir_small,t0,tdiff_backward)
  node_features = np.vstack((node_features,node_features_inout))
  
  node_features_arr = np.array(node_features).T
  print('node features shape: ' + str(node_features_arr.shape))
  #replace nan from cc = nan by -1. ('unknown')
  node_features_arr = np.nan_to_num(node_features_arr,copy=False,nan=-1,posinf = 50)
  print('done computing node features for t0 = ' + str(t0) + ', stored at ' + out_file)
  pickle.dump(node_features_arr, open(out_file,'wb'))
  return node_features_arr
  

def genEdgesForTraining(G,t0=0.7,fakeRatio=1,downsample = 1,uppirate = 0.2, yr='2014',seed=506):
  """Generate edges for training. 
    train_pos = all edges from t0 to 1. 
    train_neg = randomly sampled negative edges that remain negative by time 1.   
    
    Only look at node pairs on vertices of G_{t0}. 
    In this case, node indices follow the indices of G_{t0} (ie: from 0 to len(G_t0.vs)                                                 
    downsample = only do a fraction of of the available edges. 
    This seems to give better models
    uppirate = over-sample fraction of edges from the type (old -- new). 
    where new = nodes born in the last 0.1 time, and old = nodes born before that. 
  """
  params = locals()
  out_file = get_outfile_name(params,genEdgesForTraining)  
  if os.path.exists(out_file):
    print('loaded edges for training from ' + str(out_file))    
    return pickle.load(open(out_file,'rb'))

  print('computing edges for training')
  H_t0,V = get_subgraph(G,tmax=t0,tmin=0)
  G_t0 = G.subgraph(V) #full graph on V
  #train_pos:  new edges appeared after time t0
  new_edges_seq = G_t0.es.select(lambda e: min(e['times']) > t0)
  new_edges = [e.tuple for e in new_edges_seq]
  H_edges = H_t0.get_edgelist()
  if downsample > 1: 
    downsample = 1
  
  rng = default_rng(seed)
  new_edges = rng.choice(new_edges, int(len(new_edges)*downsample),replace=False)
  #turn new_edges into list
  new_edges = list(zip(new_edges[:,0],new_edges[:,1]))
  fake_edges = sampleNegativeEdges(len(V),set(H_edges+new_edges),int(len(new_edges)*fakeRatio*downsample))
  if uppirate > 0:
    rng = default_rng(seed)
    V_newborn = H_t0.vs.select(times_gt = t0-0.1).indices
    V_old = H_t0.vs.select(times_le = t0-0.1).indices
    #add a fixed number of positive and negative pirate edge samples. 
    pirate_pos = [e.tuple for e in new_edges_seq.select(_between = (V_newborn,V_old))]
    #random subsample an uppirate fraction from here
    pirate_pos = rng.choice(pirate_pos, int(len(pirate_pos)*uppirate))
    pirate_pos = list(zip(pirate_pos[:,0],pirate_pos[:,1]))
    #do an equal sample of fake pirates
    pirate_edges = np.array(list(zip(rng.choice(V_newborn,int(len(pirate_pos)*1.1)), rng.choice(V_old,int(len(pirate_pos)*1.1)))))
    #order
    pirate_edges = np.sort(pirate_edges)
    pirate_edges = set(list(zip(pirate_edges[:,0],pirate_edges[:,1])))
    #only look at unseen edges
    pirate_edges = pirate_edges.difference(set(H_edges))
    #separate to positive and negative
    new_edges_set = set(new_edges)
    #pirate_pos = pirate_edges.intersection(new_edges_set)
    pirate_neg = list(pirate_edges.difference(new_edges_set))
    pirate_neg = rng.choice(pirate_neg,int(len(pirate_pos)))
    pirate_neg = list(zip(pirate_neg[:,0],pirate_neg[:,1]))
    print('added ' + str(len(pirate_pos)) + ' positive pirates and ' + str(len(pirate_neg)) + ' negative pirates')
    #add 
    new_edges = new_edges + list(pirate_pos)
    fake_edges = fake_edges + list(pirate_neg)
    
  output = {'train_pos': new_edges, 'train_neg': fake_edges}
  pickle.dump(output, open(out_file,'wb'))
  print('computed edges for training, stored in ' + str(out_file))
  return output


def get_extra_edge_feats_for_train(train_edges, args, feat_type = 'hoprec'):
  """ Function to append features to the computed train set. (dict output from genEdgesForTraining)
  Write it like this so we can append hoprec, sage etc, without having to recompute everything. 
  These features by design are NOT normalized. 
  """
  X = {}
  if feat_type == 'hoprec':
    hoprec_tmin = args['hoprec_tmin']
    H_t0 = args['H_t0']
    t0 = args['t0']
    hoprec_arr = args['hoprec_arr']
    for key in ['train_pos','train_neg']:
      if hoprec_tmin == 0:
        edge_features_cosine = cosine_of_edges(hoprec_arr,train_edges[key])
      else: #impute by default
        impute_idx = H_t0.vs.select(times_ge=t0*0.9).indices
        edge_features_cosine = cosine_of_edges(hoprec_arr,train_edges[key],impute_idx=impute_idx)
      X[key] = np.reshape(edge_features_cosine,(len(edge_features_cosine),1))
    
  #return the stacked version (n x 1, train_pos stacked over train_neg)
  out = np.vstack((X['train_pos'],X['train_neg']))    
  return out
   
def genEdgeFeatures(G,t0,node_features_t0,node_features_1,train_edges,scaler = StandardScaler,yr='2014',hoprec_tmin=0.5,hoprec_weight='raw',seed=506):
  """ Create: 
    G_{<= t0}
    train = all edges from t0 to 1. 
    val_neg = randomly sampled negative edges that remain negative by time 1. 
    
    If seen: only use edges in G_{<= t0}
    
    Do edge features for the test set as well. 
    
    node_features_t0 = array of node features up to time t0.  
    node_features_1 = array of node features up to time 1.  
    train_edges = output of genEdgesForTraining
    """
  params = {'t0': t0, 'yr': yr, 'hoprec_tmin': hoprec_tmin, 'hoprec_weight': hoprec_weight, 'seed': seed}
  out_file = get_outfile_name(params,genEdgeFeatures)  
  print('looking for edge feats from file' + out_file)
  train_file = out_file+'.train'
  if os.path.exists(out_file):
    print('loading edge features for train and test')
    return pickle.load(open(out_file, 'rb'))
  
  if os.path.exists(train_file):
    print('loading edge features for train from ' + train_file)
    out = pickle.load(open(train_file,'rb'))
  else:
    print('generating edge features for training')
    H_t0,V = get_subgraph(G,tmax=t0,tmin=0)     
    #compute edge features for new and fake edges. 
    edge_features = compute_edge_features(train_edges['train_pos'],node_features_t0,H_t0)
    fake_edge_features = compute_edge_features(train_edges['train_neg'], node_features_t0,H_t0)  
    
    #concat and generate labels
    X = np.vstack((edge_features,fake_edge_features))
    print('X shape:' + str(X.shape)) 
    y = [1]*len(edge_features) + [0]*len(fake_edge_features)
    y = np.array(y)
    if scaler is not None:
      scaler = scaler()
      X_std = scaler.fit_transform(X)
    else:
      X_std = X

    #append features: hoprec
    hoprec_arr = get_hoprec_array(t0=t0,yr=yr,tmin=hoprec_tmin,weight=hoprec_weight)
    hoprec_args = {'t0': t0, 'hoprec_tmin': hoprec_tmin, 'H_t0': H_t0, 'hoprec_arr': hoprec_arr}
    hoprec_col = get_extra_edge_feats_for_train(train_edges, feat_type = 'hoprec', args= hoprec_args)
    X_std = np.hstack((X_std, hoprec_col))
    
    print('done computing edge features for t0 = ' + str(t0))
    out = {'X_std': X_std, 'y': y, 'scaler_for_X': scaler}
    pickle.dump(out,open(out_file+'.train','wb'))
  
  #deal with test
  print('generating edge features for test set')
  #generate features on everybody. Need to replace blinds by the average of newborns (in time 0.9 to 1.0)
  if yr == '2014':
    prefix = 'TrainSet'
  else:
    prefix = 'CompetitionSet'
  good_file = '../data/raw/'+prefix+yr+'_unseen_good.pkl'
  pirate_file = '../data/raw/'+prefix+yr+'_unseen_pirate_unpacked.pkl'
  blind_file = '../data/raw/'+prefix+yr+'_unseen_blind.pkl'
  if (not os.path.exists(pirate_file)) or (not os.path.exists(good_file)) or (not os.path.exists(blind_file)):
    processTestFile(yr = yr)
  if yr != '2017':
    unseen_edges_valid,true_labels_valid=pickle.load(open(good_file,'rb'))
  else:
    unseen_edges_valid = pickle.load(open(good_file,'rb'))

  edge_feats = {}  
  #compute node features for nodes in G_1, NO DICE  YET
  edge_feats['good'] = compute_edge_features(unseen_edges_valid,node_features_1)
  
  #deal with pirates    
  pirate_unpacked = pickle.load(open(pirate_file,'rb'))
  V_newborn = (G.vs.select(times_ge=0.9)).indices
  #add last node in node_features_1 as the dummy node 
  n = node_features_1.shape[0]
  node_features_1 = np.vstack((node_features_1,np.mean(node_features_1[V_newborn,:],0)))
  
  pirate_edges_itt = zip(pirate_unpacked,[n]*len(pirate_unpacked))
  pirate_edges = [e for e in pirate_edges_itt]
  edge_feats['pirate'] = compute_edge_features(pirate_edges,node_features_1)

  #deal with blinds
  obj = pickle.load(open(blind_file,'rb'))
  if yr != '2017':
    blind_labels,blind_count=obj
  else:
    blind_count = obj
  
  blind_edges = [(n,n)]
  one_blind_feats = compute_edge_features(blind_edges,node_features_1)   
  edge_feats['blind']= np.vstack([one_blind_feats]*blind_count)
  
  #add in DICE for all features computed so far.
  dice = {}  
  print('computing dice good')
  dice_seen_edges = np.array(G.similarity_dice(pairs = unseen_edges_valid))
  dice['good'] = dice_seen_edges.reshape((len(dice_seen_edges),1))
  
  #for pirates: compute dice via its interpretation. See written pdf. 
  print('computing dice pirates')
  adj_csr = G.get_adjacency_sparse()
  #get: \sum_w |N_w \cap V|
  sumw_cap_V = adj_csr[V_newborn,:].sum()
  pirate_ctr = Counter(pirate_unpacked)
  pirate_idx = list(pirate_ctr.keys())
  pirate_neighbor_size = np.array(G.neighborhood_size(pirate_idx))
  #np.array of bottoms for each pirate_idx
  bottom = len(V_newborn)*pirate_neighbor_size + sumw_cap_V
  #np.array of tops
  adj_csr_sums = adj_csr[V_newborn,:].sum(axis=0)
  top = np.array([2*adj_csr_sums[0,G.neighbors(u)].sum() for u in pirate_idx])
  dice_pirate_idx = dict(zip(pirate_idx,top/bottom))
  dice_pirates = np.array([dice_pirate_idx[u] for u in pirate_unpacked])
  dice['pirate'] = dice_pirates.reshape((len(dice_pirates),1))
  
  #compute: dice_newborn. See written pdf. 
  print('computing dice newborn')
  select = adj_csr_sums >= 2
  top = np.sum(np.multiply(adj_csr_sums[select],adj_csr_sums[select]-1)*2)
  bottom = adj_csr_sums.sum()*2*(len(V_newborn)-1)
  dice_newborn_average = top/bottom
  dice_blind = np.array([dice_newborn_average]*blind_count)
  dice['blind'] = dice_blind.reshape((len(dice_blind),1))
  
  #tack on cosine at time t = 1
  hoprec_arr = get_hoprec_array(t0=1,yr=yr,tmin=hoprec_tmin,weight=hoprec_weight)
  #cosine of blind: average over all nodes in V_newborn
  V_newborn = G.vs.select(times_gt = 0.9).indices
  if hoprec_tmin == 0:
    edge_features_cosine = cosine_of_edges(hoprec_arr,unseen_edges_valid)
  else: #impute by default
    edge_features_cosine = cosine_of_edges(hoprec_arr,unseen_edges_valid,impute_idx=V_newborn)
 
  cosine_good = np.reshape(edge_features_cosine,(len(edge_features_cosine),1))
  #cosine of blind: average over all nodes in V_newborn
  cosine_blind = cosine_of_blind(hoprec_arr[V_newborn,:])
  print('cosine_blind over newborn =' + str(cosine_blind))
  print('cosine_blind over all = ' + str(cosine_of_blind(hoprec_arr)))
  #for pirates: average cosine over all nodes in V_newborn. 
  cosine_pirate = cosine_of_pirate(hoprec_arr,pirate_idx,V_newborn)
  cosine_pirate_idx = dict(zip(pirate_idx,cosine_pirate))
  cosine_pirates = np.array([cosine_pirate_idx[u] for u in pirate_unpacked])
  #reassemble
  cosine = {}
  cosine['good'] = cosine_good
  cosine['pirate'] = cosine_pirates.reshape((len(cosine_pirates),1))
  cosine['blind'] = np.reshape(np.array([cosine_blind]*blind_count),(blind_count,1))

  #append what we have so far
  scaler = out['scaler_for_X']
  for key in ['good','pirate','blind']:
    edge_feats[key] = np.hstack((edge_feats[key], dice[key],cosine[key]))
    if scaler is not None:
        d = edge_feats[key].shape[1]
        edge_feats[key][:,0:(d-1)] = scaler.transform(edge_feats[key][:,0:(d-1)])
 

  #turn into a big X_test with the correct order
  good_idx,pirate_unpacked_idx,blind_idx = pickle.load(open('../data/raw/' +prefix + yr +'_unseen_idx_for_submission','rb'))
  total_unseen = len(good_idx) + len(pirate_unpacked_idx) + len(blind_idx)
  d = edge_feats['good'].shape[1]
  X_test = np.empty((total_unseen,d))
  X_test[good_idx,] = edge_feats['good']
  X_test[pirate_unpacked_idx,] = edge_feats['pirate']
  X_test[blind_idx,] = edge_feats['blind']
  print('cosine range : ' + str(min(X_test[:,-1])) + ',' + str(max(X_test[:,-1])))
  
  out['X_test_std'] = X_test
  if yr != '2017':
    out['y_test'] = getAnswer(yr)
  else:
    out['y_test'] = None
   
  print('edge features and y-labels for train and test generated,')
  pickle.dump(out, open(out_file,'wb'))  
  return out

