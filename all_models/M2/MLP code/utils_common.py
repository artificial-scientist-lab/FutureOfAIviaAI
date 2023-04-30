#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:39:21 2021

@author: ngoc

Common utilities functions for ALL methods (sage, MLP etc)

"""

import numpy as np
import pickle
from numpy.random import default_rng
import igraph
import os
from collections import Counter
from inspect import signature
from pathlib import Path
import json
import git
from sklearn.preprocessing import normalize
from tqdm import tqdm

def get_hoprec_array(t0, yr:str, weight='raw',tmin=0,hoprec_folder = '../data/HOPREC'):
  """ Load the hoprec array from raw, and rename. 
  Requires that we can get embedding for ALL nodes inside G_t0, otherwise raise an assert error
  """
  p = Path(hoprec_folder)
  #find folder file
  p_sub = list(p.glob('*'+yr+'*'+weight+'*'))
  print(p_sub)
  assert len(p_sub) == 1
  p_sub = p_sub[0]
  #find file: format: year-time-type-dimension
  pattern = '*'+str(tmin)+'_'+str(t0)+'_*128*'
  print('looking for ' + pattern)
  p_file = list(p_sub.glob(pattern))
  print(p_file)
  assert len(p_file) == 1
  hoprec_file = str(p_file[0])
  hoprec = pickle.load(open(hoprec_file,'rb'))
  #if hoprec is not an array: turn it into one
  if type(hoprec) is dict:
    #first, reindex
    G = get_full_IGraph(yr)
    if t0 < 1:
      G,V = get_subgraph(G,tmax=t0,tmin=0,truncate=False)
    names = [str(int(u)) for u in G.vs['original_name']]
    #first, change the keys of hoprec to int's. 
    name_to_G_idx = dict(zip(names, range(len(names))))      
    hoprec = renameDictionaryKeys(hoprec,name_to_G_idx)      
    hoprec_arr = np.zeros((len(G.vs),128))
    seen_idx = list(hoprec.keys())
    print('fraction of nodes seen in hoprec: ' + str(len(seen_idx)/len(G.vs)))
    for i in seen_idx:
      hoprec_arr[i] = np.array(hoprec[i])      
    #note: default hoprec for seen nodes but got deleted will be 0. 
    print('rewrote hoprec from dict to array format in' + hoprec_file)
    #normalize to unit norm for each sample
    hoprec_arr = normalize(hoprec_arr)
    pickle.dump(hoprec_arr,open(hoprec_file,'wb'))
  else:
    hoprec_arr = hoprec
  return hoprec_arr
    
def cosine_of_edges(hoprec_arr, edge_list,impute_idx = None):
  """get cosine similarity = pairwise inner product
  between guys in edge_list. 
  New version: impute over the average of the impute_idx. 
  The unseen_idx are the hoprec_arr which is all-zero
  """
  if impute_idx is None:
    return np.array([np.dot(hoprec_arr[i],hoprec_arr[j]) for i,j in edge_list])
  else:
    ans = []
    #impute over blinds
    blind_val = cosine_of_blind(hoprec_arr[impute_idx,:])
    print('blind val imputed with shortcut = ' + str(blind_val))
    for i,j in tqdm(edge_list):
      blind_i = np.all(hoprec_arr[i] == 0)
      blind_j = np.all(hoprec_arr[j] == 0)
      if blind_i + blind_j == 0:
        ans += [np.dot(hoprec_arr[i],hoprec_arr[j])]
      else:
        if blind_i + blind_j == 2: #both blind
          ans += [blind_val]
        else: #one eye is blind
          if blind_i == 1:
            ans += [np.mean(np.dot(hoprec_arr[j],hoprec_arr[impute_idx,:].T))]
          else:
            ans += [np.mean(np.dot(hoprec_arr[i],hoprec_arr[impute_idx,:].T))]
    return np.array(ans)
  
def cosine_of_blind(hoprec_arr,nrows=None):
  """get average cosine similarity of all pairs in hoprec_arr.
  If nrows is not None, take the sum and divide by nrows (in case we want other sort of normalization)
  Otherwise, divide by len(hoprec_arr)
  """
  hop_sq = hoprec_arr**2
  d = hoprec_arr.shape[1]
  if nrows is None:
    nrows = hoprec_arr.shape[0]
  #compute: \sum_u\sum_{v\neq u} u_iv_i for each column i
  cross_term = (np.sum(hoprec_arr,0))**2 - np.sum(hop_sq,0)
  n = nrows*(nrows-1)
  return np.sum(cross_term)/n

def cosine_of_pirate(hoprec_arr,pirate_idx,newborn_idx):
  """For each pirate u, get its average cosine similarity against all nodes in the subset newborn_idx"""
  P = hoprec_arr[pirate_idx,:]
  V = hoprec_arr[newborn_idx,:]
  return np.mean(np.dot(P,V.T),1)

def fast_index_replace(ls: np.ndarray, val_old: np.ndarray, val_new: np.ndarray, random_draw_from = None,seed=42):
  """For a list ls (eg, edge_index)
  replace val_new[i] by val_old[i]. 
  if random_draw_from is None:
    Leave things that are NOT in val_old alone (returns default)
  else: 
    randomly replace values which are NOT in val_old by values from random_draw_from. 
    Assume that ls, val_old, val_new are integers and > 0
  """
  d = dict(zip(val_old, val_new))
  ls2 = np.array([d.get(e, e) for e in ls])
  if random_draw_from is not None:
    rng = default_rng(seed) #fix seed for reproducibility
    unchanged = np.where(ls2 == ls)[0]
    #unchanged could be due to: same vals, or d[i] = i       
    to_replace = np.array([x not in d for x in ls[unchanged]])
    n_replacement = np.sum(to_replace)                       
    ls2[unchanged[to_replace]] = rng.choice(random_draw_from,n_replacement)
  return ls2

def get_kwargs_for(params, f):
  """Truncate the dictionary of params to make it eligible to pass to function. Returns a dict (arg,val)"""
  sig = signature(f)
  keys = list(sig.parameters)
  params_valid = dict([(k,params[k]) for k in keys if k in params])
  return params_valid
    

def get_outfile_name(params, f, params_omit = ['G','rerun'],params_obj = ['scaler'],subdir = '../data/processed'):
  """Returns the name of the outfile to pickle.dump
  the output of function f. 
  params: dictionary. Will extract the parameters relevant to f, and by default, their parameters and their values appear in the outfile name.
  params_omit: list of strings. Parameters to be ommitted from the outfile name.
  params_obj: parameters which are objects. When convert to string, keep the name of the obj only
  subdir: storage dir
  """
  params_for_f = get_kwargs_for(params,f)
  for k in params_omit:
    if k in params_for_f:
      params_for_f.pop(k)  
  for k in params_obj:
    if k in params_for_f:
      val = params_for_f[k].__name__
      params_for_f[k] = val
  #sort the keys
  keys_sorted = sorted(list(params_for_f))
  out_file = f.__name__ +'_'+'_'.join([str(key) +'_'+ str(params_for_f[key]) for key in keys_sorted])
  return os.path.join(subdir,out_file)


def add_to_dict(dt, key, val):
    if key in dt:
        dt[key] += [val]
    else:
        dt[key] = [val]
    return dt

def convert_time_to_fraction(data_source='../data/raw/Train2014_3',rerun =False):
  """
  :param full_graph: the graph in consideration, numpy array dim(n,3) [vertex 1, vertex 2, time stamp]
  
  Returns: np array dim(n,3) [vertex 1, vertex 2, time_fraction]
  where time_fraction is a real number in (0,1).
  
  Run once and pickle. Future runs only run if file has not been created 
  """
  new_file = data_source + '_fractional_time.pkl'
  if not os.path.exists(data_source + '.pkl'):
    raise ValueError('data source file ' + data_source + '.pkl' + ' not found.')  
  #only run if file has not been created
  if not os.path.exists(new_file) or rerun: 
    print('creating fractional time dataset at ' + new_file)
    full_dynamic_graph_sparse,unconnected_vertex_pairs,year_start,years_delta = pickle.load( open( data_source + '.pkl', "rb" ) )
    times = np.unique(full_dynamic_graph_sparse[:,2])
    #new_times = np.zeros(len(full_dynamic_graph_sparse))
    ctr = Counter(full_dynamic_graph_sparse[:,2])
    new_times = []
    for i in tqdm(range(len(times))):
      new_times += [i]*ctr[times[i]]
    new_times = np.array(new_times)/len(times)
    real_graph = full_dynamic_graph_sparse*1.0
    real_graph[:,2] = new_times
    pickle.dump((real_graph,unconnected_vertex_pairs), open( new_file, 'wb'))  


def genIGraph(edgeList, out_file ='../data/processed/igraph_2014.pkl',rerun=False):    
  if not os.path.exists(out_file) or rerun: 
    nameDict = {}
    #make sure that the order is respected
    nodeList = set(list(edgeList[:,0]) + list(edgeList[:,1]))
    nodeList = np.sort(list(nodeList))
    n_vertex = len(nodeList)
    nameDict = dict([(nodeList[i],i) for i in range(n_vertex)])  
    edgeList = renameNodes(edgeList, nameDict)
    edge_time_dt = getTimeCounter(edgeList)
    #make an igraph and compute stuff
    G = igraph.Graph()
    G.add_vertices(n_vertex)
    G.add_edges(list(edge_time_dt.keys()))
    #add edge info: all the times that this edge appeared
    G.es['times'] = list(edge_time_dt.values())
    #add vertex info: the original name of this vertex
    G.vs['original_name'] = nodeList
    #register the first times that the nodes appear
    G.vs['times'] = [min([min(G.es[e]['times']) for e in G.incident(v)]) for v in G.vs.indices]    
    pickle.dump(G, open(out_file,'wb'))
    return G
  else:
    return pickle.load(open(out_file,'rb'))

def get_subgraph(G: igraph.Graph,tmax,tmin=0,truncate=True):
  """Get subgraph of G for edges in time [tmin,tmax)
  If truncate = True, truncate the edge timestamps to be in this time range as well."""
  H_es = G.es.select(lambda e:  (min(e['times']) < tmax) and (min(e['times']) >= tmin))
  H = G.subgraph_edges(H_es,delete_vertices=False)
  if truncate:
    truncated_times = [[t for t in tseq if t < tmax and t >= tmin] for tseq in H_es['times']]
    H.es['times'] = truncated_times
  V =H.vs.select(_degree_gt = 0)
  return (H.subgraph(V),V)
  
    
def getData(data_source = '../data/raw/',yr='2014',suffix='_3_fractional_time.pkl'):
  if yr == '2014':
    data_source = data_source + 'TrainSet'
  else:
    data_source = data_source + 'CompetitionSet'
  full_path = data_source+yr+suffix
  if not os.path.exists(full_path): 
    original_file = data_source+yr+'_3'  
    #try to generate the time fraction file
    print('converting time to fraction for' + original_file)
    convert_time_to_fraction(original_file)    
  return pickle.load(open(full_path, 'rb'))
  
def getAnswer(yr = '2014'):  
  data_source = '../data/raw/TrainSet' + yr + '_3_solution.pkl'
  try: 
    print('loaded answers for yr = ' + yr)
    return pickle.load(open(data_source, 'rb'))
  except: 
    raise ValueError('file' + data_source + ' not found')

def get_full_IGraph(data_source = '../data/processed/igraph_2014.pkl',rerun=False):
  """Can specify data source, or a 4-digit year string"""
  if len(data_source) == 4:
    yr = data_source
    data_source = '../data/processed/igraph_' + yr + '.pkl'
  if (not os.path.exists(data_source)) or rerun: 
    all_edges, unseen_edges = getData(yr=yr)
    G = genIGraph(all_edges, out_file =data_source,rerun=rerun)
    print('(re)-generated full igraph, stored at ' + data_source)
    return G
  else: 
    return pickle.load(open(data_source,'rb'))
        
 
def sampleNegativeEdges(nodeList, edgeSet, target_neg_edges: int):
    rng = default_rng(506)
    edges = rng.choice(nodeList, target_neg_edges * 2, replace=True)
    edges = np.reshape(edges, (target_neg_edges, 2))
    edges = np.sort(edges)
    edges = edges[edges[:,0] != edges[:,1]]
    fakeEdges_set = set([tuple(x) for x in edges])
    fakeEdges_set.difference_update(edgeSet)
    return list(fakeEdges_set)

def convert_to_submission(y_hat):
  repo = git.Repo(search_parent_directories=True)
  sha = repo.head.object.hexsha  
  sorted_predictions_eval=np.flip(np.argsort(y_hat,axis=0))
  submit_file=str(sha) + ".json"
  all_idx_list_float=list(map(float, sorted_predictions_eval))
  with open(submit_file, "w", encoding="utf8") as json_file:
    json.dump(all_idx_list_float, json_file)
  print("Solution stored as "+submit_file)      


    

def renameNodes(edgeList, old_name_to_new_name: dict):
    edgeList[:, 0] = [old_name_to_new_name[x] for x in edgeList[:, 0]]
    edgeList[:, 1] = [old_name_to_new_name[x] for x in edgeList[:, 1]]
    return edgeList  
  
def renameDictionaryKeys(dt,nameDict):
  """Rename all keys in the dictionary using nameDict
  nameDict = dict of (old key, new name)
  """
  return dict([(nameDict[u], dt[u]) for u in dt.keys()])
  
def renameDictionaryValues(dt,nameDict):
  """Rename all values in the dictionary using nameDict
  nameDict = dict of (old key, new name)
  Assume the dt[key] is a list of values
  """
  new_dt = {}
  for u in dt.keys():
    vals = dt[u]
    vals_new = [nameDict[v] for v in vals]
    new_dt[u] = vals_new
  return new_dt  
  
def getTimeCounter(ini_graph):
  edgeTimes = dict()
  for e in ini_graph:
    u,v,t = e
    u = int(u)
    v = int(v)
    add_to_dict(edgeTimes, (u,v),t)
  return edgeTimes  

def genFollowerGraph(yr='2014'):
  """
  Create a directed graph, 
  where  u-> v if u ~ v and u is born BEFORE v
  (say that 'u follows v').
  If born at the same time, then it is BIDIRECTIONAL
  """
  out_file = '../data/processed/igraph' + yr + '_directed.pkl'
  if not os.path.exists(out_file):
    print('re-computed follower graph for yr '+ yr)
    G = get_full_IGraph(yr)
    edgeList = G.get_edgelist()
    n = len(edgeList)
    edgeList = np.array(edgeList)
    node_times = np.ravel(edgeList)
    node_times = np.array([G.vs[u]['times'] for u in node_times])
    node_times = node_times.reshape((n,2))
    different_papers = node_times[:,0] != node_times[:,1]
    same_papers = node_times[:,0] == node_times[:,1]
    #add bidirections
    edge_bidir = np.vstack((edgeList[same_papers], np.flip(edgeList[same_papers],1)))
    edge_subset = edgeList[different_papers]
    node_times_subset = node_times[different_papers]
    #sort: order should be (larger time) <- (smaller time)
    rev_edge = node_times_subset[:,0] > node_times_subset[:,1]
    younger_node = edge_subset[rev_edge][:,0]
    older_node = edge_subset[rev_edge][:,1]
    younger_node = younger_node.reshape((len(younger_node),1))
    older_node = older_node.reshape((len(older_node),1))
    edge_subset[rev_edge] = np.hstack((older_node,younger_node))
    edge_subset = np.vstack((edge_subset,edge_bidir))
    #create a directed graph
    H = igraph.Graph(n=len(G.vs),edges = list(edge_subset),directed=True)
    H.vs['times'] = G.vs['times']
    #add edge time. 
    edgeTimes = G.es['times']
    idx = np.where(different_papers)[0]
    edgeTimes_subset = [edgeTimes[i] for i in idx]
    idx = np.where(same_papers)[0]
    edgeTimes_same_paper = [edgeTimes[i] for i in idx]
    #concatenate in order
    edgeTimes_all = edgeTimes_subset + edgeTimes_same_paper + edgeTimes_same_paper
    H.es['times'] = edgeTimes_all
    #copy over node features
    H.vs['times'] = G.vs['times']
    H.vs['original_name'] = G.vs['original_name']
    pickle.dump(H, open(out_file,'wb'))
  else:
    H = pickle.load(open(out_file,'rb'))
    print('loading follower graph for yr = ' + yr)
  return H


def processTestFile(yr='2014'):  
  """Note: output file node indices are w.r.t. graph G_1. """
  if yr == '2014':
    prefix = 'TrainSet'+yr
  else:
    prefix = 'CompetitionSet'+yr
  dt = getData(yr=yr)
  all_edges,unseen = dt
  G = genIGraph(all_edges,out_file='../data/processed/igraph_' + yr + '.pkl')
  if yr != '2017':
    true_labels = getAnswer(yr=yr)
    good_labels = []
    pirate_labels = {} #dictionary (u,number_of_actual_edges_of_u in test set). 
    blind_labels = 0

  #split unseen into: blind, pirate, good
  original_name = G.vs['original_name']
  nameDict = dict([(int(original_name[i]),i) for i in range(len(original_name))])
  blind = 0 #counter
  pirate = {} #dictionary of (known_node, number of unseen nodes it become friends with)
  pirate_unpacked = [] #list of pirates (just the good eye) in the order that they are found in the test file  
  pirate_unpacked_labels = []
  good = [] #list of good edges (both nodes seen)
  
  #keep track of indices to build submission
  pirate_unpacked_idx = [] 
  good_idx = [] 
  blind_idx = []
  
  for i in range(len(unseen)):
    u,v = unseen[i]
    u_in = u in nameDict
    v_in = v in nameDict
    if u_in + v_in == 2:
      good_idx += [i]
      good += [(nameDict[u],nameDict[v])]
      if yr != '2017':
        good_labels.append(true_labels[i])
    if u_in + v_in == 1:
      pirate_unpacked_idx += [i]
      if yr != '2017':
        pirate_unpacked_labels += [true_labels[i]]
      if u_in: 
        pirate_unpacked += [nameDict[u]]
        if nameDict[u] in pirate:
          pirate[nameDict[u]] += 1
          if yr != '2017':
            pirate_labels[nameDict[u]] += true_labels[i]
        else:
          pirate[nameDict[u]] = 1
          if yr != '2017':
            pirate_labels[nameDict[u]] = true_labels[i]          
      else: 
        pirate_unpacked += [nameDict[v]]
        if nameDict[v] in pirate:
          pirate[nameDict[v]] += 1
          if yr != '2017':
            pirate_labels[nameDict[v]] += true_labels[i]          
        else:
          pirate[nameDict[v]] = 1        
          if yr != '2017':
            pirate_labels[nameDict[v]] = true_labels[i]                    
    if u_in + v_in == 0:
      blind_idx += [i]
      blind += 1
      if yr != '2017':
        blind_labels += true_labels[i]

  print('fraction of edges in blind: ' + str(blind/len(unseen)))
  print('fraction of edges in good: ' + str(len(good)/len(unseen)))
  
  #ok, so: 66% in good, 3% in blind, 31% in pirate for competition set 

  #dump indices
  pickle.dump((good_idx,pirate_unpacked_idx,blind_idx),open('../data/raw/' + prefix +'_unseen_idx_for_submission','wb'))

  #dump pirates unpacked
  pickle.dump(pirate_unpacked, open('../data/raw/'+prefix+'_unseen_pirate_unpacked.pkl','wb'))

  if yr != '2017':
    pickle.dump((good,good_labels), open('../data/raw/' + prefix + '_unseen_good.pkl','wb'))
    pickle.dump((pirate,pirate_labels),open('../data/raw/' + prefix + '_unseen_pirate.pkl','wb'))
    print('fraction of edges in blind which are real edges: ' + str(blind_labels/blind))
    pickle.dump((blind_labels,blind), open('../data/raw/' +prefix+ '_unseen_blind.pkl','wb'))
    pickle.dump(pirate_unpacked_labels,open('../data/raw/'+prefix+'_unseen_pirate_unpacked_labels.pkl','wb'))
  else: 
    pickle.dump(good, open('../data/raw/' + prefix + '_unseen_good.pkl','wb'))
    pickle.dump(pirate,open('../data/raw/' + prefix + '_unseen_pirate.pkl','wb'))  
    pickle.dump(blind,open('../data/raw/' + prefix + '_unseen_blind.pkl','wb'))  

def reproducibility_check(f_new = '../425370062f34665ae86215aa9b3cdb156ba573f3.json', f_old = '../ee5ae68292bde292277c41920b08f0a824f62b77.json'):
  """Function to check similarity of two json submissions. Used to verify reproducibility of the code"""
  print("Running reproducibility check for files " + f_new + " and " + f_old)
  pred1 = json.load(open(f_new,'rb'))
  pred2 = json.load(open(f_old,'rb'))
  #agreement fraction
  agree = np.sum(np.array(pred1) == np.array(pred2))
  print("agreement fraction between runs: " + str(agree/len(pred1))) 
  #location of unmatched
  unmatched = np.where(np.array(pred1) != np.array(pred2))
  print('rankings of the unmatched' + str(unmatched))  
