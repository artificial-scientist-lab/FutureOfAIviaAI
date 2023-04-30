#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:20:22 2021

@author: ngoc

Functions specific to computing node and edge features
  
"""

from utils_common import *
 
def _compute_paper_statistics(H: igraph.Graph):
  """For each statistics, compute a list for each node and return as a list of lists.
  Stats computed:
  n_papers: number of unique time stamps in the list of edges incident to this node. 
  """
  n_papers = []
  for u in H.vs:
    edges = H.incident(u)
    unique_times = []
    for e in edges:
      unique_times += H.es[e]['times']
    unique_times = np.sort(np.unique(unique_times))
    n_paper = len(unique_times)
    n_papers.append(n_paper)
  return [n_papers]

def _compute_initial_node_time(H: igraph.Graph,t0):
  """Returns a list of: first time this node appeared
  """
  ini_time = []
  for u in H.vs:
    edges = H.incident(u)
    unique_times = []
    for e in edges:
      unique_times += H.es[e]['times']
    ini_time.append(min(unique_times)/t0)
  return ini_time

def _compute_node_features(H: igraph.Graph,plus_one=True):
  """ Compute the node features that so far have been useful"""
  funcs = [igraph.Graph.transitivity_local_undirected, igraph.Graph.degree, igraph.Graph.pagerank]
  #funcs = [igraph.Graph.transitivity_local_undirected, _normalized_degree, igraph.Graph.pagerank]
  func_names = ['cc','deg','page']
  kwargs = [{'mode':1},{}, {'directed': False}]
  for i in range(len(funcs)):
    func = funcs[i]
    func_name = func_names[i]
    H.vs[func_name] = func(H, **kwargs[i])
  feats = [H.vs[func_name] for func_name in func_names]
  feats += _compute_paper_statistics(H)
  feats =  np.array(feats) #shape: (4,V)
  #do log transforms of n_paper, degree and pagerank
  if plus_one:
    feats[1,:] = np.log(feats[1,:]+1) #degree
    feats[2,:] = np.log(feats[2,:]) #pagerank
    feats[3,:] = np.log(feats[3,:]+1) #n_paper
  else: 
    feats[1,:] = np.log(feats[1,:])
    feats[2,:] = np.log(feats[2,:])
    feats[3,:] = np.log(feats[3,:])
  return feats


def take_derivatives(feats_list):
  """Return: 0th, first and second derivatives"""
  feats_diff = []
  feats_diff += [feats_list[0]]
  feats_diff += [feats_list[0] - feats_list[1]]
  feats_diff += [feats_list[0] + feats_list[2] - 2*feats_list[1]]
  #return np.vstack(feats_diff)
  return feats_diff

def compute_node_inout_degree(H: igraph.Graph, t0, tdiff):
  """Compute the vector of in/out degree and their differentials
  for all nodes in the graph H
  #EXP: truncate the other way and look at in/out degrees too
  """
  H_list = {}
  for i in range(3):
    H_list[i] =  H.subgraph_edges(H.es.select(lambda e: min(e['times']) <=t0-i*tdiff), delete_vertices=False)
  #add the in/out degree for the early-truncation graph too
  for j in [1,2]:
    H_list[2+j] =  H.subgraph_edges(H.es.select(lambda e: max(e['times']) >= j*tdiff), delete_vertices=False)
  
  indeg = []
  outdeg = []
  for i in range(5):
    indeg += [H_list[i].indegree()]
    outdeg += [H_list[i].outdegree()]

  #take log(blah+1) and differentials
  indeg = np.vstack(indeg)*1.0
  indeg_log = np.log(indeg+1)
  indeg[0] = indeg_log[0]
  indeg[1] = indeg_log[0]-indeg_log[1]  
  indeg[2] = indeg_log[0]+indeg_log[2]-2*indeg_log[1]  
  indeg[3] = indeg_log[0]-indeg_log[3]
  indeg[4] = indeg_log[0]+indeg_log[4]-2*indeg_log[3]
  
  outdeg = np.vstack(outdeg)*1.0
  outdeg_log = np.log(outdeg+1)
  outdeg[0] = outdeg_log[0]
  outdeg[1] = outdeg_log[0]-outdeg_log[1]  
  outdeg[2] = outdeg_log[0]+outdeg_log[2]-2*outdeg_log[1]  
  outdeg[3] = outdeg_log[0]-outdeg_log[3]
  outdeg[4] = outdeg_log[0]+outdeg_log[4]-2*outdeg_log[4]
  
  return np.vstack((indeg,outdeg))



def compute_node_features(H: igraph.Graph, t0, tdiff):
  """compute node features for H_[t0],H_[t0-tdiff],H_[t0-2*tdiff]
  and do the difference transforms, 
  then return the result as an array
  """
  #compute node features on H, H2, H2'
  H_list = {0: H}
  feats_list = {0: _compute_node_features(H)}
  for i in range(3):
    H_list[i] =  H.subgraph_edges(H.es.select(lambda e: min(e['times']) <=t0-i*tdiff), delete_vertices=False)    
    feats_list[i] = _compute_node_features(H_list[i])
  feats_list = take_derivatives(feats_list)
  #!!EXP: truncate the graph the other way: only look at the graph from time (2*tdiff,t0) and (tdiff*4,t0)
  feats_list_2 = [feats_list[0]]
  H_m = {}
  for j in [1,2]:
    H_m[j] = H.subgraph_edges(H.es.select(lambda e: max(e['times']) >=j*tdiff), delete_vertices=False)    
    #truncate edge times
    truncated_times = [[t for t in tseq if t >= j*tdiff] for tseq in H_m[j].es['times']]
    H_m[j].es['times'] = truncated_times
    feats_list_2 += [_compute_node_features(H_m[j])]
  #take differentials
  feats_list_2 = take_derivatives(feats_list_2)
  feats_list = feats_list + [feats_list_2[1], feats_list_2[2]]
  #add the initial appearance time
  feats_list += [_compute_initial_node_time(H,t0)]
  return np.vstack(feats_list)

def get_feature_names():
  """Get the names of the features"""
  func_names = ['cc','deg','page','n_paper','times']
  idx= [0,1,2]
  feat_names = []
  for i in idx:
    for f in func_names:
      feat_names.append(f + str(i))
  feat_names.append('ini_time')
  return feat_names
  

def compute_edge_features(edgeList,node_features_arr, H_t0 = None): 
  """ Compute edge features based on node features array. 
  If pass in the igraph H_t0, will also compute:
    dice similarity 
  """
  edgeList = np.array(edgeList)
  firstNode = list(edgeList[:,0])
  secondNode = list(edgeList[:,1])
  firstNode_feats = np.vstack(node_features_arr[firstNode])
  secondNode_feats = np.vstack(node_features_arr[secondNode])
  min_feats = np.minimum(firstNode_feats, secondNode_feats)
  max_feats = np.maximum(firstNode_feats, secondNode_feats)
  
  edge_features = np.hstack((min_feats,max_feats))
  if H_t0 is None:
    return edge_features
  else:
    edgeList = list(edgeList)
    dice = np.array(H_t0.similarity_dice(pairs = edgeList))
    dice = dice.reshape((len(dice),1))    
    edge_features = np.hstack((edge_features,dice))
  return edge_features
