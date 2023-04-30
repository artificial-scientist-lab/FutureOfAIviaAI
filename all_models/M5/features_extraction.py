# -*- coding: utf-8 -*-
"""
@author: Francisco Valente

Note: some parts of this script are based on the tutorial of the competition:
    https://github.com/iarai/science4cast/blob/main/Tutorial/tutorial.ipynb
"""

# imports
import pickle
import numpy as np
import networkx as nx
from datetime import date
import math
import random
from scipy import sparse
from itertools import combinations


### user input
vertex_degree_cutoff = 10

## global variables
day_origin = date(1990,1,1)
num_nodes = 64719 

######## FUNCTIONS TO EXTRACT TOPOLOGICAL FEATURES

def compute_topological_features(v1, v2, adjmat0, adjmat1, adjmat2, squared_adjmat0, squared_adjmat1, squared_adjmat2, 
                                 graph0, graph1, graph2, degrees0, degrees1, degrees2):

    pair_features = []
    
    # degree centrality

    pair_features.append(degrees0[v1]) # 0
    pair_features.append(degrees0[v2]) # 1
    pair_features.append(degrees1[v1]) # 2
    pair_features.append(degrees1[v2]) # 3
    pair_features.append(degrees2[v1]) # 4
    pair_features.append(degrees2[v2]) # 5
    
    
    # total number of neighbors
    
    sn01 = np.array(adjmat0.sum(0))[0][v1]
    sn02 = np.array(adjmat0.sum(0))[0][v2]
    sn0 = sn01+sn02
    sn11 = np.array(adjmat1.sum(0))[0][v1]
    sn12 = np.array(adjmat1.sum(0))[0][v2]
    sn1 = sn11+sn12
    sn21 = np.array(adjmat2.sum(0))[0][v1]
    sn22 = np.array(adjmat2.sum(0))[0][v2]
    sn2 = sn21+sn22
    
    pair_features.append(sn0) # 6
    pair_features.append(sn1) # 7
    pair_features.append(sn2) # 8
   
    
    # Common neighbors index

    pair_features.append(squared_adjmat0[v1,v2]) # 9
    pair_features.append(squared_adjmat1[v1,v2]) # 10
    pair_features.append(squared_adjmat2[v1,v2]) # 11  
    
    
    # Jaccard index 
   
    if degrees0[v1]==0 and degrees0[v2]==0:
        jc0 = 0 
    else:
        jc0 = squared_adjmat0[v1,v2]/(degrees0[v1]+degrees0[v2])
    if degrees1[v1]==0 and degrees1[v2]==0:
        jc1 = 0 
    else:
        jc1 = squared_adjmat1[v1,v2]/(degrees1[v1]+degrees1[v2])
    if degrees2[v1]==0 and degrees2[v2]==0:
        jc2 = 0 
    else:
        jc2 = squared_adjmat2[v1,v2]/(degrees2[v1]+degrees2[v2])

    pair_features.append(jc0) # 12
    pair_features.append(jc1) # 13
    pair_features.append(jc2) # 14


    # Simpson index 
    
    if degrees0[v1]==0 or degrees0[v2]==0:
        sp0 = 0 
    else:
        sp0 = squared_adjmat0[v1,v2]/np.min([degrees0[v1],degrees0[v2]])
    if degrees1[v1]==0 or degrees1[v2]==0:
        sp1 = 0 
    else:
        sp1 = squared_adjmat1[v1,v2]/np.min([degrees1[v1],degrees1[v2]])
    if degrees2[v1]==0 or degrees2[v2]==0:
        sp2 = 0 
    else:
        sp2 = squared_adjmat2[v1,v2]/np.min([degrees2[v1],degrees2[v2]])

    pair_features.append(sp0) # 15
    pair_features.append(sp1) # 16
    pair_features.append(sp2) # 17
    
    
    # geometric index
    
    if degrees0[v1]==0 or degrees0[v2]==0:
        gm0 = 0 
    else:
        gm0 = squared_adjmat0[v1,v2]**2/(degrees0[v1]*degrees0[v2])
    if degrees1[v1]==0 or degrees1[v2]==0:
        gm1 = 0 
    else:
        gm1 = squared_adjmat1[v1,v2]**2/(degrees1[v1]*degrees1[v2])
    if degrees2[v1]==0 or degrees2[v2]==0:
        gm2 = 0 
    else:
        gm2 = degrees1[v1,v2]**2/(degrees2[v1]*degrees2[v2])
    
    pair_features.append(gm0) # 18
    pair_features.append(gm1) # 19
    pair_features.append(gm2) # 20
    
    
    # cosine index 
    
    pair_features.append(math.sqrt(gm0)) # 21
    pair_features.append(math.sqrt(gm1)) # 22
    pair_features.append(math.sqrt(gm2)) # 23
    
    
    # adamic-adar index
    
    pred = nx.adamic_adar_index(graph0, [(v1, v2)])
    for u,v,aa0 in pred:
        pair_features.append(aa0) # 24
    
    pred = nx.adamic_adar_index(graph1, [(v1, v2)])
    for u,v,aa1 in pred:
        pair_features.append(aa1) # 25
    
    pred = nx.adamic_adar_index(graph2, [(v1, v2)])
    for u,v,aa2 in pred:
        pair_features.append(aa2) # 26
       
        
    # resource-allocation index 
    
    pred = nx.resource_allocation_index(graph0, [(v1, v2)])
    for u,v,ra0 in pred:
        pair_features.append(ra0) # 27
    
    pred = nx.resource_allocation_index(graph1, [(v1, v2)])
    for u,v,ra1 in pred:
        pair_features.append(ra1) # 28
        
    pred = nx.resource_allocation_index(graph2, [(v1, v2)])
    for u,v,ra2 in pred:
        pair_features.append(ra2) # 29
        
        
    # preferential attatchement
        
    pred = nx.preferential_attachment(graph0, [(v1, v2)])
    for u,v,pa0 in pred:
        pair_features.append(pa0) # 30
    
    pred = nx.preferential_attachment(graph1, [(v1, v2)])
    for u,v,pa1 in pred:
        pair_features.append(pa1) # 31
        
    pred = nx.preferential_attachment(graph2, [(v1, v2)])
    for u,v,pa2 in pred:
        pair_features.append(pa2) # 32     
    
    
    # average degree of the neighborhood
    
    av0 = nx.average_neighbor_degree(graph0, nodes = [v1,v2])
    av01 = av0[v1]
    av02 = av0[v2]
    av1 = nx.average_neighbor_degree(graph1, nodes = [v1,v2])
    av11 = av1[v1]
    av12 = av1[v2]
    av2 = nx.average_neighbor_degree(graph2, nodes = [v1,v2])
    av21 = av2[v1]
    av22 = av2[v2]
    
    pair_features.append(av01) # 33
    pair_features.append(av02) # 34
    pair_features.append(av11) # 35
    pair_features.append(av12) # 36
    pair_features.append(av21) # 37
    pair_features.append(av22) # 38


    # clustering coefficient

    ci01  = nx.clustering(graph0, v1)
    ci02  = nx.clustering(graph0, v2)
    ci11  = nx.clustering(graph1, v1)
    ci12  = nx.clustering(graph1, v2)
    ci21  = nx.clustering(graph2, v1)
    ci22  = nx.clustering(graph2, v2)
    
    pair_features.append(ci01) # 39
    pair_features.append(ci02) # 40
    pair_features.append(ci11) # 41
    pair_features.append(ci12) # 42
    pair_features.append(ci21) # 43
    pair_features.append(ci22) # 44
    

    return pair_features


def extract_features(all_ajdmat, pairs_list):

    # adjacency matrix
    adjmat0 = all_ajdmat[0]
    adjmat1 = all_ajdmat[1]
    adjmat2 = all_ajdmat[2]
    
    # squared adjacency matrix
    squared_adjmat0 = adjmat0**2
    squared_adjmat1 = adjmat1**2
    squared_adjmat2 = adjmat2**2
    
    # graphs
    graph0 = nx.from_scipy_sparse_matrix(adjmat0)
    graph1 = nx.from_scipy_sparse_matrix(adjmat1)
    graph2 = nx.from_scipy_sparse_matrix(adjmat2)
    
    # nodes' degrees
    degrees0 = np.array(adjmat0.sum(0))[0]
    degrees1 = np.array(adjmat1.sum(0))[0]
    degrees2 = np.array(adjmat2.sum(0))[0]

    all_pairs_features = []
    
    for idx in range(len(pairs_list)):
        
        v1 = pairs_list[idx][0]
        v2 = pairs_list[idx][1]

        if idx%100000==0:
            print(idx, '/', len(pairs_list), ' pairs computed')
            
        single_pair_features = compute_topological_features(v1, v2, adjmat0, adjmat1, adjmat2, squared_adjmat0, squared_adjmat1, squared_adjmat2,
                                          graph0, graph1, graph2, degrees0, degrees1, degrees2)

        all_pairs_features.append(single_pair_features)
        
    print('Topological features extracted for all pairs of nodes')

    return all_pairs_features


######## GET TRAINING OR EVALUATION DATASETS

## TRAINING DATA

data_source='TrainSet2014_3.pkl'
full_dynamic_graph_sparse, unconnected_vertex_pairs, year_start, years_delta = pickle.load(open(data_source, "rb" ))

# function to get training data from the semantic network (based on the tutorial of the competion)
def create_training_data(full_graph, year_start, years_delta, vertex_degree_cutoff=10):
    """
    :param full_graph: Full graph, numpy array dim(n,3) [vertex 1, vertex 2, time stamp]
    :param year_start: year of graph
    :param years_delta: distance for prediction in years (prediction on graph of year_start+years_delta)
    :param vertex_degree_cutoff: optional filter, for vertices in training set having a minimal degree of at least vertex_degree_cutoff  (default: 10)
    :return:

    all_edge_list: graph of year_start, numpy array dim(n,2)
    unconnected_vertex_pairs: potential edges for year_start+years_delta
    unconnected_vertex_pairs_solution: numpy array with integers (0=unconnected, 1=connected), solution, length = len(unconnected_vertex_pairs)
    vertex_large_degs: list of nodes with vertex_degree_cutoff >= the selected threshold
    """

    years=[year_start,year_start+years_delta]    

    # create graph for year start (2011) and year start+delta (2014)
    all_G=[]
    all_edge_lists=[]
    all_sparse=[]
        
    for y in years:
        day_curr=date(y,12,31)
        all_edges_curr=full_graph[full_graph[:,2]<(day_curr-day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix((np.ones(len(all_edges_curr)), (all_edges_curr[:,0], all_edges_curr[:,1])), shape=(num_nodes, num_nodes))
        G_curr=nx.from_scipy_sparse_matrix(adj_mat_sparse_curr, parallel_edges=False, create_using=None, edge_attribute='weight')

        all_G.append(G_curr)
        all_sparse.append(adj_mat_sparse_curr)
        all_edge_lists.append(all_edges_curr)
        
    all_degs=np.array(all_sparse[0].sum(0))[0]

    ## Create all edges to be predicted
    all_vertices=np.array(range(num_nodes))
    vertex_large_degs=all_vertices[all_degs>=vertex_degree_cutoff] # use only vertices with degrees larger than 10.

    unconnected_vertex_pairs=[]
    unconnected_vertex_pairs_solution=[]
    
    # list of nodes with degree>=10
    vertex_large_degs = vertex_large_degs.tolist()
    # list of all combinations of those nodes
    list_of_edges = list(combinations(vertex_large_degs,2))
   
    # get all edges between nodes with degree>=10 and that are not connected in 2014
    for idx in range(0, len(list_of_edges)):
        v1,v2 = list_of_edges[idx]
        if not all_G[0].has_edge(v1,v2):
            
            if idx%10000000==0:
                print(idx, '/', len(list_of_edges), ' edges computed')
        
            unconnected_vertex_pairs.append((v1,v2))
            unconnected_vertex_pairs_solution.append(all_G[1].has_edge(v1,v2))
            
    print('All training edges computed...')
            
    unconnected_vertex_pairs=np.array(unconnected_vertex_pairs)
    unconnected_vertex_pairs_solution=np.array(list(map(int, unconnected_vertex_pairs_solution)))
    all_edge_list=np.array(all_edge_lists[0])
    
    return all_edge_list, unconnected_vertex_pairs, unconnected_vertex_pairs_solution, vertex_large_degs


train_dynamic_graph_sparse, training_edges, training_edges_solution, vertex_large_degs = create_training_data(full_dynamic_graph_sparse, year_start-years_delta, years_delta, vertex_degree_cutoff=vertex_degree_cutoff)

## create training adjacency matrices (2009-2011)

years=[year_start-3,year_start-4,year_start-5]
training_adjmats = []
 
for y in years:
    day_curr=date(y,12,31)
    train_edges_curr=train_dynamic_graph_sparse[train_dynamic_graph_sparse[:,2]<(day_curr-day_origin).days]
    adj_mat_sparse_curr = sparse.csr_matrix((np.ones(len(train_edges_curr)), (train_edges_curr[:,0], train_edges_curr[:,1])), shape=(num_nodes,num_nodes))
    training_adjmats.append(adj_mat_sparse_curr)


# Get positive edges (connected in 2014)
idx_pos = np.where(training_edges_solution==1)[0]
data_edges_pos = training_edges[idx_pos]

# Get same number of negative edges (unconnected in 2014)
idx_to_select = len(idx_pos)
all_idx_neg = np.where(training_edges_solution==1)[0]
idx_neg = random.sample(all_idx_neg, idx_to_select)
data_edges_neg = training_edges[idx_neg]

# Extract topological features

print('Computing the topoligical features for the training data...')

data_train_positive = extract_features(training_adjmats, data_edges_pos)
data_train_positive = np.array(data_train_positive)
data_train_negative = extract_features(training_adjmats, data_edges_neg)
data_train_negative = np.array(data_train_negative)

# Final training dataset
training_features = np.concatenate((data_train_positive, data_train_negative))
training_label = np.concatenate((np.ones(idx_to_select), np.zeros(idx_to_select)))

# save data
with open('data_training.npy', 'wb') as f:
    np.save(f, training_features)
with open('data_evaluation.npy', 'wb') as f:
    np.save(f, training_label)
    
print('Training data computed and saved...')


##### EVALUTION DATA

data_source='CompetitionSet2017_3.pkl'
full_dynamic_graph_sparse, unconnected_vertex_pairs, year_start, years_delta = pickle.load(open(data_source, "rb"))

## create evaluation adjacency matrices (2015-2017)

years=[year_start,year_start-1,year_start-2]
evaluation_adjmats = []

for y in years:

    day_curr=date(y,12,31)
    eval_edges_curr=full_dynamic_graph_sparse[full_dynamic_graph_sparse[:,2]<(day_curr-day_origin).days]
    adj_mat_sparse_curr = sparse.csr_matrix((np.ones(len(eval_edges_curr)), (eval_edges_curr[:,0], eval_edges_curr[:,1])),
                                            shape=(num_nodes, num_nodes))

    evaluation_adjmats.append(adj_mat_sparse_curr)

print('Computing the topoligical features for the evaluation data...')

pairs_to_predict = unconnected_vertex_pairs
evaluation_features = extract_features(evaluation_adjmats, pairs_to_predict)
evaluation_features = np.array(evaluation_features)

# save data
with open('data_evaluation.npy', 'wb') as f:
    np.save(f, evaluation_features)
    
print('Evaluation data computed and saved...')
