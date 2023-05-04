import random
import time
import numpy as np
from datetime import date
import math
import os

from scipy import sparse
import networkx as nx

from utils import create_training_data, NUM_OF_VERTICES



def compute_all_properties(v1, v2, adjmat0, adjmat1, adjmat2, squared_adjmat0, squared_adjmat1, squared_adjmat2, 
                                 graph0, graph1, graph2, degrees0, degrees1, degrees2):

    pair_features = []
    
    # ---------------------------------------------------------
    # degree centrality
        
    dc01 = degrees0[v1]
    dc02 = degrees0[v2]
    dc11 = degrees1[v1]
    dc12 = degrees1[v2]
    dc21 = degrees2[v1]
    dc22 = degrees2[v2]

    # get min and max value for each par
    dc0_min = min([dc01, dc02])
    dc0_max = max([dc01, dc02])
    dc1_min = min([dc11, dc12])
    dc1_max = max([dc11, dc12])
    dc2_min = min([dc21, dc22])
    dc2_max = max([dc21, dc22])

    pair_features.append(dc0_min)
    pair_features.append(dc0_max)
    pair_features.append(dc1_min)
    pair_features.append(dc1_max)
    pair_features.append(dc2_min)
    pair_features.append(dc2_max)
        
    # ---------------------------------------------------------
    # total number of neighbors
    
    sn0 = dc01+dc02
    sn1 = dc11+dc12
    sn2 = dc21+dc22
    
    pair_features.append(sn0)
    pair_features.append(sn1)
    pair_features.append(sn2)
       
    # ---------------------------------------------------------
    # Common neighbors index

    pair_features.append(squared_adjmat0[v1,v2]) 
    pair_features.append(squared_adjmat1[v1,v2]) 
    pair_features.append(squared_adjmat2[v1,v2])  
    
    # ---------------------------------------------------------
    # Jaccard index 
       
    if degrees0[v1] == 0 and degrees0[v2] == 0:
        jc0 = 0 
    else:
        jc0 = squared_adjmat0[v1,v2] / (degrees0[v1]+degrees0[v2])
    if degrees1[v1] == 0 and degrees1[v2] == 0:
        jc1 = 0 
    else:
        jc1 = squared_adjmat1[v1,v2] / (degrees1[v1]+degrees1[v2])
    if degrees2[v1] == 0 and degrees2[v2] == 0:
        jc2 = 0 
    else:
        jc2 = squared_adjmat2[v1,v2] / (degrees2[v1]+degrees2[v2])

    pair_features.append(jc0) 
    pair_features.append(jc1)
    pair_features.append(jc2) 

    # ---------------------------------------------------------
    # Simpson index 
    
    if degrees0[v1] == 0 or degrees0[v2] == 0:
        sp0 = 0 
    else:
        sp0 = squared_adjmat0[v1,v2] / np.min([degrees0[v1], degrees0[v2]])
    if degrees1[v1] == 0 or degrees1[v2] == 0:
        sp1 = 0 
    else:
        sp1 = squared_adjmat1[v1,v2] / np.min([degrees1[v1], degrees1[v2]])
    if degrees2[v1] == 0 or degrees2[v2] == 0:
        sp2 = 0 
    else:
        sp2 = squared_adjmat2[v1,v2] / np.min([degrees2[v1], degrees2[v2]])

    pair_features.append(sp0) 
    pair_features.append(sp1) 
    pair_features.append(sp2) 

    # ---------------------------------------------------------
    # geometric index
    
    if degrees0[v1] == 0 or degrees0[v2] == 0:
        gm0 = 0 
    else:
        gm0 = squared_adjmat0[v1,v2]**2 / (degrees0[v1]*degrees0[v2])
    if degrees1[v1] == 0 or degrees1[v2]==0:
        gm1 = 0 
    else:
        gm1 = squared_adjmat1[v1,v2]**2 / (degrees1[v1]*degrees1[v2])
    if degrees2[v1] == 0 or degrees2[v2] == 0:
        gm2 = 0 
    else:
        gm2 = squared_adjmat2[v1,v2]**2 / (degrees2[v1]*degrees2[v2])
    
    pair_features.append(gm0)
    pair_features.append(gm1)
    pair_features.append(gm2) 
    
    # ---------------------------------------------------------
    # cosine index
    
    pair_features.append(math.sqrt(gm0)) 
    pair_features.append(math.sqrt(gm1))
    pair_features.append(math.sqrt(gm2)) 
    
    # ---------------------------------------------------------
    # adamic-adar index
    
    pred = nx.adamic_adar_index(graph0, [(v1, v2)])
    for u,v,aa0 in pred:
        pair_features.append(aa0)
    
    pred = nx.adamic_adar_index(graph1, [(v1, v2)])
    for u,v,aa1 in pred:
        pair_features.append(aa1)
    
    pred = nx.adamic_adar_index(graph2, [(v1, v2)])
    for u,v,aa2 in pred:
        pair_features.append(aa2) 
       
    # --------------------------------------------------------- 
    # resource-allocation index 
    
    pred = nx.resource_allocation_index(graph0, [(v1, v2)])
    for u,v,ra0 in pred:
        pair_features.append(ra0) 
    
    pred = nx.resource_allocation_index(graph1, [(v1, v2)])
    for u,v,ra1 in pred:
        pair_features.append(ra1)
        
    pred = nx.resource_allocation_index(graph2, [(v1, v2)])
    for u,v,ra2 in pred:
        pair_features.append(ra2) 
        
    # ---------------------------------------------------------   
    # preferential attatchement

    pred = nx.preferential_attachment(graph0, [(v1, v2)])
    for u,v,pa0 in pred:
        pair_features.append(pa0) 
    
    pred = nx.preferential_attachment(graph1, [(v1, v2)])
    for u,v,pa1 in pred:
        pair_features.append(pa1) 
        
    pred = nx.preferential_attachment(graph2, [(v1, v2)])
    for u,v,pa2 in pred:
        pair_features.append(pa2)

    return pair_features


def compute_all_properties_of_list(all_ajdmat,vlist,data_source):
    """
    Computes hand-crafted properties for all vertices in vlist
    """
    time_start=time.time()
    
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
    
    all_properties=[]
    print('  All matrix squares are now computed...')
    for ii in range(len(vlist)):
        
        v1 = vlist[ii][0]
        v2 = vlist[ii][1]
        
        vals=compute_all_properties(v1, v2, adjmat0, adjmat1, adjmat2, squared_adjmat0, squared_adjmat1, squared_adjmat2,
                                          graph0, graph1, graph2, degrees0, degrees1, degrees2)

        all_properties.append(vals)
        
        log_checking = round(0.01*len(vlist))
        
        if ii%log_checking==0:
            print('   extract features progress: (',time.time()-time_start,'sec) ', str(round(ii/len(vlist),3)*100), '% pairs computed')
            time_start=time.time()

    return all_properties


def feature_extraction(full_dynamic_graph_sparse,unconnected_vertex_pairs,year_start,years_delta,vertex_degree_cutoff,min_edges,data_source):
    """
    code used for extract features and create training and testing datasets
    """
    
    print(' \nLearning to predict using training data from '+str(year_start-years_delta)+' -> '+str(year_start))

    print('  Create training data for year '+str(year_start-years_delta))

    train_dynamic_graph_sparse,training_edges,training_edges_solution = create_training_data(full_dynamic_graph_sparse, year_start-years_delta, years_delta, min_edges=min_edges, vertex_degree_cutoff=vertex_degree_cutoff, data_source=data_source)

    day_origin = date(1990,1,1)
    years=[year_start-years_delta,year_start-years_delta-1,year_start-years_delta-2]

    training_adjmats = []
     
    for y in years:
        print('   Creating the graph for year: ', y)
        day_curr=date(y,12,31)
        train_edges_curr=train_dynamic_graph_sparse[train_dynamic_graph_sparse[:,2]<(day_curr-day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix((np.ones(len(train_edges_curr)), (train_edges_curr[:,0], train_edges_curr[:,1])), shape=(NUM_OF_VERTICES,NUM_OF_VERTICES))
        training_adjmats.append(adj_mat_sparse_curr)
        
    # Get positive edges
    idx_pos = np.where(training_edges_solution==1)[0]

    if len(idx_pos)>500000:
        idx_to_select = 500000
        idx_pos = np.asarray(random.sample(idx_pos.tolist(), idx_to_select))
        data_edges_pos = training_edges[idx_pos]
    else:
        data_edges_pos = training_edges[idx_pos]

    # Get same number of negative edges
    idx_to_select = len(idx_pos)
    all_idx_neg = np.where(training_edges_solution==0)[0]
    idx_neg = np.asarray(random.sample(all_idx_neg.tolist(), idx_to_select))
    data_edges_neg = training_edges[idx_neg]
    
    print('  Computing network properties for positive edges of training data year=',year_start-3)

    data_train_positive = compute_all_properties_of_list(training_adjmats, data_edges_pos, data_source)
    data_train_positive = np.array(data_train_positive)
    
    print('  \nComputing network properties for negative edges of training data year=',year_start-3)
        
    data_train_negative = compute_all_properties_of_list(training_adjmats, data_edges_neg, data_source)
    data_train_negative = np.array(data_train_negative)
    
    
    # Final training dataset
    training_features = np.concatenate((data_train_positive, data_train_negative))
    training_label = np.concatenate((np.ones(idx_to_select), np.zeros(idx_to_select)))
    
    print('  Features extracted to all training pairs')

    # save training data

    folder_to_save = os.path.join(os.getcwd(),'extracted_data')
    
    training_features_file_name = "TrainingFeatures_delta_"+str(years_delta)+"_cutoff_"+str(vertex_degree_cutoff)+"_minedge_"+str(min_edges)+".npy"
    training_label_file_name = "TrainingLabel_delta_"+str(years_delta)+"_cutoff_"+str(vertex_degree_cutoff)+"_minedge_"+str(min_edges)+".npy"
    
    training_features_file = os.path.join(folder_to_save,training_features_file_name)
    training_label_file = os.path.join(folder_to_save,training_label_file_name)
    
    with open(training_features_file, 'wb') as f:
        np.save(f, training_features)
    with open(training_label_file, 'wb') as f:
        np.save(f, training_label)

    # Create properties for evaluation
    
    print('  Extracting features for '+str(year_start)+' -> '+str(year_start+years_delta)+' data.')
    
    years=[year_start,year_start-1,year_start-2]
    
    evaluation_adjmats=[]
    for y in years:
   
        print('    Create Graph for ', y)
        day_curr=date(y,12,31)
        eval_edges_curr=full_dynamic_graph_sparse[full_dynamic_graph_sparse[:,2]<(day_curr-day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix(
                                                (np.ones(len(eval_edges_curr)), (eval_edges_curr[:,0], eval_edges_curr[:,1])),
                                                shape=(NUM_OF_VERTICES,NUM_OF_VERTICES)
                                               )

        evaluation_adjmats.append(adj_mat_sparse_curr)
    
    
    print('  \nComputing network properties for the evaluation data')

    pairs_to_predict = unconnected_vertex_pairs

    print('  testing data = ' + str(len(pairs_to_predict)) + '  pairs')

    evaluation_features = compute_all_properties_of_list(evaluation_adjmats, pairs_to_predict, data_source)
    evaluation_features = np.array(evaluation_features)
    
    # save evaluation data
    
    evaluation_features_file_name = "EvaluationFeatures_delta_"+str(years_delta)+"_cutoff_"+str(vertex_degree_cutoff)+"_minedge_"+str(min_edges)+".npy"
    evaluation_label_file = os.path.join(folder_to_save,evaluation_features_file_name)
    
    with open(evaluation_label_file, 'wb') as f:
        np.save(f, evaluation_features)
    
    print('  Features extracted to all evaluation pairs')
   
        
