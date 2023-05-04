from datetime import date
import time
import numpy as np
from itertools import combinations

from scipy import sparse
import networkx as nx

NUM_OF_VERTICES=64719


def create_training_data(full_graph, year_start, years_delta, min_edges=1, vertex_degree_cutoff=10, data_source=''):
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
    
    # with open("logs_"+data_source+".txt", "a") as myfile:
    #     myfile.write('\ncreate_training_data')      
    print('\nCreating the following data: ')
    
    print(' year_start: ', year_start)
    print(' years_delta: ', years_delta)
    print(' vertex_degree_cutoff: ', vertex_degree_cutoff)
    
    # in my method a minimal degree of 10 is set in training data, even for smaller testing cutoffs |(as in the competition)
    if vertex_degree_cutoff < 10:
        vertex_degree_cutoff = 10

    years=[year_start,year_start+years_delta]    
    day_origin = date(1990,1,1)
    
    # create graph for year start (2011) and year start+delta (2014)
    all_G=[]
    all_edge_lists=[]
    all_sparse=[]
        
    for y in years:
        # with open("logs_"+data_source+".txt", "a") as myfile:
        #     myfile.write('\n    Create Graph for '+str(y))    
        print('   Create Graph for year: ', y)
        day_curr=date(y,12,31)
        all_edges_curr=full_graph[full_graph[:,2]<(day_curr-day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix((np.ones(len(all_edges_curr)), (all_edges_curr[:,0], all_edges_curr[:,1])), shape=(NUM_OF_VERTICES, NUM_OF_VERTICES))
        G_curr=nx.from_scipy_sparse_matrix(adj_mat_sparse_curr, parallel_edges=False, create_using=nx.MultiGraph)

        all_G.append(G_curr)
        all_sparse.append(adj_mat_sparse_curr)
        all_edge_lists.append(all_edges_curr)
        
        print('  Done: Create Graph for year:', y)
        print('  num of edges: ', G_curr.number_of_edges())
        
    all_degs=np.array(all_sparse[0].sum(0))[0]

    ## Create all edges to be predicted
    all_vertices=np.array(range(NUM_OF_VERTICES))
    vertex_large_degs=all_vertices[all_degs>=vertex_degree_cutoff] # use only vertices with degrees larger than 10.

    unconnected_vertex_pairs=[]
    unconnected_vertex_pairs_solution=[]
    
    # list of nodes with degree>=10
    vertex_large_degs = vertex_large_degs.tolist()
    # list of all combinations of those nodes
    list_of_edges = list(combinations(vertex_large_degs,2))
   
    print('  len(vertex_large_degs): ',len(vertex_large_degs))

    log_checking = round(0.01*len(list_of_edges))
    time_start=time.time()
    # get all edges between nodes with degree>=10 and that are not connected in 2014
    for idx in range(0, len(list_of_edges)):
        v1,v2 = list_of_edges[idx]
        if not all_G[0].has_edge(v1,v2):
            
            if idx%log_checking==0:
                time_end=time.time()
                print('   edge progress (',time_end-time_start,'sec): ', str(round(idx/len(list_of_edges),2)*100), '% edges computed')
                time_start=time.time()
                
            is_bigger=False
            if all_G[1].has_edge(v1,v2):
                curr_weight=all_G[1].get_edge_data(v1,v2)[0]['weight']
                if curr_weight>=min_edges:
                    is_bigger=True

            unconnected_vertex_pairs.append((v1,v2))
            unconnected_vertex_pairs_solution.append(is_bigger)
            
    print('  All training edges computed...')
        
    unconnected_vertex_pairs=np.array(unconnected_vertex_pairs)
    unconnected_vertex_pairs_solution=np.array(list(map(int, unconnected_vertex_pairs_solution)))
    all_edge_list=np.array(all_edge_lists[0])
    
    print('  number unconnected_vertex_pairs_solution: ',sum(unconnected_vertex_pairs_solution))
    
    return all_edge_list, unconnected_vertex_pairs, unconnected_vertex_pairs_solution



def calculate_ROC(data_vertex_pairs,data_solution):
    data_solution=np.array(data_solution)
    data_vertex_pairs_sorted=data_solution[data_vertex_pairs]
    
    xpos=[0]
    ypos=[0]
    ROC_vals=[]
    for ii in range(len(data_vertex_pairs_sorted)):
        if data_vertex_pairs_sorted[ii]==1:
            xpos.append(xpos[-1])
            ypos.append(ypos[-1]+1)
        if data_vertex_pairs_sorted[ii]==0:
            xpos.append(xpos[-1]+1)
            ypos.append(ypos[-1])      
            ROC_vals.append(ypos[-1])
    
        # # # # # # # # # # # # # # # 
        # 
        # We normalize the ROC curve such that it starts at (0,0) and ends at (1,1).
        # Then our final metric of interest is the Area under that curve.
        # AUC is between [0,1].
        # AUC = 0.5 is acchieved by random predictions
        # AUC = 1.0 stands for perfect prediction.
    
    ROC_vals=np.array(ROC_vals)/max(ypos)
    ypos=np.array(ypos)/max(ypos)
    xpos=np.array(xpos)/max(xpos)
    
    # plt.plot(xpos, ypos)
    # plt.show()
    
    AUC=sum(ROC_vals)/len(ROC_vals)
    return AUC