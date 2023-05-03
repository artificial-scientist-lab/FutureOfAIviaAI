from datetime import date
from scipy import sparse
import random, time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

NUM_OF_VERTICES = 64719


def create_training_data(full_graph,year_start,years_delta,min_edges=1,edges_used=500000,vertex_degree_cutoff=10):
    """
    :param full_graph: Full graph, numpy array dim(n,3) [vertex 1, vertex 2, time stamp]
    :param year_start: year of graph
    :param years_delta: distance for prediction in years (prediction on graph of year_start+years_delta)
    :param min_edges: minimal number of edges that is considered
    :param edges_used: optional filter to create a random subset of edges for rapid prototyping (default: 500,000)
    :param vertex_degree_cutoff: optional filter, for vertices in training set having a minimal degree of at least vertex_degree_cutoff  (default: 10)
    :return:

    all_edge_list: graph of year_start, numpy array dim(n,2)
    unconnected_vertex_pairs: potential edges for year_start+years_delta
    unconnected_vertex_pairs_solution: numpy array with integers (0=unconnected, 1=connected), solution, length = len(unconnected_vertex_pairs)
    """
    print('\n\n\n\n')
    print('Creating the following data: ')
    
    print('    year_start: ', year_start)
    print('    years_delta: ', years_delta)
    print('    min_edges: ', min_edges)
    print('    edges_used: ', edges_used)
    print('    vertex_degree_cutoff: ', vertex_degree_cutoff)

    years=[year_start,year_start+years_delta]    
    day_origin = date(1990,1,1)

    all_G=[]
    all_edge_lists=[]
    all_sparse=[]
    all_degs=[]
    for yy in years:
        print('    Create Graph for ', yy)
        day_curr=date(yy,12,31)
        all_edges_curr=full_graph[full_graph[:,2]<(day_curr-day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix((np.ones(len(all_edges_curr)), (all_edges_curr[:,0], all_edges_curr[:,1])), shape=(NUM_OF_VERTICES,NUM_OF_VERTICES))
        G_curr=nx.from_scipy_sparse_matrix(adj_mat_sparse_curr, parallel_edges=False, create_using=nx.MultiGraph)

        all_G.append(G_curr)
        all_sparse.append(adj_mat_sparse_curr)
        all_edge_lists.append(all_edges_curr)

        print('    Done: Create Graph for ', yy)
        print('    num of edges: ', G_curr.number_of_edges())

    all_degs=np.array(all_G[0].degree)[:,1]

    ## Create all edges to be predicted
    all_vertices=np.array(range(NUM_OF_VERTICES))
    vertex_large_degs=all_vertices[all_degs>=vertex_degree_cutoff] # use only vertices with degrees larger than 10.

    print('len(vertex_large_degs): ',len(vertex_large_degs))

    unconnected_vertex_pairs=[]
    unconnected_vertex_pairs_solution=[]

    time_start=time.time()
    while len(unconnected_vertex_pairs)<edges_used:        
        i1,i2=random.sample(range(len(vertex_large_degs)), 2)
        
        v1=vertex_large_degs[i1]
        v2=vertex_large_degs[i2]

        if v1!=v2 and not all_G[0].has_edge(v1,v2):
            if len(unconnected_vertex_pairs)%10**6==0:
                time_end=time.time()
                print('    edge progress (',time_end-time_start,'sec): ',len(unconnected_vertex_pairs)/10**6,'M/',edges_used/10**6,'M')
                time_start=time.time()

            is_bigger=False
            if all_G[1].has_edge(v1,v2):
                curr_weight=all_G[1].get_edge_data(v1,v2)[0]['weight']
                if curr_weight>=min_edges:
                    is_bigger=True
            unconnected_vertex_pairs.append((v1,v2))
            unconnected_vertex_pairs_solution.append(is_bigger)

    unconnected_vertex_pairs=np.array(unconnected_vertex_pairs)
    unconnected_vertex_pairs_solution=np.array(list(map(int, unconnected_vertex_pairs_solution)))

    all_edge_list=np.array(all_edge_lists[0])
    
    print('unconnected_vertex_pairs_solution: ',sum(unconnected_vertex_pairs_solution))
    
    return all_edge_list, unconnected_vertex_pairs, unconnected_vertex_pairs_solution




def create_training_data_biased(full_graph,year_start,years_delta,min_edges=1,edges_used=500000,vertex_degree_cutoff=10,data_source=''):
    """
    :param full_graph: Full graph, numpy array dim(n,3) [vertex 1, vertex 2, time stamp]
    :param year_start: year of graph
    :param years_delta: distance for prediction in years (prediction on graph of year_start+years_delta)
    :param min_edges: minimal number of edges that is considered
    :param edges_used: optional filter to create a random subset of edges for rapid prototyping (default: 500,000)
    :param vertex_degree_cutoff: optional filter, for vertices in training set having a minimal degree of at least vertex_degree_cutoff  (default: 10)
    :return:

    all_edge_list: graph of year_start, numpy array dim(n,2)
    unconnected_vertex_pairs: potential edges for year_start+years_delta
    unconnected_vertex_pairs_solution: numpy array with integers (0=unconnected, 1=connected), solution, length = len(unconnected_vertex_pairs)
    """
    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\nin create_training_data_biased')      
    print('\n\n\n\n')
    print('Creating the following data: ')
    
    print('    year_start: ', year_start)
    print('    years_delta: ', years_delta)
    print('    min_edges: ', min_edges)
    print('    edges_used: ', edges_used)
    print('    vertex_degree_cutoff: ', vertex_degree_cutoff)

    years=[year_start,year_start+years_delta]    
    day_origin = date(1990,1,1)

    all_G=[]
    all_edge_lists=[]
    all_sparse=[]
    all_degs=[]
    for yy in years:
        with open("logs_"+data_source+".txt", "a") as myfile:
            myfile.write('\n    Create Graph for '+str(yy))    
        print('    Create Graph for ', yy)
        day_curr=date(yy,12,31)
        all_edges_curr=full_graph[full_graph[:,2]<(day_curr-day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix((np.ones(len(all_edges_curr)), (all_edges_curr[:,0], all_edges_curr[:,1])), shape=(NUM_OF_VERTICES,NUM_OF_VERTICES))
        G_curr=nx.from_scipy_sparse_matrix(adj_mat_sparse_curr, parallel_edges=False, create_using=nx.MultiGraph)

        all_G.append(G_curr)
        all_sparse.append(adj_mat_sparse_curr)
        all_edge_lists.append(all_edges_curr)

        print('    Done: Create Graph for ', yy)
        print('    num of edges: ', G_curr.number_of_edges())

    all_degs=np.array(all_G[0].degree)[:,1]

    ## Create all edges to be predicted
    all_vertices=np.array(range(NUM_OF_VERTICES))
    vertex_large_degs=all_vertices[all_degs>=vertex_degree_cutoff] # use only vertices with degrees larger than 10.

    print('len(vertex_large_degs): ',len(vertex_large_degs))
    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\nlen(vertex_large_degs): '+str(len(vertex_large_degs)))

    unconnected_vertex_pairs=[]
    unconnected_vertex_pairs_solution=[]

    time_start=time.time()
    cT=0
    cF=0
    old_c=0
    while (cT<(edges_used/2)) or (cF<(edges_used/2)):        
        i1,i2=random.sample(range(len(vertex_large_degs)), 2)
        
        v1=vertex_large_degs[i1]
        v2=vertex_large_degs[i2]

        if v1!=v2 and not all_G[0].has_edge(v1,v2):
            if len(unconnected_vertex_pairs)%10**4==0 and len(unconnected_vertex_pairs)!=old_c:
                time_end=time.time()
                print('    edge progress (',time_end-time_start,'sec): ',len(unconnected_vertex_pairs)/10**6,'M/',edges_used/10**6,'M', cT, cF)
                with open("logs_"+data_source+".txt", "a") as myfile:
                    myfile.write('\n    edge progress ('+str(time_end-time_start)+'sec): '+str(len(unconnected_vertex_pairs)/10**6)+'M/'+str(edges_used/10**6)+'M '+str(cT)+' '+str(cF))
                old_c=len(unconnected_vertex_pairs)
                time_start=time.time()
                

            is_bigger=False
            if all_G[1].has_edge(v1,v2):
                curr_weight=all_G[1].get_edge_data(v1,v2)[0]['weight']
                if curr_weight>=min_edges:
                    is_bigger=True

            if is_bigger==False and cF<edges_used/2:
                unconnected_vertex_pairs.append((v1,v2))
                unconnected_vertex_pairs_solution.append(is_bigger)
                cF+=1
            if is_bigger==True and cT<edges_used/2:
                unconnected_vertex_pairs.append((v1,v2))
                unconnected_vertex_pairs_solution.append(is_bigger)
                cT+=1
            
            
    print("(edges_used/2), cT, cF: ", (edges_used/2), cT, cF)
    with open("logs_"+data_source+".txt", "a") as myfile:
        myfile.write('\nnearly done here')

    unconnected_vertex_pairs=np.array(unconnected_vertex_pairs)
    unconnected_vertex_pairs_solution=np.array(list(map(int, unconnected_vertex_pairs_solution)))

    all_edge_list=np.array(all_edge_lists[0])
    
    print('unconnected_vertex_pairs_solution: ',sum(unconnected_vertex_pairs_solution))
    
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
    
    #plt.plot(xpos, ypos)
    #plt.show()
    
    AUC=sum(ROC_vals)/len(ROC_vals)
    return AUC