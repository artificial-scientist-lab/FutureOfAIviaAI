import pickle
import os
import numpy as np
from datetime import date

delta_list=[1,3,5]
cutoff_list=[25,5,0]
min_edges_list=[1,3]

fns = []
for current_delta in delta_list:
    for curr_vertex_degree_cutoff in cutoff_list:
        for current_min_edges in min_edges_list:
            s = "SemanticGraph_delta_"+str(current_delta)+"_cutoff_"+str(curr_vertex_degree_cutoff)+"_minedge_"+str(current_min_edges)
            fns.append(s)

NUM_OF_VERTICES=64719
past = 5

def filter_by_year(full_graph, yy):
    day_origin = date(1990, 1, 1)
    day_curr = date(yy, 12, 31)
    return full_graph[full_graph[:,2]<(day_curr-day_origin).days]

for fn in fns:
    data_source = fn+".pkl"

    if os.path.isfile(data_source):
        print('reading', data_source)
        
        with open(data_source, "rb" ) as pkl_file:
            full_dynamic_graph_sparse, unconnected_vertex_pairs, unconnected_vertex_pairs_solution, year_start, years_delta, vertex_degree_cutoff, min_edges = pickle.load(pkl_file)
            
            for yy in range(1994, 2021): 
                print('year', yy)
                edg_fn = "edg/"+fn+'_year_'+str(yy)+'.edg'
                if os.path.isfile(edg_fn):
                    continue
                
                graph = filter_by_year(full_dynamic_graph_sparse, yy)
                np.savetxt(edg_fn, graph[:, :-1], delimiter='\t', fmt='%d')
            
