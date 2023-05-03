import pickle
import numpy as np
from utils import calculate_ROC
import os

from preferential_attachment import preferential_attachment
from common_neighbours import common_neighbours

from multiprocessing import Process

def f(params):

    current_min_edges = params[0]
    curr_vertex_degree_cutoff = params[1]
    current_delta = params[2]

    data_source="SemanticGraph_delta_"+str(current_delta)+"_cutoff_"+str(curr_vertex_degree_cutoff)+"_minedge_"+str(current_min_edges)+".pkl"

    if os.path.isfile(data_source):
        with open(data_source, "rb" ) as pkl_file:
            full_dynamic_graph_sparse, unconnected_vertex_pairs, unconnected_vertex_pairs_solution, year_start, years_delta, vertex_degree_cutoff, min_edges = pickle.load(pkl_file)

            ########################
            ## CHANGE MODEL BELOW ##
            ########################

            # common_neighbours
            # preferential_attachment

            all_idx = common_neighbours(full_dynamic_graph_sparse,
                                                unconnected_vertex_pairs,
                                                data_source)
        
            AUC = calculate_ROC(all_idx, np.array(unconnected_vertex_pairs_solution))
            print('AUC: ', AUC)
        
            with open("AUC Summary.txt", "a") as log_file:
                log_file.write("- ("+str(current_delta)+", "+str(curr_vertex_degree_cutoff)+", "+str(current_min_edges)+"): AUC = "+str(AUC)+"\n")

    else:
            print('File ', data_source, ' does not exist. Proceed to next parameter setting.')

if __name__ == '__main__':

    
    # Loading the validation data.
    #
    # full_dynamic_graph_sparse
    #           The entire semantic network until 2014 (Validation,CompetitionRun=False) or 2017 (Evaluation&Submission,CompetitionRun=True).
    #           numpy array, each entry describes an edge in the semantic network.
    #           Each edge is described by three numbers [v1, v2, t], where the edge is formed at time t between vertices v1 and v2
    #           t is measured in days since the 1.1.1990
    #
    # unconnected_vertex_pairs
    #           numpy array of vertex pairs v1,v2 with deg(v1)>=vertex_degree_cutoff, deg(v2)>=vertex_degree_cutoff, and no edge exists in the year 2014 (2017 for CompetitionRun=True). 
    #           The question that the neural network needs to solve: Will it form at least min_edges edges?
    #
    # unconnected_vertex_pairs_solution
    #           Solution, either yes or no whether edges have been connected
    #
    # year_start
    #           year_start=2014 (2017 for CompetitionRun=True)
    #
    # years_delta
    #           years_delta=3
    #
    # vertex_degree_cutoff
    #           The minimal vertex degree to be used in predictions
    #
    # min_edges
    #           Prediction: From zero to min_edges edges between two vertices
    
    # Testing the model, for validation.
    
    delta_list     = [1,3,5]
    cutoff_list    = [0,5,25]
    min_edges_list = [1,3]

    with open("AUC Summary.txt", "w") as log_file:
        log_file.write("- Prediction from year (2021 - delta, 2021), with delta = [1, 3, 5]\n")
        log_file.write("- Minimal vertex degree: cutoff = [0, 5, 25]\n")
        log_file.write("- Prediction from unconnected to edge_weight = [1, 3] edges\n\n")
        log_file.write("(delta, cutoff, edge_weight):\n")
   
    for current_min_edges in min_edges_list:
        for curr_vertex_degree_cutoff in cutoff_list:           
            for current_delta in delta_list:
            
                params = [current_min_edges, curr_vertex_degree_cutoff, current_delta]

                p = Process(target=f, args=(params,))

                p.start()