import pickle
import numpy as np
from utils import calculate_ROC
import os

from simple_model import link_prediction_semnet

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
    
    delta_list=[1,3,5]
    cutoff_list=[25,5,0]
    min_edges_list=[1,3]
    
    
            
    for current_delta in delta_list:
        for curr_vertex_degree_cutoff in cutoff_list:
            for current_min_edges in min_edges_list:
    
                data_source="SemanticGraph_delta_"+str(current_delta)+"_cutoff_"+str(curr_vertex_degree_cutoff)+"_minedge_"+str(current_min_edges)+".pkl"
        
                if os.path.isfile(data_source):
                    with open(data_source, "rb" ) as pkl_file:
                        full_dynamic_graph_sparse, unconnected_vertex_pairs, unconnected_vertex_pairs_solution, year_start, years_delta, vertex_degree_cutoff, min_edges = pickle.load(pkl_file)

                    with open("logs_"+data_source+".txt", "a") as myfile:
                        myfile.write('Read'+str(data_source)+'\n') 

                    edges_used=1*10**6
                    percent_positive_examples=1
                    batch_size=400
                    lr_enc=3*10**-5
                    full_rnd_seed=[42]
        
                    for rnd_seed in full_rnd_seed:
                        hyper_paramters=[edges_used,percent_positive_examples,batch_size,lr_enc,rnd_seed]
        
                        all_idx=link_prediction_semnet(full_dynamic_graph_sparse,
                                                       unconnected_vertex_pairs,
                                                       year_start,
                                                       years_delta,
                                                       vertex_degree_cutoff,
                                                       min_edges,
                                                       hyper_paramters,
                                                       data_source
                                                       )
        
                        AUC=calculate_ROC(all_idx, np.array(unconnected_vertex_pairs_solution))
                        print('Area Under Curve for Evaluation: ', AUC,'\n\n\n')
        
                        with open("logs"+data_source[0:-4]+".txt", "a") as log_file:
                            log_file.write("---\n")  
                            log_file.write("edges_used="+str(edges_used)+"\n") 
                            log_file.write("percent_positive_examples="+str(percent_positive_examples)+"\n") 
                            log_file.write("batch_size="+str(batch_size)+"\n") 
                            log_file.write("lr_enc="+str(lr_enc)+"\n") 
                            log_file.write("rnd_seed="+str(rnd_seed)+"\n") 
                            log_file.write("AUC="+str(AUC)+"\n\n") 
                else:
                    print('File ', data_source, ' does not exist. Proceed to next parameter setting.')
