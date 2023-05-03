import pickle
import numpy as np
import os
from main import main

def clean_up():
    for file in ['field.txt', 'train_full_2017_0.5_0.9.txt', 'train_full_2017_0.5_1.txt', 'embed']:
        os.system(f'rm {file}')
    for directory in ['raw/*', 'processed', 'HOPREC/2017_raw_count/*']:
        os.system(f'rm -rfv ../data/{directory}')
    os.system('rm -rfv ../model_outputs')

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
    
    ROC_vals=np.array(ROC_vals)/max(ypos)
    
    AUC=sum(ROC_vals)/len(ROC_vals)
    return AUC

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
    
                data_source="../data/SemanticGraph_delta_"+str(current_delta)+"_cutoff_"+str(curr_vertex_degree_cutoff)+"_minedge_"+str(current_min_edges)+".pkl"

                if os.path.isfile(data_source):
                    with open(data_source, "rb" ) as pkl_file:
                        full_dynamic_graph_sparse, unconnected_vertex_pairs, unconnected_vertex_pairs_solution, year_start, years_delta, vertex_degree_cutoff, min_edges = pickle.load(pkl_file)
                    
                    curr_file_name = '../data/raw/CompetitionSet2017_3.pkl'
                    full_dynamic_graph_sparse,unconnected_vertex_pairs,year_start,years_delta
                    with open(curr_file_name, "wb") as output_file:
                        pickle.dump([
                            full_dynamic_graph_sparse,
                            unconnected_vertex_pairs,
                            year_start,                
                            years_delta], output_file)


                    all_idx=main()
        
                    AUC=calculate_ROC(all_idx, np.array(unconnected_vertex_pairs_solution))
                    print(data_source)
                    print('Area Under Curve for Evaluation: ', AUC,'\n\n\n')
                    with open("logs.txt", "a") as myfile:
                        myfile.write(f'Read {data_source}\n')
                        myfile.write(f'Area Under Curve for Evaluation: {AUC} \n\n\n')
                    clean_up()
                else:
                    print('File ', data_source, ' does not exist. Proceed to next parameter setting.')
