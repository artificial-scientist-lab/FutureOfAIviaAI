import pickle
import os
import numpy as np

from features_functions import feature_extraction
from model_functions import pca_data, get_predictions
from utils import calculate_ROC


if __name__ == '__main__':

    # create folder to save data (if it does not exist yet)
    folder_path = os.path.join(os.getcwd(),'extracted_data')
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    # parameters for evaluation
    delta_list = [3,5]
    cutoff_list = [25,5,0]
    min_edges_list = [1,3]

            
    for current_delta in delta_list:
        for curr_vertex_degree_cutoff in cutoff_list:
            for current_min_edges in min_edges_list:
                
                task = "delta_"+str(current_delta)+"_cutoff_"+str(curr_vertex_degree_cutoff)+"_minedge_"+str(current_min_edges)
                print('\n======\nCURRENT TASK: ' + task + '\n======')
                data_source = os.path.join(os.getcwd(),'semantic_graphs', 'SemanticGraph_'+ task+'.pkl')
            
                if os.path.isfile(data_source):
                    with open(data_source, "rb" ) as pkl_file:
                        full_dynamic_graph_sparse, unconnected_vertex_pairs, unconnected_vertex_pairs_solution, year_start, years_delta, vertex_degree_cutoff, min_edges = pickle.load(pkl_file)

                    print('\nEXTRACTING FEATURES....')

                    feature_extraction(full_dynamic_graph_sparse,
                                        unconnected_vertex_pairs,
                                        year_start,
                                        years_delta,
                                        vertex_degree_cutoff,
                                        min_edges,
                                        data_source)

                    print('\nPERFORMING PCA TRANSFORMATIONS....')
                    data_train, data_eval, label_train = pca_data(task)
                    
                    print('\nGETTING PREDICTIONS FOR EVALUATION DATA....')
                    eval_predictions = get_predictions(data_train, label_train, data_eval)

                    print('\nCOMPUTING MODEL PERFORMANCE....')
                    AUC = AUC=calculate_ROC(eval_predictions, np.array(unconnected_vertex_pairs_solution))
                    
                    print('\n----\nAUC: ' + str(AUC) + '\n----')
        
                    with open("models_performance.txt", "a") as log_file:
                        log_file.write("---\n")  
                        log_file.write(task+"\n") 
                        log_file.write("AUC="+str(AUC)+"\n\n") 
                        
                    print('result saved in models_performance.txt file')

                else:
                    print('\nFile ', data_source, ' does not exist. Proceed to next parameter setting.')
                    
                print('\n..............................................................')