import numpy as np
import scipy.sparse as ss
import pickle

from utils import create_training_data, create_training_data_biased, calculate_ROC, NUM_OF_VERTICES

def preferential_attachment(full_dynamic_graph_sparse, unconnected_vertex_pairs, data_source):

    #### CREATE ADJACENCY MATRIX ####

    # The concatenation is used to produce a symmetric adjacency matrix
    data_rows = np.concatenate([full_dynamic_graph_sparse[:, 0], full_dynamic_graph_sparse[:, 1]])
    data_cols = np.concatenate([full_dynamic_graph_sparse[:, 1], full_dynamic_graph_sparse[:, 0]])
    data_ones = np.ones(len(data_rows), np.uint32)

    adjM = ss.csr_matrix((data_ones, (data_rows, data_cols)), shape=(NUM_OF_VERTICES, NUM_OF_VERTICES))


    #### PREFERENTIAL ATTACHMENT SCORES ####

    degree_vec    = np.asarray(adjM.sum(1)).flatten()
    
    pred_degree_0 = degree_vec[unconnected_vertex_pairs[:,0]]
    pred_degree_1 = degree_vec[unconnected_vertex_pairs[:,1]]
    
    score_list_pa = pred_degree_0 + pred_degree_1


    #### ORDERING THE PREDICTIONS ####

    sorted_predictions_eval = np.argsort(-1.0*score_list_pa)

    return sorted_predictions_eval