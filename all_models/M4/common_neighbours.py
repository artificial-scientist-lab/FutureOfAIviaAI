import numpy as np
import scipy.sparse as ss
import pickle

from utils import create_training_data, create_training_data_biased, calculate_ROC, NUM_OF_VERTICES

def common_neighbours(full_dynamic_graph_sparse, unconnected_vertex_pairs, data_source):
    
    #### CREATE ADJACENCY MATRIX ####

    # The concatenation is used to produce a symmetric adjacency matrix
    data_rows = np.concatenate([full_dynamic_graph_sparse[:, 0], full_dynamic_graph_sparse[:, 1]])
    data_cols = np.concatenate([full_dynamic_graph_sparse[:, 1], full_dynamic_graph_sparse[:, 0]])
    data_ones = np.ones(len(data_rows), np.uint32)

    adjM_csr = ss.csr_matrix((data_ones, (data_rows, data_cols)), shape=(NUM_OF_VERTICES, NUM_OF_VERTICES))
    adjM_csc = ss.csc_matrix((data_ones, (data_rows, data_cols)), shape=(NUM_OF_VERTICES, NUM_OF_VERTICES))


    #### COMMON NEIGHBOURS SCORES ####
    
    score_list_cn = np.array([])

    n = 10**7

    for pair in unconnected_vertex_pairs[:n]:
        cn_val = (adjM_csr[pair[0],:]*adjM_csc[:,pair[1]])[0,0]
        score_list_cn = np.append(score_list_cn, cn_val)


    #### ORDERING THE PREDICTIONS ####

    sorted_predictions_eval = np.argsort(-1.0*score_list_cn)

    return sorted_predictions_eval