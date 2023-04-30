import pickle
import numpy as np
from utils import create_training_data

if not ('full_dynamic_graph_sparse' in locals() or 'full_dynamic_graph_sparse' in globals()):
    print('Read full graph')
    with open('all_edges.pkl', "rb") as pkl_file:
        full_dynamic_graph_sparse_read = pickle.load(pkl_file)

    full_dynamic_graph_sparse = []
    cc = 0
    for edge in full_dynamic_graph_sparse_read:
        if edge[0] != edge[1]:
            full_dynamic_graph_sparse.append(edge)
        else:
            print(cc, ': ', edge)

    full_dynamic_graph_sparse_read = []

else:
    print('already stored all_edges')

print('done')

all_vertex_degree_cutoff = [25, 5, 0]
all_delta = [1, 3, 5]
all_min_edges = [1, 3, 5]

for curr_vertex_degree_cutoff in all_vertex_degree_cutoff:
    for current_delta in all_delta:
        for current_min_edges in all_min_edges:
            print('---')
            year_start = 2020 - current_delta
            train_dynamic_graph_sparse, train_edges_for_checking, train_edges_solution = create_training_data(
                full_graph=np.array(full_dynamic_graph_sparse),
                year_start=year_start,
                years_delta=current_delta,
                min_edges=current_min_edges,
                edges_used=1e7,
                vertex_degree_cutoff=curr_vertex_degree_cutoff
            )
            print('current_delta: ', current_delta, '; curr_vertex_degree_cutoff: ', curr_vertex_degree_cutoff,
                  '; current_min_edges: ', current_min_edges)
            print('len(train_dynamic_graph_sparse): ', len(train_dynamic_graph_sparse))
            curr_file_name = "SemanticGraph_delta_" + str(current_delta) + "_cutoff_" + str(
                curr_vertex_degree_cutoff) + "_minedge_" + str(current_min_edges) + ".pkl"

            with open(curr_file_name, "wb") as output_file:
                pickle.dump([
                    train_dynamic_graph_sparse,
                    train_edges_for_checking,
                    train_edges_solution,
                    year_start,
                    current_delta,
                    curr_vertex_degree_cutoff,
                    current_min_edges
                ], output_file)
