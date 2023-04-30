import argparse
import os
import pickle
import sys

import numpy as np

from utils import calculate_ROC, write_to_log, is_andrew
if is_andrew():
    from FutureOfAIviaAI import embedding_model_grarep as embedding_model
    from FutureOfAIviaAI import settings
    from FutureOfAIviaAI.embedding_model_grarep import link_prediction_embednet, log_file, results_log_file
else:
    import embedding_model_grarep as embedding_model
    import settings
    from embedding_model_grarep import link_prediction_embednet, log_file, results_log_file

con = 1000000
# original_stdout = sys.stdout
# sys.stdout = open('log.txt', 'w')

if __name__ == '__main__':

    # Log file
    # embedding_model.log_file = open('log.log', 'w')
    # embedding_model.results_log_file = open('results.log', 'w')

    # Command line args
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("-delta", help="Delta", type=int)
    parser.add_argument("-cutoff", help="Cutoff", type=int)
    parser.add_argument("-minedges", help="Min Edges", type=int)
    parser.add_argument("-eu", help="Edges used", type=int)
    parser.add_argument("-ppe", help="Percent positive examples", type=int)
    parser.add_argument("-bs", help="Batch size", type=int)
    parser.add_argument("-lr", help="Batch size", type=float)
    parser.add_argument("-dim", help="Dimension", type=int)
    parser.add_argument("-wl", help="Walk length", type=int)
    parser.add_argument("-nw", help="Num walks", type=int)
    parser.add_argument("-p", help="Num walks", type=float)
    parser.add_argument("-q", help="Num walks", type=float)
    parser.add_argument("-w", help="Window", type=int)
    parser.add_argument("-neg", help="Negative", type=int)
    parser.add_argument("-e", help="Epochs", type=int)
    parser.add_argument("-bw", help="Batch words", type=int)
    parser.add_argument("-s", help="Size", type=int)
    parser.add_argument("-os", help="Osize", type=int)
    args = parser.parse_args()

    embedding_model.DIMENSIONS = args.dim
    embedding_model.WALK_LENGTH = args.wl
    embedding_model.NUM_WALKS = args.nw
    embedding_model.P = args.p
    embedding_model.Q = args.q
    embedding_model.WINDOW = args.w
    embedding_model.NEGATIVE = args.neg
    embedding_model.EPOCHS = args.e
    embedding_model.BATCH_WORDS = args.bw
    embedding_model.SIZE = args.bw
    embedding_model.OSIZE = args.bw

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

    # delta_list = [1, 3, 5]
    delta_list = [args.delta]
    # cutoff_list = [25, 5, 0]
    cutoff_list = [args.cutoff]
    # min_edges_list = [1, 3]
    min_edges_list = [args.minedges]

    for current_delta in delta_list:
        for curr_vertex_degree_cutoff in cutoff_list:
            for current_min_edges in min_edges_list:
                settings.DELTA_VAL = current_delta
                settings.CUTOFF_VAL = curr_vertex_degree_cutoff
                settings.MINEDGES_VAL = current_min_edges

                write_to_log(os.getcwd(), file=log_file)
                data_source = "../data/SemanticGraph_delta_" + str(current_delta) + "_cutoff_" + str(
                    curr_vertex_degree_cutoff) + "_minedge_" + str(current_min_edges) + ".pkl"
                write_to_log('Data source: ' + data_source + "\n", file=results_log_file)

                if os.path.isfile(data_source):
                    with open(data_source, "rb") as pkl_file:
                        full_dynamic_graph_sparse, unconnected_vertex_pairs, unconnected_vertex_pairs_solution, year_start, years_delta, vertex_degree_cutoff, min_edges = pickle.load(
                            pkl_file)

                    # with open(get_log_location(data_source), "a") as myfile:
                    #     myfile.write('Read' + str(data_source) + '\n')

                    edges_used = args.eu
                    percent_positive_examples = args.ppe
                    batch_size = args.bs
                    lr_enc = args.lr
                    full_rnd_seed = [42, 55, 69, 120]

                    for rnd_seed in full_rnd_seed:
                        hyper_paramters = [edges_used, percent_positive_examples, batch_size, lr_enc, rnd_seed]

                        all_idx = link_prediction_embednet(full_dynamic_graph_sparse,
                                                           unconnected_vertex_pairs[:con],
                                                           year_start,
                                                           years_delta,
                                                           vertex_degree_cutoff,
                                                           min_edges,
                                                           hyper_paramters,
                                                           data_source
                                                           )

                        AUC = calculate_ROC(all_idx, np.array(unconnected_vertex_pairs_solution[:con]))
                        write_to_log('Area Under Curve for Evaluation: ', AUC, '\n\n\n', file=log_file)

                        with open("morelogs" + data_source[0:-4] + ".txt", "a") as log_file:
                            log_file.write("---\n")
                            log_file.write("edges_used=" + str(edges_used) + "\n")
                            log_file.write("percent_positive_examples=" + str(percent_positive_examples) + "\n")
                            log_file.write("batch_size=" + str(batch_size) + "\n")
                            log_file.write("lr_enc=" + str(lr_enc) + "\n")
                            log_file.write("rnd_seed=" + str(rnd_seed) + "\n")
                            log_file.write("AUC=" + str(AUC) + "\n\n")
                else:
                    write_to_log('File ', data_source, ' does not exist. Proceed to next parameter setting.', file=log_file)
