import argparse
import datetime
import random

import numpy as np
from scipy.stats import rankdata
from tqdm import trange

from config import NUM_VERTICES
from file_utils import read_pickle
from preprocess_utils import construct_adjacency_matrix, get_jaccard_coefficient, get_pagerank_scores
from time_utils import Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=3090)
    return parser.parse_args()


def extract_features(
        full_dynamic_graph_sparse: np.ndarray,
        unconnected_vertex_pairs: np.ndarray,
        end_date: datetime.date
):
    start_date = datetime.date(year=1990, month=1, day=1)
    cutoff_date_2000 = datetime.date(year=1999, month=12, day=31)
    end_date_1_year = datetime.date(end_date.year - 1, month=end_date.month, day=end_date.day)
    end_date_2_year = datetime.date(end_date.year - 2, month=end_date.month, day=end_date.day)
    end_date_180_day = end_date - datetime.timedelta(days=180)

    dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date - start_date).days]
    adjacency_matrix = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse)
    adjacency_matrix_squared = adjacency_matrix * adjacency_matrix
    adjacency_matrix_squared = adjacency_matrix_squared.toarray()

    dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date_1_year - start_date).days]
    adjacency_matrix_1_year = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse)
    adjacency_matrix_squared_1_year = adjacency_matrix_1_year * adjacency_matrix_1_year
    adjacency_matrix_squared_1_year = adjacency_matrix_squared_1_year.toarray()

    dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date_2_year - start_date).days]
    adjacency_matrix_2_year = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse)
    adjacency_matrix_squared_2_year = adjacency_matrix_2_year * adjacency_matrix_2_year
    adjacency_matrix_squared_2_year = adjacency_matrix_squared_2_year.toarray()

    dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date_180_day - start_date).days]
    adjacency_matrix_180_day = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse)
    adjacency_matrix_squared_180_day = adjacency_matrix_180_day * adjacency_matrix_180_day
    adjacency_matrix_squared_180_day = adjacency_matrix_squared_180_day.toarray()

    dynamic_graph_sparse_2000 = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date - start_date).days]
    dynamic_graph_sparse_2000 = dynamic_graph_sparse_2000[dynamic_graph_sparse_2000[:, 2] >= (cutoff_date_2000 - start_date).days]
    adjacency_matrix_2000 = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse_2000)
    adjacency_matrix_squared_2000 = adjacency_matrix_2000 * adjacency_matrix_2000
    adjacency_matrix_squared_2000 = adjacency_matrix_squared_2000.toarray()

    num_neighbors = np.array(adjacency_matrix.sum(axis=0)).reshape(-1)
    num_neighbors_1_year = np.array(adjacency_matrix_1_year.sum(axis=0)).reshape(-1)
    num_neighbors_2_year = np.array(adjacency_matrix_2_year.sum(axis=0)).reshape(-1)
    num_neighbors_180_day = np.array(adjacency_matrix_180_day.sum(axis=0)).reshape(-1)
    num_neighbors_diff_1_year = num_neighbors - num_neighbors_1_year
    num_neighbors_diff_2_year = num_neighbors - num_neighbors_2_year
    num_neighbors_diff_180_day = num_neighbors - num_neighbors_180_day
    num_neighbors_2000 = np.array(adjacency_matrix_2000.sum(axis=0)).reshape(-1)

    num_neighbors_rank = rankdata(num_neighbors)
    num_neighbors_1_year_rank = rankdata(num_neighbors_1_year)
    num_neighbors_2_year_rank = rankdata(num_neighbors_2_year)
    num_neighbors_180_day_rank = rankdata(num_neighbors_180_day)
    num_neighbors_diff_1_year_rank = rankdata(num_neighbors_diff_1_year)
    num_neighbors_diff_2_year_rank = rankdata(num_neighbors_diff_2_year)
    num_neighbors_diff_180_day_rank = rankdata(num_neighbors_diff_180_day)
    num_neighbors_2000_rank = rankdata(num_neighbors_2000)

    pagerank_scores = get_pagerank_scores(adjacency_matrix=adjacency_matrix)
    pagerank_scores_1_year = get_pagerank_scores(adjacency_matrix=adjacency_matrix_1_year)
    pagerank_scores_2_year = get_pagerank_scores(adjacency_matrix=adjacency_matrix_2_year)
    pagerank_scores_180_day = get_pagerank_scores(adjacency_matrix=adjacency_matrix_180_day)

    node_features = []

    for u in trange(NUM_VERTICES):
        node_features.append([
            num_neighbors_rank[u],
            num_neighbors_1_year_rank[u],
            num_neighbors_2_year_rank[u],
            num_neighbors_2000_rank[u],
            num_neighbors_diff_1_year_rank[u],
            num_neighbors_diff_2_year_rank[u],
            pagerank_scores[u],
            pagerank_scores_1_year[u],
            pagerank_scores_2_year[u],
            pagerank_scores[u] - pagerank_scores_1_year[u],
            pagerank_scores[u] - pagerank_scores_2_year[u]
        ])

    node_features = np.array(node_features, dtype=np.float32)

    pairwise_features = []

    for i in trange(len(unconnected_vertex_pairs)):
        u, v = unconnected_vertex_pairs[i]
        jaccard_coefficient = get_jaccard_coefficient(
            u=u,
            v=v,
            num_neighbors=num_neighbors,
            adjacency_matrix_squared=adjacency_matrix_squared
        )
        jaccard_coefficient_1_year = get_jaccard_coefficient(
            u=u,
            v=v,
            num_neighbors=num_neighbors_1_year,
            adjacency_matrix_squared=adjacency_matrix_squared_1_year
        )
        jaccard_coefficient_2_year = get_jaccard_coefficient(
            u=u,
            v=v,
            num_neighbors=num_neighbors_2_year,
            adjacency_matrix_squared=adjacency_matrix_squared_2_year
        )
        jaccard_coefficient_180_day = get_jaccard_coefficient(
            u=u,
            v=v,
            num_neighbors=num_neighbors_180_day,
            adjacency_matrix_squared=adjacency_matrix_squared_180_day
        )
        jaccard_coefficient_2000 = get_jaccard_coefficient(
            u=u,
            v=v,
            num_neighbors=num_neighbors_2000,
            adjacency_matrix_squared=adjacency_matrix_squared_2000
        )

        pairwise_features.append([
            num_neighbors_rank[u],
            num_neighbors_rank[v],
            num_neighbors_1_year_rank[u],
            num_neighbors_1_year_rank[v],
            num_neighbors_2_year_rank[u],
            num_neighbors_2_year_rank[v],
            num_neighbors_2000_rank[u],
            num_neighbors_2000_rank[v],
            num_neighbors_diff_1_year_rank[u],
            num_neighbors_diff_1_year_rank[v],
            num_neighbors_diff_2_year_rank[u],
            num_neighbors_diff_2_year_rank[v],
            pagerank_scores[u],
            pagerank_scores[v],
            pagerank_scores_1_year[u],
            pagerank_scores_1_year[v],
            pagerank_scores_2_year[u],
            pagerank_scores_2_year[v],
            pagerank_scores[u] - pagerank_scores_1_year[u],
            pagerank_scores[v] - pagerank_scores_1_year[v],
            pagerank_scores[u] - pagerank_scores_2_year[u],
            pagerank_scores[v] - pagerank_scores_2_year[v],
            jaccard_coefficient,
            jaccard_coefficient_1_year,
            jaccard_coefficient_2_year,
            jaccard_coefficient_2000,
            jaccard_coefficient - jaccard_coefficient_1_year if jaccard_coefficient is not None and jaccard_coefficient_1_year is not None else 0.0,
            jaccard_coefficient - jaccard_coefficient_2_year if jaccard_coefficient is not None and jaccard_coefficient_2_year is not None else 0.0,
            jaccard_coefficient - jaccard_coefficient_180_day if jaccard_coefficient is not None and jaccard_coefficient_180_day is not None else 0.0,
            jaccard_coefficient - jaccard_coefficient_2000 if jaccard_coefficient is not None and jaccard_coefficient_2000 is not None else 0.0,
            num_neighbors_180_day_rank[u],
            num_neighbors_180_day_rank[v],
            num_neighbors_diff_180_day_rank[u],
            num_neighbors_diff_180_day_rank[v],
            pagerank_scores_180_day[u],
            pagerank_scores_180_day[v],
            pagerank_scores[u] - pagerank_scores_180_day[u],
            pagerank_scores[v] - pagerank_scores_180_day[v],
            jaccard_coefficient_180_day,
        ])

    pairwise_features = np.array(pairwise_features, dtype=np.float32)

    return node_features, pairwise_features


def main():
    args = parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    full_dynamic_graph_sparse, valid_unconnected_vertex_pairs, year_start, _ = read_pickle('data/CompetitionSet2017_3.pkl')
    valid_end_date = datetime.date(year=year_start, month=12, day=31)

    with Timer(name='extract_features'):
        valid_node_features, valid_pairwise_features = extract_features(
            full_dynamic_graph_sparse=full_dynamic_graph_sparse,
            unconnected_vertex_pairs=valid_unconnected_vertex_pairs,
            end_date=valid_end_date
        )

    np.save('cache/valid_node_features.npy', valid_node_features)
    np.save('cache/valid_pairwise_features.npy', valid_pairwise_features)


if __name__ == '__main__':
    main()
