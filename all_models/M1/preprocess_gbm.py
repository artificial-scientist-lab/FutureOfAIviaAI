import argparse
import datetime
import os
import random

import numpy as np
from scipy.stats import rankdata
from tqdm import trange

from file_utils import read_pickle
from preprocess_utils import construct_adjacency_matrix, format_date, get_jaccard_coefficient, get_pagerank_scores, get_unconnected_vertex_pairs
from time_utils import Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=int, choices=[1, 3, 5], default=None)
    parser.add_argument('--cutoff', type=int, choices=[0, 5, 25], default=None)
    parser.add_argument('--minedge', type=int, choices=[1, 3], default=None)
    parser.add_argument('--num_unconnected_vertex_pairs', type=int, default=1_000_000_000)
    parser.add_argument('--random_seed', type=int, default=24)
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
    adjacency_matrix_squared_filename = f'adjacency_matrices/adjacency_matrix_squared_{format_date(end_date)}.npy'
    if os.path.exists(adjacency_matrix_squared_filename):
        adjacency_matrix_squared = np.load(adjacency_matrix_squared_filename)
    else:
        adjacency_matrix_squared = adjacency_matrix * adjacency_matrix
        adjacency_matrix_squared = adjacency_matrix_squared.toarray().astype(np.uint64)
        np.save(adjacency_matrix_squared_filename, adjacency_matrix_squared)

    dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date_1_year - start_date).days]
    adjacency_matrix_1_year = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse)
    adjacency_matrix_squared_1_year_filename = f'adjacency_matrices/adjacency_matrix_squared_{format_date(end_date_1_year)}.npy'
    if os.path.exists(adjacency_matrix_squared_1_year_filename):
        adjacency_matrix_squared_1_year = np.load(adjacency_matrix_squared_1_year_filename)
    else:
        adjacency_matrix_squared_1_year = adjacency_matrix_1_year * adjacency_matrix_1_year
        adjacency_matrix_squared_1_year = adjacency_matrix_squared_1_year.toarray().astype(np.uint64)
        np.save(adjacency_matrix_squared_1_year_filename, adjacency_matrix_squared_1_year)

    dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date_2_year - start_date).days]
    adjacency_matrix_2_year = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse)
    adjacency_matrix_squared_2_year_filename = f'adjacency_matrices/adjacency_matrix_squared_{format_date(end_date_2_year)}.npy'
    if os.path.exists(adjacency_matrix_squared_2_year_filename):
        adjacency_matrix_squared_2_year = np.load(adjacency_matrix_squared_2_year_filename)
    else:
        adjacency_matrix_squared_2_year = adjacency_matrix_2_year * adjacency_matrix_2_year
        adjacency_matrix_squared_2_year = adjacency_matrix_squared_2_year.toarray().astype(np.uint64)
        np.save(adjacency_matrix_squared_2_year_filename, adjacency_matrix_squared_2_year)

    dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date_180_day - start_date).days]
    adjacency_matrix_180_day = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse)
    adjacency_matrix_squared_180_day_filename = f'adjacency_matrices/adjacency_matrix_squared_{format_date(end_date_180_day)}.npy'
    if os.path.exists(adjacency_matrix_squared_180_day_filename):
        adjacency_matrix_squared_180_day = np.load(adjacency_matrix_squared_180_day_filename)
    else:
        adjacency_matrix_squared_180_day = adjacency_matrix_180_day * adjacency_matrix_180_day
        adjacency_matrix_squared_180_day = adjacency_matrix_squared_180_day.toarray().astype(np.uint64)
        np.save(adjacency_matrix_squared_180_day_filename, adjacency_matrix_squared_180_day)

    dynamic_graph_sparse_2000 = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date - start_date).days]
    dynamic_graph_sparse_2000 = dynamic_graph_sparse_2000[dynamic_graph_sparse_2000[:, 2] >= (cutoff_date_2000 - start_date).days]
    adjacency_matrix_2000 = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse_2000)
    adjacency_matrix_squared_2000_filename = f'adjacency_matrices/adjacency_matrix_squared_2000_01_01_{format_date(end_date)}.npy'
    if os.path.exists(adjacency_matrix_squared_2000_filename):
        adjacency_matrix_squared_2000 = np.load(adjacency_matrix_squared_2000_filename)
    else:
        adjacency_matrix_squared_2000 = adjacency_matrix_2000 * adjacency_matrix_2000
        adjacency_matrix_squared_2000 = adjacency_matrix_squared_2000.toarray().astype(np.uint64)
        np.save(adjacency_matrix_squared_2000_filename, adjacency_matrix_squared_2000)

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

    pagerank_scores_filename = f'link_analysis/page_rank_scores_{format_date(end_date)}.npy'
    if os.path.exists(pagerank_scores_filename):
        pagerank_scores = np.load(pagerank_scores_filename)
    else:
        pagerank_scores = get_pagerank_scores(adjacency_matrix=adjacency_matrix)
        np.save(pagerank_scores_filename, pagerank_scores)

    pagerank_scores_1_year_filename = f'link_analysis/page_rank_scores_{format_date(end_date_1_year)}.npy'
    if os.path.exists(pagerank_scores_1_year_filename):
        pagerank_scores_1_year = np.load(pagerank_scores_1_year_filename)
    else:
        pagerank_scores_1_year = get_pagerank_scores(adjacency_matrix=adjacency_matrix_1_year)
        np.save(pagerank_scores_1_year_filename, pagerank_scores_1_year)

    pagerank_scores_2_year_filename = f'link_analysis/page_rank_scores_{format_date(end_date_2_year)}.npy'
    if os.path.exists(pagerank_scores_2_year_filename):
        pagerank_scores_2_year = np.load(pagerank_scores_2_year_filename)
    else:
        pagerank_scores_2_year = get_pagerank_scores(adjacency_matrix=adjacency_matrix_2_year)
        np.save(pagerank_scores_2_year_filename, pagerank_scores_2_year)

    pagerank_scores_180_day_filename = f'link_analysis/page_rank_scores_{format_date(end_date_180_day)}.npy'
    if os.path.exists(pagerank_scores_180_day_filename):
        pagerank_scores_180_day = np.load(pagerank_scores_180_day_filename)
    else:
        pagerank_scores_180_day = get_pagerank_scores(adjacency_matrix=adjacency_matrix_180_day)
        np.save(pagerank_scores_180_day_filename, pagerank_scores_180_day)

    features = []

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

        features.append([
            num_neighbors_rank[u],
            num_neighbors_rank[v],
            num_neighbors_1_year_rank[u],
            num_neighbors_1_year_rank[v],
            num_neighbors_2_year_rank[u],
            num_neighbors_2_year_rank[v],
            num_neighbors_180_day_rank[u],
            num_neighbors_180_day_rank[v],
            num_neighbors_2000_rank[u],
            num_neighbors_2000_rank[v],
            num_neighbors_diff_1_year_rank[u],
            num_neighbors_diff_1_year_rank[v],
            num_neighbors_diff_2_year_rank[u],
            num_neighbors_diff_2_year_rank[v],
            num_neighbors_diff_180_day_rank[u],
            num_neighbors_diff_180_day_rank[v],
            pagerank_scores[u],
            pagerank_scores[v],
            pagerank_scores_1_year[u],
            pagerank_scores_1_year[v],
            pagerank_scores_2_year[u],
            pagerank_scores_2_year[v],
            pagerank_scores_180_day[u],
            pagerank_scores_180_day[v],
            pagerank_scores[u] - pagerank_scores_1_year[u],
            pagerank_scores[v] - pagerank_scores_1_year[v],
            pagerank_scores[u] - pagerank_scores_2_year[u],
            pagerank_scores[v] - pagerank_scores_2_year[v],
            pagerank_scores[u] - pagerank_scores_180_day[u],
            pagerank_scores[v] - pagerank_scores_180_day[v],
            jaccard_coefficient,
            jaccard_coefficient_1_year,
            jaccard_coefficient_2_year,
            jaccard_coefficient_180_day,
            jaccard_coefficient_2000,
            jaccard_coefficient - jaccard_coefficient_1_year if jaccard_coefficient is not None and jaccard_coefficient_1_year is not None else None,
            jaccard_coefficient - jaccard_coefficient_2_year if jaccard_coefficient is not None and jaccard_coefficient_2_year is not None else None,
            jaccard_coefficient - jaccard_coefficient_180_day if jaccard_coefficient is not None and jaccard_coefficient_180_day is not None else None,
            jaccard_coefficient - jaccard_coefficient_2000 if jaccard_coefficient is not None and jaccard_coefficient_2000 is not None else None,
            u,
            v
        ])

    features = np.array(features, dtype=np.float32)

    return features


def main():
    args = parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    (
        full_dynamic_graph_sparse,
        valid_unconnected_vertex_pairs,
        valid_targets,
        year,
        delta,
        cutoff,
        minedge
    ) = read_pickle(f'data/SemanticGraph_delta_{args.delta}_cutoff_{args.cutoff}_minedge_{args.minedge}.pkl')

    assert args.delta == delta
    assert args.cutoff == cutoff
    assert args.minedge == minedge

    train_end_date = datetime.date(year=year - 1, month=12, day=31)
    valid_end_date = datetime.date(year=year, month=12, day=31)

    with Timer(name='get_unconnected_vertex_pairs'):
        train_unconnected_vertex_pairs, train_unconnected_vertex_pairs_solution = get_unconnected_vertex_pairs(
            full_dynamic_graph_sparse=full_dynamic_graph_sparse,
            train_end_date=train_end_date,
            valid_end_date=valid_end_date,
            num_unconnected_vertex_pairs=args.num_unconnected_vertex_pairs
        )
    with Timer(name='extract_features'):
        train_features = extract_features(
            full_dynamic_graph_sparse=full_dynamic_graph_sparse,
            unconnected_vertex_pairs=train_unconnected_vertex_pairs,
            end_date=train_end_date
        )
    with Timer(name='extract_features'):
        valid_features = extract_features(
            full_dynamic_graph_sparse=full_dynamic_graph_sparse,
            unconnected_vertex_pairs=valid_unconnected_vertex_pairs,
            end_date=valid_end_date
        )

    np.save('cache/train_features.npy', train_features)
    np.save('cache/train_targets.npy', train_unconnected_vertex_pairs_solution)
    np.save('cache/valid_features.npy', valid_features)
    np.save('cache/valid_targets.npy', valid_targets)


if __name__ == '__main__':
    main()
