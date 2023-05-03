import datetime
import random

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from config import NUM_VERTICES


def format_date(date: datetime.date):
    return f'{date.year}-{date.month:02d}-{date.day:02d}'


def construct_adjacency_matrix(dynamic_graph_sparse: np.ndarray):
    row = np.concatenate([dynamic_graph_sparse[:, 0], dynamic_graph_sparse[:, 1]], axis=0)
    col = np.concatenate([dynamic_graph_sparse[:, 1], dynamic_graph_sparse[:, 0]], axis=0)
    data = np.ones(shape=(dynamic_graph_sparse.shape[0] * 2,), dtype=np.uint64)
    adjacency_matrix = csr_matrix((data, (row, col)), shape=(NUM_VERTICES, NUM_VERTICES))
    adjacency_matrix = adjacency_matrix.tolil()
    adjacency_matrix[54, 54] = 0
    adjacency_matrix[3224, 3224] = 0
    adjacency_matrix = adjacency_matrix.tocsr()
    adjacency_matrix.eliminate_zeros()
    adjacency_matrix = (adjacency_matrix > 0).astype(np.uint64)
    return adjacency_matrix


def get_jaccard_coefficient(
        u: int,
        v: int,
        num_neighbors: np.ndarray,
        adjacency_matrix_squared: np.ndarray
):
    num_neighbors_intersection = adjacency_matrix_squared[u, v]
    num_neighbors_union = num_neighbors[u] + num_neighbors[v] - num_neighbors_intersection
    return num_neighbors_intersection / num_neighbors_union if num_neighbors_union > 0 else None


def get_pagerank_scores(adjacency_matrix: csr_matrix):
    pagerank = nx.algorithms.link_analysis.pagerank(nx.from_scipy_sparse_matrix(adjacency_matrix))
    pagerank_scores = np.zeros(shape=(NUM_VERTICES,), dtype=np.float32)
    for i in range(NUM_VERTICES):
        pagerank_scores[i] = pagerank[i]
    return pagerank_scores


def get_unconnected_vertex_pairs(
        full_dynamic_graph_sparse: np.ndarray,
        train_end_date: datetime.date,
        valid_end_date: datetime.date,
        num_unconnected_vertex_pairs: int
):
    start_date = datetime.date(year=1990, month=1, day=1)

    train_dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (train_end_date - start_date).days]
    valid_dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (valid_end_date - start_date).days]

    train_adjacency_matrix = construct_adjacency_matrix(dynamic_graph_sparse=train_dynamic_graph_sparse)
    train_graph = nx.from_scipy_sparse_matrix(train_adjacency_matrix)
    valid_adjacency_matrix = construct_adjacency_matrix(dynamic_graph_sparse=valid_dynamic_graph_sparse)
    valid_graph = nx.from_scipy_sparse_matrix(valid_adjacency_matrix)

    unconnected_vertex_pairs, unconnected_vertex_pairs_solution = [], []

    progress_bar = tqdm(total=num_unconnected_vertex_pairs)

    for u, v in valid_graph.edges:
        if not train_graph.has_edge(u, v):
            if random.random() < 0.5:
                u, v = v, u
            unconnected_vertex_pairs.append((u, v))
            unconnected_vertex_pairs_solution.append(1)
            train_graph.add_edge(u, v)
            progress_bar.update()

    print(f'Number of positive samples: {len(unconnected_vertex_pairs)}...')

    vertices = list(range(NUM_VERTICES))

    while len(unconnected_vertex_pairs) < num_unconnected_vertex_pairs:
        u, v = random.sample(vertices, k=2)
        if not train_graph.has_edge(u, v):
            unconnected_vertex_pairs.append((u, v))
            unconnected_vertex_pairs_solution.append(0)
            train_graph.add_edge(u, v)
            progress_bar.update()

    unconnected_vertex_pairs = np.array(unconnected_vertex_pairs, dtype=np.uint64)
    unconnected_vertex_pairs_solution = np.array(unconnected_vertex_pairs_solution, dtype=np.uint8)

    return unconnected_vertex_pairs, unconnected_vertex_pairs_solution
