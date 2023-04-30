import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from config import NUM_VERTICES


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
    pagerank_scores = np.zeros(shape=(NUM_VERTICES,), dtype=np.float64)
    for i in range(NUM_VERTICES):
        pagerank_scores[i] = pagerank[i]
    return pagerank_scores
