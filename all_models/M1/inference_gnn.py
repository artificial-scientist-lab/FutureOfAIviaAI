import argparse
import datetime
import math

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from dgl.nn.pytorch import SAGEConv
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import NUM_VERTICES
from file_utils import read_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--dnn_hidden_dim', type=int, default=128)
    parser.add_argument('--gnn_dropout_rate', type=float, default=0.2)
    parser.add_argument('--dnn_dropout_rate', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()


def construct_adjacency_matrix(dynamic_graph_sparse: np.ndarray, end_date_index: int):
    adjacency_matrix = csr_matrix(
        (
            np.ones(shape=len(dynamic_graph_sparse) * 2, dtype=np.uint64),
            (
                np.concatenate([dynamic_graph_sparse[:, 0], dynamic_graph_sparse[:, 1]]),
                np.concatenate([dynamic_graph_sparse[:, 1], dynamic_graph_sparse[:, 0]])
            )
        ),
        shape=(NUM_VERTICES, NUM_VERTICES)
    )
    adjacency_matrix = adjacency_matrix.tolil()
    adjacency_matrix[54, 54] = 0
    adjacency_matrix[3224, 3224] = 0
    adjacency_matrix = adjacency_matrix.tocsr()
    adjacency_matrix.eliminate_zeros()
    adjacency_matrix = (adjacency_matrix > 0).astype(np.float32)
    adjacency_matrix = adjacency_matrix.tolil()
    for u, v, t in tqdm(dynamic_graph_sparse):
        weight = math.e ** (-0.0001 * (end_date_index - t))
        if u != v:
            adjacency_matrix[u, v] = weight
            adjacency_matrix[v, u] = weight
    adjacency_matrix = adjacency_matrix.tocsr()
    return adjacency_matrix


def construct_graph(full_dynamic_graph_sparse: np.ndarray, end_date: datetime.date):
    start_date = datetime.date(year=1990, month=1, day=1)
    dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date - start_date).days]
    adjacency_matrix = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse, end_date_index=(end_date - start_date).days)
    dgl_graph = dgl.from_scipy(sp_mat=adjacency_matrix, eweight_name='weight')
    return dgl_graph


class Science4castDataset(Dataset):
    def __init__(self, vertex_pairs: np.ndarray, pairwise_features: np.ndarray):
        assert vertex_pairs.shape[0] == pairwise_features.shape[0]
        self.vertex_pairs = vertex_pairs
        self.pairwise_features = pairwise_features

    def __getitem__(self, index):
        return self.vertex_pairs[index, :], self.pairwise_features[index, :]

    def __len__(self):
        return self.pairwise_features.shape[0]


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float):
        super(DenseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=in_channels, out_features=out_channels)
        )

    def forward(self, x: torch.FloatTensor):
        return self.block(x)


class Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Model, self).__init__()
        self.args = args
        gnn_hidden_dim = args.num_node_features + args.embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=NUM_VERTICES,
            embedding_dim=args.embedding_dim
        )
        self.conv_1 = SAGEConv(
            in_feats=gnn_hidden_dim,
            out_feats=gnn_hidden_dim,
            aggregator_type='mean'
        )
        self.conv_2 = SAGEConv(
            in_feats=gnn_hidden_dim,
            out_feats=gnn_hidden_dim,
            aggregator_type='mean'
        )
        self.conv_3 = SAGEConv(
            in_feats=gnn_hidden_dim,
            out_feats=gnn_hidden_dim,
            aggregator_type='mean'
        )
        self.bn_1 = nn.BatchNorm1d(num_features=gnn_hidden_dim)
        self.bn_2 = nn.BatchNorm1d(num_features=gnn_hidden_dim)
        self.bn_3 = nn.BatchNorm1d(num_features=gnn_hidden_dim)
        self.dense_1 = DenseBlock(
            in_channels=gnn_hidden_dim * 2 + args.num_pairwise_features,
            out_channels=args.dnn_hidden_dim,
            dropout_rate=args.dnn_dropout_rate
        )
        self.dense_2 = DenseBlock(
            in_channels=gnn_hidden_dim * 2 + args.num_pairwise_features + args.dnn_hidden_dim,
            out_channels=args.dnn_hidden_dim,
            dropout_rate=args.dnn_dropout_rate
        )
        self.dense_3 = DenseBlock(
            in_channels=gnn_hidden_dim * 2 + args.num_pairwise_features + args.dnn_hidden_dim * 2,
            out_channels=args.dnn_hidden_dim,
            dropout_rate=args.dnn_dropout_rate
        )
        self.dense_4 = DenseBlock(
            in_channels=gnn_hidden_dim * 2 + args.num_pairwise_features + args.dnn_hidden_dim * 3,
            out_channels=args.dnn_hidden_dim,
            dropout_rate=args.dnn_dropout_rate
        )
        self.dense_5 = DenseBlock(
            in_channels=gnn_hidden_dim * 2 + args.num_pairwise_features + args.dnn_hidden_dim * 4,
            out_channels=1,
            dropout_rate=args.dnn_dropout_rate
        )

    def forward(
            self,
            graph: dgl.DGLGraph,
            node_features: torch.FloatTensor,
            vertex_pairs: torch.LongTensor,
            pairwise_features: torch.FloatTensor
    ):
        graph_features = torch.cat([node_features, self.embedding.weight], dim=-1)
        graph_features = F.dropout(
            F.relu(self.bn_1(self.conv_1.forward(graph, graph_features, edge_weight=graph.edata['weight']))),
            p=self.args.gnn_dropout_rate
        )
        graph_features = graph_features + F.dropout(
            F.relu(self.bn_2(self.conv_2.forward(graph, graph_features, edge_weight=graph.edata['weight']))),
            p=self.args.gnn_dropout_rate
        )
        graph_features = graph_features + F.dropout(
            F.relu(self.bn_3(self.conv_3.forward(graph, graph_features, edge_weight=graph.edata['weight']))),
            p=self.args.gnn_dropout_rate
        )
        hidden_states = torch.cat([
            graph_features[vertex_pairs[:, 0]],
            graph_features[vertex_pairs[:, 1]],
            pairwise_features
        ], dim=1)
        hidden_states = torch.cat([hidden_states, self.dense_1(hidden_states)], dim=1)
        hidden_states = torch.cat([hidden_states, self.dense_2(hidden_states)], dim=1)
        hidden_states = torch.cat([hidden_states, self.dense_3(hidden_states)], dim=1)
        hidden_states = torch.cat([hidden_states, self.dense_4(hidden_states)], dim=1)
        predictions = self.dense_5(hidden_states).squeeze(1)
        return predictions


def main():
    args = parse_args()

    full_dynamic_graph_sparse, valid_vertex_pairs, year_start, _ = read_pickle('data/CompetitionSet2017_3.pkl')

    valid_end_date = datetime.date(year=year_start, month=12, day=31)

    valid_graph = construct_graph(
        full_dynamic_graph_sparse=full_dynamic_graph_sparse,
        end_date=valid_end_date
    )
    valid_node_features = np.load('cache/valid_node_features.npy')
    valid_pairwise_features = np.load('cache/valid_pairwise_features.npy')

    args.num_node_features = valid_node_features.shape[1]
    args.num_pairwise_features = valid_pairwise_features.shape[1]

    gnn_stats = np.load('cache/gnn_stats.npz')

    valid_node_features = (valid_node_features - gnn_stats['train_node_mean']) / gnn_stats['train_node_std']
    valid_pairwise_features = (valid_pairwise_features - gnn_stats['train_pairwise_mean']) / gnn_stats['train_pairwise_std']
    valid_pairwise_features = np.nan_to_num(valid_pairwise_features, nan=0.0)

    valid_node_features = torch.from_numpy(valid_node_features).cuda()

    valid_dataset = Science4castDataset(
        vertex_pairs=valid_vertex_pairs,
        pairwise_features=valid_pairwise_features
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = Model(args=args)
    model.load_state_dict(torch.load('model/model.bin')['model'])
    model.cuda()
    valid_graph = valid_graph.to('cuda')

    predictions = []

    model.eval()

    with torch.no_grad():
        for valid_vertex_pairs, valid_pairwise_features in valid_loader:
            valid_vertex_pairs = valid_vertex_pairs.long()
            valid_pairwise_features = valid_pairwise_features.cuda()
            valid_predictions = torch.sigmoid(model(
                graph=valid_graph,
                node_features=valid_node_features,
                vertex_pairs=valid_vertex_pairs,
                pairwise_features=valid_pairwise_features
            ))
            predictions.append(valid_predictions.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    np.save('submission/gnn_predictions.npy', predictions)


if __name__ == '__main__':
    main()
