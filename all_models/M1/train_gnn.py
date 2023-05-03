import argparse
import datetime
import random

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.optim as optim
from dgl.nn.pytorch import SAGEConv
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import NUM_VERTICES
from file_utils import read_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=int, choices=[1, 3, 5], default=None)
    parser.add_argument('--cutoff', type=int, choices=[0, 5, 25], default=None)
    parser.add_argument('--minedge', type=int, choices=[1, 3], default=None)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--dnn_hidden_dim', type=int, default=128)
    parser.add_argument('--gnn_dropout_rate', type=float, default=0.2)
    parser.add_argument('--dnn_dropout_rate', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=5_000)
    parser.add_argument('--eval_interval', type=int, default=10_000)
    parser.add_argument('--random_seed', type=int, default=24)
    return parser.parse_args()


def construct_adjacency_matrix(dynamic_graph_sparse: np.ndarray):
    adjacency_matrix = csr_matrix(
        (
            np.ones(shape=len(dynamic_graph_sparse) * 2, dtype=np.uint32),
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
    adjacency_matrix = (adjacency_matrix > 0).astype(np.uint32)
    return adjacency_matrix


def construct_graph(full_dynamic_graph_sparse: np.ndarray, end_date: datetime.date):
    start_date = datetime.date(year=1990, month=1, day=1)
    dynamic_graph_sparse = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (end_date - start_date).days]
    adjacency_matrix = construct_adjacency_matrix(dynamic_graph_sparse=dynamic_graph_sparse)
    dgl_graph = dgl.from_scipy(sp_mat=adjacency_matrix)
    return dgl_graph


class Science4castDataset(Dataset):
    def __init__(self, vertex_pairs: np.ndarray, pairwise_features: np.ndarray, targets: np.ndarray):
        assert vertex_pairs.shape[0] == pairwise_features.shape[0] == targets.shape[0]
        self.vertex_pairs = vertex_pairs
        self.pairwise_features = pairwise_features
        self.targets = targets

    def __getitem__(self, index):
        return self.vertex_pairs[index, :], self.pairwise_features[index, :], self.targets[index]

    def __len__(self):
        return self.targets.shape[0]


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
            F.relu(self.bn_1(self.conv_1.forward(graph, graph_features))),
            p=self.args.gnn_dropout_rate
        )
        graph_features = graph_features + F.dropout(
            F.relu(self.bn_2(self.conv_2.forward(graph, graph_features))),
            p=self.args.gnn_dropout_rate
        )
        graph_features = graph_features + F.dropout(
            F.relu(self.bn_3(self.conv_3.forward(graph, graph_features))),
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

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

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

    train_graph = construct_graph(
        full_dynamic_graph_sparse=full_dynamic_graph_sparse,
        end_date=train_end_date
    )
    valid_graph = construct_graph(
        full_dynamic_graph_sparse=full_dynamic_graph_sparse,
        end_date=valid_end_date
    )
    train_graph = dgl.add_self_loop(train_graph)
    valid_graph = dgl.add_self_loop(valid_graph)
    train_vertex_pairs = np.load(f'cache/train_vertex_pairs_{args.random_seed}.npy').astype(np.int32)
    train_node_features = np.load(f'cache/train_node_features_{args.random_seed}.npy')
    valid_node_features = np.load(f'cache/valid_node_features_{args.random_seed}.npy')
    train_pairwise_features = np.load(f'cache/train_pairwise_features_{args.random_seed}.npy')
    valid_pairwise_features = np.load(f'cache/valid_pairwise_features_{args.random_seed}.npy')
    train_targets = np.load(f'cache/train_targets_{args.random_seed}.npy')

    args.num_node_features = train_node_features.shape[1]
    args.num_pairwise_features = train_pairwise_features.shape[1]

    train_node_mean = np.mean(train_node_features, axis=0, keepdims=True)
    train_node_std = np.std(train_node_features, axis=0, keepdims=True)
    train_pairwise_mean = np.nanmean(train_pairwise_features, axis=0, keepdims=True)
    train_pairwise_std = np.nanstd(train_pairwise_features, axis=0, keepdims=True)

    train_node_features = (train_node_features - train_node_mean) / train_node_std
    valid_node_features = (valid_node_features - train_node_mean) / train_node_std
    train_pairwise_features = (train_pairwise_features - train_pairwise_mean) / train_pairwise_std
    valid_pairwise_features = (valid_pairwise_features - train_pairwise_mean) / train_pairwise_std
    train_pairwise_features = np.nan_to_num(train_pairwise_features, nan=0.0)
    valid_pairwise_features = np.nan_to_num(valid_pairwise_features, nan=0.0)

    train_node_features = torch.from_numpy(train_node_features).cuda()
    valid_node_features = torch.from_numpy(valid_node_features).cuda()

    train_dataset = Science4castDataset(
        vertex_pairs=train_vertex_pairs,
        pairwise_features=train_pairwise_features,
        targets=train_targets
    )
    valid_dataset = Science4castDataset(
        vertex_pairs=valid_unconnected_vertex_pairs,
        pairwise_features=valid_pairwise_features,
        targets=valid_targets
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
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
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.cuda()
    train_graph = train_graph.to('cuda')
    valid_graph = valid_graph.to('cuda')

    total_loss = 0.0
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        for train_vertex_pairs, train_pairwise_features, train_targets in tqdm(train_loader):
            train_vertex_pairs = train_vertex_pairs.long()
            train_pairwise_features = train_pairwise_features.cuda()
            train_targets = train_targets.float().cuda()
            train_predictions = model(
                graph=train_graph,
                node_features=train_node_features,
                vertex_pairs=train_vertex_pairs,
                pairwise_features=train_pairwise_features
            )
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(input=train_predictions, target=train_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            global_step += 1
            if global_step % args.log_interval == 0:
                total_loss /= args.log_interval
                print(f'Step {global_step}: loss {total_loss}...')
                total_loss = 0.0
            if global_step % args.eval_interval == 0:
                predictions = []

                model.eval()

                with torch.no_grad():
                    for valid_vertex_pairs, valid_pairwise_features, _ in valid_loader:
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

                np.save(f'cache/gnn_valid_predictions_{global_step}.npy', predictions)

                model.train()

    assert global_step >= 200_000


if __name__ == '__main__':
    main()
