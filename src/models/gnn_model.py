"""
GNN-based meta-evaluator for Neural IVM.

Architecture:
  Input: graph A (kNN), node features X, partition labels π
  → GNN layers process (X || π_encoding, A)
  → Partition-aware pooling (pool within each cluster, then aggregate)
  → MLP head → scalar AMI prediction
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "data" / "clustering_runs"

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GINConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning("torch_geometric not installed. GNN model unavailable.")


def _sparse_to_edge_index(A_sparse):
    """Convert scipy sparse matrix to PyG edge_index tensor."""
    A_coo = sparse.coo_matrix(A_sparse)
    edge_index = torch.tensor(
        np.vstack([A_coo.row, A_coo.col]), dtype=torch.long
    )
    return edge_index


def _encode_partition(labels, max_clusters=32):
    """
    Encode partition labels as node features.
    Uses a hash-based encoding for cluster IDs to handle variable k.

    Args:
        labels: (n,) array of cluster labels.
        max_clusters: Dimension of the encoding.

    Returns:
        (n, max_clusters) tensor.
    """
    n = len(labels)
    encoding = torch.zeros(n, max_clusters)

    unique = np.unique(labels)
    for c in unique:
        mask = labels == c
        if c == -1:
            # Noise gets a special encoding (last column)
            encoding[mask, -1] = 1.0
        else:
            col = int(c) % (max_clusters - 1)
            encoding[mask, col] = 1.0

    return encoding


class GNNSurrogateNet(nn.Module):
    """GNN network for predicting AMI from (graph, features, partition)."""

    def __init__(self, input_dim, hidden_dim=128, n_layers=3, partition_dim=32, dropout=0.1):
        super().__init__()
        self.partition_dim = partition_dim

        # Input projection
        total_input = input_dim + partition_dim
        self.input_proj = nn.Linear(total_input, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

        # Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+partition-aware pool
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        partition = data.partition  # (N, partition_dim)

        # Concatenate node features with partition encoding
        x = torch.cat([x, partition], dim=-1)
        x = F.relu(self.input_proj(x))

        # GNN layers
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # residual

        # Global mean pool
        global_pool = global_mean_pool(x, batch)

        # Partition-aware pooling: mean within each cluster, then mean of cluster reps
        # For batched graphs, we do this per-graph
        partition_pool = self._partition_aware_pool(x, data.partition_labels, batch)

        # Combine pools
        combined = torch.cat([global_pool, partition_pool], dim=-1)
        return self.readout(combined).squeeze(-1)

    def _partition_aware_pool(self, x, partition_labels, batch):
        """Pool within each cluster, then average cluster representations."""
        device = x.device
        batch_size = batch.max().item() + 1
        result = torch.zeros(batch_size, x.size(-1), device=device)

        for b in range(batch_size):
            mask = batch == b
            x_b = x[mask]
            labels_b = partition_labels[mask]

            unique_labels = torch.unique(labels_b)
            # Exclude noise (-1)
            unique_labels = unique_labels[unique_labels >= 0]

            if len(unique_labels) == 0:
                result[b] = x_b.mean(dim=0)
                continue

            cluster_reps = []
            for c in unique_labels:
                c_mask = labels_b == c
                cluster_reps.append(x_b[c_mask].mean(dim=0))

            result[b] = torch.stack(cluster_reps).mean(dim=0)

        return result


class GNNSurrogate:
    """
    GNN meta-evaluator wrapper with fit/predict interface.
    """

    def __init__(self, hidden_dim=128, n_layers=3, lr=0.001,
                 epochs=100, patience=15, knn_k=10, batch_size=32, seed=0,
                 fixed_node_dim=50):
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for GNN model")

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.knn_k = knn_k
        self.batch_size = batch_size
        self.seed = seed
        self.fixed_node_dim = fixed_node_dim  # truncate/pad all node features to this dim
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_graph_data(self, row):
        """Build a PyG Data object for a single clustering run."""
        ds_id = row["dataset_id"]
        algo = row["algo"]
        run_id = row["run_id"]

        proc_dir = PROCESSED_DIR / ds_id
        runs_dir = RUNS_DIR / ds_id

        X = np.load(proc_dir / "X.npy")
        labels = np.load(runs_dir / f"{algo}_{run_id}.npy")

        # Load kNN graph
        graph_path = proc_dir / f"A_knn_k{self.knn_k}.npz"
        if graph_path.exists():
            A = sparse.load_npz(graph_path)
        else:
            # Fallback: try other k values
            for k in [10, 15, 20]:
                alt_path = proc_dir / f"A_knn_k{k}.npz"
                if alt_path.exists():
                    A = sparse.load_npz(alt_path)
                    break
            else:
                raise FileNotFoundError(f"No kNN graph found for {ds_id}")

        edge_index = _sparse_to_edge_index(A)
        x = torch.tensor(X, dtype=torch.float32)
        partition = _encode_partition(labels, max_clusters=32)

        data = Data(
            x=x,
            edge_index=edge_index,
            partition=partition,
            partition_labels=torch.tensor(labels, dtype=torch.long),
            y=torch.tensor([row["ami"]], dtype=torch.float32),
        )
        return data

    def _build_dataset(self, df):
        """Build list of PyG Data objects from DataFrame."""
        data_list = []
        for _, row in df.iterrows():
            try:
                data = self._build_graph_data(row)
                data_list.append(data)
            except Exception as e:
                logger.debug(f"Skipping {row['dataset_id']}/{row['run_id']}: {e}")
                continue
        return data_list

    def _normalize_dim(self, data_list):
        """Truncate/pad node features to fixed_node_dim so batching works."""
        target_dim = self.fixed_node_dim
        for data in data_list:
            d = data.x.size(-1)
            if d > target_dim:
                data.x = data.x[:, :target_dim]
            elif d < target_dim:
                pad = torch.zeros(data.x.size(0), target_dim - d)
                data.x = torch.cat([data.x, pad], dim=-1)
        return data_list

    def fit(self, train_df, val_df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        logger.info("Building GNN training graphs...")
        train_data = self._build_dataset(train_df)
        val_data = self._build_dataset(val_df)

        if not train_data:
            logger.error("No training graphs could be built")
            return

        # Normalize all graphs to fixed_node_dim (truncate or pad)
        train_data = self._normalize_dim(train_data)
        val_data = self._normalize_dim(val_data)
        logger.info(f"Normalized node features to {self.fixed_node_dim} dimensions")

        input_dim = self.fixed_node_dim

        self.model = GNNSurrogateNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = np.inf
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            np.random.shuffle(train_data)
            train_loss = 0.0

            for i in range(0, len(train_data), self.batch_size):
                batch_data = train_data[i:i + self.batch_size]
                batch = Batch.from_data_list(batch_data).to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch)
                loss = F.mse_loss(pred, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_data)

            train_loss /= len(train_data)

            # Validate
            if val_data:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for i in range(0, len(val_data), self.batch_size):
                        batch_data = val_data[i:i + self.batch_size]
                        batch = Batch.from_data_list(batch_data).to(self.device)
                        pred = self.model(batch)
                        loss = F.mse_loss(pred, batch.y)
                        val_loss += loss.item() * len(batch_data)
                val_loss /= len(val_data)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                            f"val_loss={val_loss:.4f}" if val_data else "")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def predict(self, test_df):
        if self.model is None:
            return np.zeros(len(test_df))

        self.model.eval()
        test_data = self._build_dataset(test_df)

        if not test_data:
            return np.zeros(len(test_df))

        # Normalize to same fixed dimension as training
        test_data = self._normalize_dim(test_data)

        predictions = []
        with torch.no_grad():
            for i in range(0, len(test_data), self.batch_size):
                batch_data = test_data[i:i + self.batch_size]
                batch = Batch.from_data_list(batch_data).to(self.device)
                pred = self.model(batch)
                predictions.extend(pred.cpu().numpy().tolist())

        # Map back to original DataFrame order (some rows may have been skipped)
        # For now, return predictions for successfully built graphs, 0 for others
        result = np.zeros(len(test_df))
        built_idx = 0
        for i, (_, row) in enumerate(test_df.iterrows()):
            try:
                _ = self._build_graph_data(row)
                if built_idx < len(predictions):
                    result[i] = predictions[built_idx]
                    built_idx += 1
            except Exception:
                pass

        return result
