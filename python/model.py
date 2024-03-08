import numpy as np
import torch_scatter
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, to_hetero, HeteroConv
from torchmetrics import F1Score, Accuracy


class HierarchicalHeteroGraphSage(torch.nn.Module):
    def __init__(self, edge_types, input_channels, hidden_channels, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.convs.append(
            HeteroConv(
                {
                    edge_type: SAGEConv(input_channels, hidden_channels, normalize=True, project=True)
                    for edge_type in edge_types
                }, aggr='mean')
        )
        for _ in range(num_layers-1):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(hidden_channels, hidden_channels)
                    for edge_type in edge_types
                }, aggr='mean')
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node,
                neighbor_mask_edge):
        for i, conv in enumerate(self.convs[:-1]):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.layer_norms[i](x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        return x_dict


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, metadata, dropout=0.5):
        super(GNN, self).__init__()
        self.gnn = HierarchicalHeteroGraphSage(metadata[1], input_dim, hidden_dim, num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge):
        x_dict = self.gnn(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)
        note = x_dict["note"]
        # Return the output
        out = self.mlp(note)
        return out


class SMOTE(object):
    """
    Minority Sampling with SMOTE.
    """
    def __init__(self, distance='custom', dims=512, k=2):
        super(SMOTE, self).__init__()
        self.newindex = 0
        self.k = k
        self.dims = dims
        self.distance_measure = distance

    def populate(self, N, i, nnarray, min_samples, k, device='cpu'):
        new_index = torch.arange(self.newindex, self.newindex + N, dtype=torch.int64, device=device)
        nn = torch.randint(0, k-1, (N, ), dtype=torch.int64, device=device)
        diff = min_samples[nnarray[nn]] - min_samples[i]
        gap = torch.rand((N, self.dims), dtype=torch.float32, device=device)
        self.synthetic_arr[new_index] = min_samples[i] + gap * diff
        self.newindex += N

    def k_neighbors(self, euclid_distance, k, device='cpu'):
        nearest_idx = torch.zeros((euclid_distance.shape[0], euclid_distance.shape[0]), dtype=torch.int64, device=device)

        idxs = torch.argsort(euclid_distance, dim=1)
        nearest_idx[:, :] = idxs

        return nearest_idx[:, 1:k+1]

    def find_k(self, X, k, device='cpu'):
        z = F.normalize(X, p=2, dim=1)
        distance = torch.mm(z, z.t())
        return self.k_neighbors(distance, k, device=device)

    def find_k_euc(self, X, k, device='cpu'):
        euclid_distance = torch.cdist(X, X)
        return self.k_neighbors(euclid_distance, k, device=device)

    def find_k_cos(self, X, k, device='cpu'):
        cosine_distance = F.cosine_similarity(X, X)
        return self.k_neighbors(cosine_distance, k, device=device)

    def generate(self, min_samples, N, k, device='cpu'):
        """
        Returns (N/100) * n_minority_samples synthetic minority samples.
        Parameters
        ----------
        min_samples : Numpy_array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples
        N : percetange of new synthetic samples:
            n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
        k : int. Number of nearest neighbours.
        Returns
        -------
        S : Synthetic samples. array,
            shape = [(N/100) * n_minority_samples, n_features].
        """
        T = min_samples.shape[0]
        self.synthetic_arr = torch.zeros(int(N / 100) * T, self.dims, dtype=torch.float32, device=device)
        N = int(N / 100)
        if self.distance_measure == 'euclidian':
            indices = self.find_k_euc(min_samples, k, device=device)
        elif self.distance_measure == 'cosine':
            indices = self.find_k_cos(min_samples, k, device=device)
        else:
            indices = self.find_k(min_samples, k, device=device)
        for i in range(indices.shape[0]):
            self.populate(N, i, indices[i], min_samples, k, device=device)
        self.newindex = 0
        return self.synthetic_arr

    def fit_generate(self, X, y):
        """
        Over-samples using SMOTE. Returns synthetic samples concatenated at the end of the original samples.
        Parameters
        ----------
        X : Numpy_array-like, shape = [n_samples, n_features]
            The input features
        y : Numpy_array-like, shape = [n_samples]
            The target labels.

        Returns
        -------
        X_resampled : Numpy_array, shape = [(n_samples + n_synthetic_samples), n_features]
            The array containing the original and synthetic samples.
        y_resampled : Numpy_array, shape = [(n_samples + n_synthetic_samples)]
            The corresponding labels of `X_resampled`.
        """
        # get occurence of each class
        occ = torch.eye(int(y.max() + 1), int(y.max() + 1), device=X.device)[y].sum(axis=0)
        # get the dominant class
        dominant_class = torch.argmax(occ)
        # get occurence of the dominant class
        n_occ = int(occ[dominant_class].item())
        for i in range(len(occ)):
            # For Mini-Batch Training exclude examples with less than k occurances in the mini banch.
            if i != dominant_class and occ[i] >= self.k:
                # calculate the amount of synthetic data to generate
                N = (n_occ - occ[i]) * 100 / occ[i]
                if N != 0:
                    candidates = X[y == i]
                    xs = self.generate(candidates, N, self.k, device=X.device)
                    X = torch.cat((X, xs))
                    ys = torch.ones(xs.shape[0], device=y.device) * i
                    y = torch.cat((y, ys))
        return X, y.long()


class CadenceGNNPytorch(nn.Module):
    def __init__(self, metadata, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super(CadenceGNNPytorch, self).__init__()
        self.gnn = GNN(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim // 2,
            num_layers=num_layers, metadata=metadata, dropout=dropout)

        hidden_dim = hidden_dim // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cad_clf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def encode(self, x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge, batch_note=None, onset_div=None):
        if batch_note is None:
            batch_note = torch.zeros((x_dict["note"].shape[0], ), device=x_dict["note"].device).long()
        if onset_div is None:
            # check if onset_div is present in x_dict
            if "onset_div" in x_dict:
                onset_div = x_dict.pop("onset_div")
        x = self.gnn(
            x_dict, edge_index_dict, neighbor_mask_node=neighbor_mask_node, neighbor_mask_edge=neighbor_mask_edge)
        if onset_div is not None:
            # This is a pooling operation that aggregates the features of notes with the same onset
            batch_note = batch_note[neighbor_mask_node["note"] == 0]
            onset_div = onset_div[neighbor_mask_node["note"] == 0]
            a = torch.stack((batch_note, onset_div), dim=-1)
            unique, cluster = torch.unique(a, return_inverse=True, dim=0, sorted=True)
            multiplicity = torch.ones_like(batch_note)
            # mean the features of notes with the same onset
            x = torch.zeros(unique.size(0), x.size(1), device=x.device).scatter_add(0, cluster.unsqueeze(1).expand(-1, x.size(1)), x)
            multiplicity = torch.zeros(unique.size(0), device=x.device, dtype=torch.long).scatter_add(0, cluster, multiplicity)
            x = x / multiplicity.unsqueeze(1)
            x = self.norm(x)
            x = self.pool_mlp(x)
            x = x[cluster]
        else:
            onset_index = edge_index_dict["note", "onset", "note"]
            x = torch_scatter.scatter_mean(x[onset_index[0]], onset_index[1], dim=0, out=x.clone())
            x = self.norm(x)
            x = self.pool_mlp(x)
        return x

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node=None, neighbor_mask_edge=None):
        if neighbor_mask_node is None:
            neighbor_mask_node = {k: torch.zeros((x_dict[k].shape[0], ), device=x_dict[k].device).long() for k in x_dict}
        if neighbor_mask_edge is None:
            neighbor_mask_edge = {k: torch.zeros((edge_index_dict[k].shape[-1], ), device=edge_index_dict[k].device).long() for k in edge_index_dict}
        x = self.encode(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)
        logits = self.cad_clf(x)
        return torch.softmax(logits, dim=-1)

    def clf(self, x):
        return self.cad_clf(x)


class CadencePLModel(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            metadata,
            dropout=0.5,
            lr=0.0001,
            weight_decay=5e-4,
            subgraph_size=500,
            reg_loss_weight=0.1,
    ):
        super(CadencePLModel, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.subgraph_size = subgraph_size
        self.reg_loss_weight = reg_loss_weight
        self.module = CadenceGNNPytorch(
            metadata=metadata, input_dim=input_dim, hidden_dim=hidden_dim,
            output_dim=output_dim, num_layers=num_layers, dropout=dropout)
        self.smote = SMOTE(dims=hidden_dim//2, distance='euclidian', k=3)
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.f1 = F1Score(num_classes=output_dim, task="multiclass", average="macro")
        self.acc = Accuracy(task="multiclass", num_classes=output_dim)

    def _common_step(self, batch, batch_idx):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        # out = model(batch.x_dict, batch.edge_index_dict)
        neighbor_mask_node = {k: batch[k].neighbor_mask for k in batch.node_types}
        neighbor_mask_edge = {k: batch[k].neighbor_mask for k in batch.edge_types}
        x = self.module.encode(
            x_dict, edge_index_dict,
            neighbor_mask_node=neighbor_mask_node, neighbor_mask_edge=neighbor_mask_edge,
            batch_note=batch["note"].batch, onset_div=batch["note"].onset_div
        )

        # Trim the labels to the target nodes (i.e. layer 0)
        y = batch["note"].y[neighbor_mask_node["note"] == 0]
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._common_step(batch, batch_idx)
        # feature loss verifies that the features are not too different
        feature_loss = x.pow(2).mean()
        x_over, y_over = self.smote.fit_generate(x, y)
        # Penalize when distance is too large between original and synthetic samples of the same class
        # Calculate Euclidean distance between synthetic and original samples
        for class_label in y_over.unique():
            mask = y_over == class_label
            x_over_class = x_over[mask]
            x_class = x[y == class_label]
            # Sample a few points from x_class and x_over to reduce computational cost
            if len(x_class) > 100:
                indices = np.random.choice(len(x_class), 100, replace=False)
                x_class = x_class[indices]
            if len(x_over_class) > 100:
                indices = np.random.choice(len(x_over_class), 100, replace=False)
                x_over_class = x_over_class[indices]
            distances = torch.cdist(x_over_class, x_class)
            min_distances, _ = torch.min(distances, dim=1)
            # Add penalty if distance is too large
            threshold = 1.0  # Set your own threshold
            penalties = torch.clamp(min_distances - threshold, min=0)
            feature_loss += penalties.mean()

        logits = self.module.clf(x_over)
        loss = self.loss(logits, y_over.long()) + (self.reg_loss_weight * feature_loss)
        self.log('train_loss', loss.item(), batch_size=len(y), prog_bar=True)
        self.log('train_f1', self.f1(logits, y_over.long()), prog_bar=True, batch_size=len(y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._common_step(batch, batch_idx)
        logits = self.module.cad_clf(x)
        num_classes = logits.shape[-1]
        # make the loss weighted by the number of labels
        num_labels = torch.bincount(y)
        # fix the shape of num_labels to match the shape of logits
        num_labels = torch.cat([num_labels, torch.zeros(num_classes - num_labels.shape[0], device=num_labels.device)])
        weight = 1 / (num_labels.float() + 1e-6) # avoid division by zero
        loss = F.cross_entropy(logits, y.long(), weight=weight)
        self.log('val_loss', loss.item(), batch_size=len(y), prog_bar=True)
        self.log('val_acc', self.acc(logits, y.long()), prog_bar=True, batch_size=len(y))
        self.log('val_f1', self.f1(logits, y.long()), prog_bar=True, batch_size=len(y))

    def test_step(self, batch, batch_idx):
        x, y = self._common_step(batch, batch_idx)
        logits = self.module.cad_clf(x)
        loss = self.loss(logits, y.long())
        self.log('test_loss', loss.item(), batch_size=len(y), prog_bar=True)
        self.log('test_acc', self.acc(logits, y.long()), prog_bar=True, batch_size=len(y))
        self.log('test_f1', self.f1(logits, y.long()), prog_bar=True, batch_size=len(y))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 40, 80], gamma=0.2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


