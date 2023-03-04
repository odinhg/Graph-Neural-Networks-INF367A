import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, MetaLayer
from torch_geometric.utils import scatter
from torch_geometric.nn.norm import BatchNorm

class EdgeModel(nn.Module):
    def __init__(self, node_features, edge_features, graph_features, hidden_dim):
        super().__init__()
        self.MLP = nn.Sequential(
                    nn.Linear(node_features[0] * 2 + edge_features[0] + graph_features[0], hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, edge_features[1])
                )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        out = self.MLP(out)
        return out 

class NodeModel(nn.Module):
    def __init__(self, node_features, edge_features, graph_features, hidden_dim):
        super().__init__()
        self.MLP1 = nn.Sequential(
                    nn.Linear(node_features[0] + edge_features[1] + graph_features[0], hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, node_features[1])
                )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        e_aggr = scatter(edge_attr, col, dim=0, reduce="mean")
        out = torch.cat([x, e_aggr, u[batch]], dim=1) 
        out = self.MLP1(out)
        return out

class GlobalModel(nn.Module):
    def __init__(self, node_features, edge_features, graph_features, hidden_dim):
        super().__init__()
        self.MLP = nn.Sequential(
                    nn.Linear(node_features[1] + edge_features[1] + graph_features[0], hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, graph_features[1])
                )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        x_aggr = scatter(x, batch, dim=0, reduce="mean")
        e_aggr = scatter(edge_attr, batch[col], dim=0, reduce="mean")
        out = torch.cat([u, x_aggr, e_aggr], dim=1)
        out = self.MLP(out)
        return out

class GNNLayer(nn.Module):
    def __init__(self, node_features=(1, 1), edge_features=(1, 1), graph_features=(3, 3), hidden_dims=(32, 32, 32), use_edge_model=True):
        """
        Implementation of the GN block proposed in the paper Relational inductive biases, 
        
        deep learning, and graph networks (https://arxiv.org/abs/1806.01261).
        node_features   : (node features in, node features out)
        edge_features   : (edge features in, edge features_out)
        graph_features  : (graph features in, graph features_out)
        hidden_dims     : how many dimensions to use in the hidden layer of the MLPs of the
                          edge model, node model and the global model, respectively.
        use_edge_model  : If true, edge features are updated.
        """
        super().__init__()
        if use_edge_model:
            self.edge_model = EdgeModel(node_features, edge_features, graph_features, hidden_dims[0])
        else:
            self.edge_model = None
        self.node_model = NodeModel(node_features, edge_features, graph_features, hidden_dims[1])
        self.global_model = GlobalModel(node_features, edge_features, graph_features, hidden_dims[2])
        self.layers = MetaLayer(self.edge_model, self.node_model, self.global_model)

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = self.layers(x, edge_index, edge_attr, u, batch)
        return out

class GNNModel(nn.Module):
    def __init__(self, use_edge_model=True):
        super().__init__()
        node_feature_sizes = [1, 64, 64, 32, 16, 1]
        edge_feature_sizes = [1, 1, 1, 1, 1, 1]
        graph_feature_sizes = [3, 32, 32, 16, 8, 1]
        mlp_hidden_dims = (64, 128, 64) 

        if not use_edge_model:
            edge_feature_sizes = [1] * len(node_feature_sizes)

        assert len(node_feature_sizes) == len(edge_feature_sizes) == len(graph_feature_sizes), "Feature sizes does not match!"

        layers = []
        layer_header= "x, edge_index, edge_attr, u, batch -> x, edge_attr, u"

        # Add GNN layers to model
        for j in range(len(node_feature_sizes) - 1):
            layer = GNNLayer(node_feature_sizes[j:j+2], edge_feature_sizes[j:j+2], graph_feature_sizes[j:j+2], mlp_hidden_dims, use_edge_model)
            layers.append((layer, layer_header))
        self.layers = Sequential("x, edge_index, edge_attr, u, batch", layers)

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch
        x, edge_attr, u = self.layers(x, edge_index, edge_attr, u, batch)
        return x.view(-1)
