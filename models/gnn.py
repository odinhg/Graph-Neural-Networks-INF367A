import torch.nn as nn
from torch_geometric.nn import Sequential, GraphConv, MetaLayer
from torch_geometric.utils import scatter
from torch_geometric.nn.norm import BatchNorm

"""
To have global graph features (time and date in our case), we use the MetaLayer class of PyG.
This is more or less the reference implementation given in the PyG documentation.
"""

class EdgeModel(nn.Module):
    def __init__(self, node_features=1, edge_features=1, graph_features=3, hidden_dim=8):
        super().__init__()
        self.MLP = nn.Sequential(
                    nn.Linear(node_features * 2 + edge_features + graph_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, edge_features)
                )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        out = self.MLP(out)
        return out

class NodeModel(nn.Module):
    def __init__(self, node_features=1, edge_features=1, graph_features=3, hidden_dim=8):
        super().__init__()
        self.MLP1 = nn.Sequential(
                    nn.Linear(node_features + edge_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, node_features)
                )

        self.MLP2 = nn.Sequential(
                    nn.Linear(node_features * 2 + global_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, node_features)
                )

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.MLP1(out)
            out = scatter(out, col, dim=0, dim_size=x.size(0), reduce="mean")
            out = torch.cat([x, out, u[batch], dim=1)
            out = self.MLP2(out)
            return out

class GlobalModel(nn.Module):
    def __init__(self, node_features=1, edge_features=1, graph_features=3, hidden_dim=8):
        super().__init__()
        self.MLP = nn.Sequential(
                    nn.Linear(node_features + edge_features, graph_features),
                    nn.BatchNorm1d(graph_features),
                    nn.ReLU()
                )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1])
        x = scatter(x, batch, dim=0, reduce="mean")
        out = torch.cat([u, x], dim=1)
        out = self.MLP(out)
        return out

class GNNModel(nn.Module):
    def __init__(self, num_node_features=1):
        super().__init__()

    def forward(self, data):
        x = self.layers(data.x, data.edge_index, data.edge_weight)
        return x.squeeze(1)

# Old model
"""
class GNNModel(nn.Module):
    def __init__(self, num_node_features=4):
        super().__init__()
        self.layers = Sequential("x, edge_index, edge_weight", [
                        (GraphConv(num_node_features, 256), "x, edge_index, edge_weight -> x"),
                        BatchNorm(256),
                        nn.ReLU(inplace=True),
                        
                        (GraphConv(256, 256), "x, edge_index, edge_weight -> x"),
                        BatchNorm(256),
                        nn.ReLU(inplace=True),
                        
                        (GraphConv(256, 256), "x, edge_index, edge_weight -> x"),
                        BatchNorm(256),
                        nn.ReLU(inplace=True),
                        
                        (GraphConv(256, 256), "x, edge_index, edge_weight -> x"),
                        BatchNorm(256),
                        nn.ReLU(inplace=True),
                        
                        (GraphConv(256, 1), "x, edge_index, edge_weight -> x")
                    ])

    def forward(self, data):
        x = self.layers(data.x, data.edge_index, data.edge_weight)
        return x.squeeze(1)
"""
