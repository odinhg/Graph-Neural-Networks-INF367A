import torch
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
                    nn.Linear(node_features + edge_features + graph_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, node_features)
                )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        e_aggr = scatter(edge_attr, col, dim=0, reduce="mean")
        out = torch.cat([x, e_aggr, u[batch]], dim=1) 
        out = self.MLP1(out)
        return out

class GlobalModel(nn.Module):
    def __init__(self, node_features=1, edge_features=1, graph_features=3, hidden_dim=8):
        super().__init__()
        self.MLP = nn.Sequential(
                    nn.Linear(node_features + edge_features + graph_features, graph_features),
                    nn.BatchNorm1d(graph_features),
                    nn.ReLU()
                )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        x_aggr = scatter(x, batch, dim=0, reduce="mean")
        e_aggr = scatter(edge_attr, batch[col], dim=0, reduce="mean")
        out = torch.cat([u, x_aggr, e_aggr], dim=1)
        out = self.MLP(out)
        return out

class GNNLayer(nn.Module):
    def __init__(self, node_features=1, edge_features=1, graph_features=3, hidden_dim=128):
        super().__init__()
        self.edge_model = EdgeModel(node_features, edge_features, graph_features, hidden_dim)
        self.node_model = NodeModel(node_features, edge_features, graph_features, hidden_dim)
        self.global_model = GlobalModel(node_features, edge_features, graph_features, hidden_dim)
        self.layers = MetaLayer(self.edge_model, self.node_model, self.global_model)

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = self.layers(x, edge_index, edge_attr, u, batch)
        return out

class GNNModel(nn.Module):
    def __init__(self, node_features=1, edge_features=1, graph_features=3):
        super().__init__()
        # TODO:
        # - Make this pretty
        # - Have more features in hidden layers
        self.layer1 = GNNLayer(node_features, edge_features, graph_features)
        self.layer2 = GNNLayer(node_features, edge_features, graph_features)
        self.layer3 = GNNLayer(node_features, edge_features, graph_features)
        self.layer4 = GNNLayer(node_features, edge_features, graph_features)
        self.layer5 = GNNLayer(node_features, edge_features, graph_features)
        self.layer6 = GNNLayer(node_features, edge_features, graph_features)
        self.layer7 = GNNLayer(node_features, edge_features, graph_features)
        self.layer8 = GNNLayer(node_features, edge_features, graph_features)
        self.layer9 = GNNLayer(node_features, edge_features, graph_features)
        self.layer10 = GNNLayer(node_features, edge_features, graph_features)
        self.layer11 = GNNLayer(node_features, edge_features, graph_features)
        self.layer12 = GNNLayer(node_features, edge_features, graph_features)
    
    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch
        x, edge_attr, u = self.layer1(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer2(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer3(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer4(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer5(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer6(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer7(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer8(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer9(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer10(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer11(x, edge_index, edge_attr, u, batch) 
        x, edge_attr, u = self.layer12(x, edge_index, edge_attr, u, batch) 
        return x.view(-1)


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
