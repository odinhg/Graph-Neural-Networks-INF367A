import torch.nn as nn
from torch_geometric.nn import Sequential, GraphConv
from torch_geometric.nn.norm import BatchNorm

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
