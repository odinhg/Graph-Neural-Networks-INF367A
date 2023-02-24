import networkx as nx
import numpy as np
from utils import TrafficVolumeGraphDataLoader
from config import *
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN, GCNConv, Sequential, GATConv
from torch.optim import Adam

class GCNModel(nn.Module):
    def __init__(self, num_node_features=4):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 128) 
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu() 
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu() 
        x = self.conv3(x, edge_index, edge_weight)
        return x


train_dl = TrafficVolumeGraphDataLoader(train_data_file, stations_data_file, stations_included_file, graph_file, batch_size=128, num_workers=4, shuffle=False, drop_last=True)

model = GCNModel()
model.to("cpu")
loss_function = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4) 
model.train()
for epoch in range(10):
    train_losses = []
    for i, data in enumerate(train_dl):
        data.to("cpu")
        optimizer.zero_grad()
        pred = model(data).squeeze(1)
        loss = loss_function(pred, data.y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if i % 10 == 0:
            print(f"Loss: {np.mean(train_losses):.4f}")
            train_losses = []
