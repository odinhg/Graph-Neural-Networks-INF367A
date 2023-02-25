import torch
import torch.nn as nn
from torch_geometric.seed import seed_everything
from os.path import isfile

from config import *
from utils import choose_model, BaselineTrainer, GNNTrainer
from models import BaseLineModel, GNNModel
from utils import TrafficVolumeDataLoader, TrafficVolumeGraphDataLoader, create_edge_index_and_features

seed_everything(0)

config = choose_model(configs)
name = config["name"]
lr = config["lr"]
batch_size = config["batch_size"]

if name == "GNN": 
    model = GNNModel()
    edge_index, edge_weight = create_edge_index_and_features(stations_included_file, graph_file, stations_data_file)
    train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True)
    val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers)
    test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers)
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    trainer = GNNTrainer(model, train_dataloader, val_dataloader, config, loss_function, optimizer, device)
elif name == "Baseline":
    model = BaseLineModel()
    train_dataloader = TrafficVolumeDataLoader(train_data_file, batch_size, num_workers, shuffle=True)
    val_dataloader = TrafficVolumeDataLoader(val_data_file, batch_size, num_workers)
    test_dataloader = TrafficVolumeDataLoader(test_data_file, batch_size, num_workers)
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    trainer = BaselineTrainer(model, train_dataloader, val_dataloader, config, loss_function, optimizer, device)

trainer.print_model_size()

if not isfile(config["checkpoint_file"]):
    trainer.train()
    trainer.save_loss_plot()
else:
    print("Checkpoint file exists. Please delete checkpoint file to re-train model.")

trainer.evaluate(test_dataloader)
