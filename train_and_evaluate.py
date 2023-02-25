import torch
import torch.nn as nn

from config import *
from utils import choose_model, BaselineTrainer, GNNTrainer
from models import BaseLineModel, GNNModel
from utils import TrafficVolumeDataLoader, TrafficVolumeGraphDataLoader

config = choose_model(configs)

name = config["name"]
lr = config["lr"]
batch_size = config["batch_size"]

if name == "GNN": 
    model = GNNModel()
    train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, stations_data_file, stations_included_file, graph_file, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, stations_data_file, stations_included_file, graph_file, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, stations_data_file, stations_included_file, graph_file, batch_size=batch_size, num_workers=num_workers)
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    trainer = GNNTrainer(model, train_dataloader, val_dataloader, config, loss_function, optimizer, device)
elif name == "Baseline":
    model = BaseLineModel()
    train_dataloader = TrafficVolumeDataLoader(train_data_file, batch_size, num_workers, shuffle=True, drop_last=True)
    val_dataloader = TrafficVolumeDataLoader(val_data_file, batch_size, num_workers)
    test_dataloader = TrafficVolumeDataLoader(test_data_file, batch_size, num_workers)
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    trainer = BaselineTrainer(model, train_dataloader, val_dataloader, config, loss_function, optimizer, device)

trainer.print_model_size()
trainer.train()
trainer.save_loss_plot()
