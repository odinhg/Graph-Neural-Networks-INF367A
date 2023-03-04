import torch
import torch.nn as nn
from torch_geometric.seed import seed_everything
from os.path import isfile

from config import *
from utils import choose_model, BaselineTrainer, GNNTrainer
from models import BaseLineModel, GNNModel
from utils import TrafficVolumeDataLoader, TrafficVolumeGraphDataLoader, create_edge_index_and_features

if __name__ == "__main__":  
    seed_everything(0)

    config = choose_model(configs)
    name = config["name"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    if name == "GNN": 
        # Graph NN with edge, node and graph models
        model = GNNModel()
        edge_index, edge_weight = create_edge_index_and_features(stations_included_file, graph_file, stations_data_file)
        train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True)
        val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers)
        test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers)
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, optimizer, device)
    elif name == "GNN_NE":
        # Graph NN with node and graph model
        model = GNNModel(use_edge_model=False)
        edge_index, edge_weight = create_edge_index_and_features(stations_included_file, graph_file, stations_data_file)
        train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True)
        val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers)
        test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers)
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, optimizer, device)
    elif name == "Baseline":
        # Baseline fully connected NN model
        model = BaseLineModel()
        train_dataloader = TrafficVolumeDataLoader(train_data_file, batch_size, num_workers, shuffle=True)
        val_dataloader = TrafficVolumeDataLoader(val_data_file, batch_size, num_workers)
        test_dataloader = TrafficVolumeDataLoader(test_data_file, batch_size, num_workers)
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        trainer = BaselineTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, optimizer, device)

    trainer.print_model_size()

    if not isfile(config["checkpoint_file"]):
        trainer.train()
        trainer.save_loss_plot()
    else:
        print("Checkpoint file exists. Please delete checkpoint file to re-train model.")

    # Evaluate model on test data and compute test loss
    trainer.evaluate()

    # Make some prediction and save plot
    from_index = 500 
    length = 500
    trainer.save_prediction_plot(from_index, length)
