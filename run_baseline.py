from config import *
from models import BaseLineModel
from utils import TrafficVolumeDataLoader, EarlyStopper

from os.path import isfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torchinfo import summary

if "cuda" in device and not torch.cuda.is_available():
    print(f"Warning: Device set to {device} in config but no GPU available. Using CPU instead.")
    device = "cpu"
    num_workers = 4

lr = config_baseline["lr"]
batch_size = config_baseline["batch_size"]
epochs = config_baseline["epochs"]
checkpoint_file = config_baseline["checkpoint_file"]

print(f"Model name:\t{config_baseline['name']}")
print(f"Learning rate:\t{config_baseline['lr']}")
print(f"Batch size:\t{config_baseline['batch_size']}")

train_dataloader = TrafficVolumeDataLoader(train_data_file, batch_size, num_workers, random_seed=0, shuffle=True, drop_last=True)
val_dataloader = TrafficVolumeDataLoader(val_data_file, batch_size, num_workers, random_seed=0, shuffle=False, drop_last=False)
test_dataloader = TrafficVolumeDataLoader(test_data_file, batch_size, num_workers, random_seed=0, shuffle=False, drop_last=False)

val_steps = len(train_dataloader) // validations_per_epoch 

model = BaseLineModel()
loss_function = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=lr) 
earlystopper = EarlyStopper(limit=config_baseline["earlystop_limit"])

summary(model, (batch_size, 98))
model.to(device)

if not isfile(checkpoint_file):
    train_history = {"train_loss" : [], "val_loss" : []}

    for epoch in range(epochs):
        train_losses = []
        for i, data in enumerate((pbar := tqdm(train_dataloader))):
            data_now, target, index = data[0].to(device), data[1].to(device), data[2]
            
            # Train step
            optimizer.zero_grad()
            pred = model(data_now)
            loss = loss_function(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if i % val_steps == val_steps - 1:
                mean_train_loss = np.mean(train_losses)
                train_losses = []
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for data in val_dataloader:
                        data_now, target = data[0].to(device), data[1].to(device)
                        preds = model(data_now)
                        val_loss = loss_function(preds, target).item()
                        val_losses.append(val_loss)
                mean_val_loss = np.mean(val_losses)
                if mean_val_loss <= np.min(train_history["val_loss"], initial=np.inf):
                    torch.save(model.state_dict(), checkpoint_file)
                if earlystopper(mean_val_loss):
                    print(f"Early stopped at epoch {epoch}!")
                    break
                model.train()
                train_history["train_loss"].append(mean_train_loss)
                train_history["val_loss"].append(mean_val_loss)
                pbar_str = f"Epoch {epoch:02}/{epochs:02} | Loss (Train): {mean_train_loss:.4f} | Loss (Val): {mean_val_loss:.4f} | ES: {earlystopper.counter:02}/{earlystopper.limit:02}"
                pbar.set_description(pbar_str)
        else:
            continue
        break   # Break on early stop

    # Save loss plot
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(train_history["train_loss"])
    axes[0].title.set_text("Training L1-Loss")
    axes[0].set_yscale('log')
    axes[1].plot(train_history["val_loss"])
    axes[1].title.set_text("Validation L1-Loss")
    axes[1].set_yscale('log')
    fig.tight_layout()
    plt.savefig("figs/baseline_training_plot.png", dpi=200)

# Evaluate on test data
print("Loading checkpoint...")
model.load_state_dict(torch.load(checkpoint_file))
model.eval()

print("Evaluating model on test data.")

# For plotting predicted vs ground truth values for some stations
station_indices = [9,93,17] # Indices of stations to plot
actual_volumes = [[]]*len(station_indices)
predicted_volumes = [[]]*len(station_indices)
idx_start = 5000 
idx_stop = 5336

df = pd.read_pickle(test_data_file)
station_names = df.columns[station_indices]
timestamps = df.iloc[idx_start : idx_stop].index.values

with torch.no_grad():
    test_losses = []
    for data in tqdm(test_dataloader): 
        data_now, target, index = data[0].to(device), data[1].to(device), data[2]
        preds = model(data_now)
        test_loss = loss_function(preds, target)
        test_losses.append(test_loss.item())

        for i, j in enumerate(station_indices):
            actual_volumes[i] = np.concatenate((actual_volumes[i], data_now.cpu().detach().numpy()[:, 3+j]))
            predicted_volumes[i] = np.concatenate((predicted_volumes[i], preds.cpu().detach().numpy()[:, j]))

    mean_test_loss = np.mean(test_losses)

print(f"Test L1 Loss: {mean_test_loss:.4f}")

fig, axes = plt.subplots(nrows=len(station_indices), ncols=1, figsize=(20,10))
for i,j in enumerate(station_indices):
    axes[i].plot(timestamps, actual_volumes[i][idx_start:idx_stop], label="True", c="blue", alpha=0.5)
    axes[i].plot(timestamps, predicted_volumes[i][idx_start:idx_stop], label="Predicted", c="red", alpha=0.5)
    axes[i].title.set_text(f"Traffic station {station_names[i]}")
    axes[i].legend()
fig.tight_layout()

plt.savefig(config_baseline["prediction_plot_file"], dpi=100)

