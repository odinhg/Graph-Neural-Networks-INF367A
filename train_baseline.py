from config import *
from models import BaseLineModel
from utils import TrafficVolumeDataLoader, EarlyStopper
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam

lr = config_baseline["lr"]
batch_size = config_baseline["batch_size"]
epochs = config_baseline["epochs"]
checkpoint_file = config_baseline["checkpoint_file"]

train_dataloader, val_dataloader, test_dataloader = TrafficVolumeDataLoader(time_series_file, val_fraction, test_fraction, batch_size, num_workers, 0, normalize_data)

val_steps = len(train_dataloader) // validations_per_epoch 

model = BaseLineModel()
loss_function = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=lr) 
earlystopper = EarlyStopper() 

model.to(device)

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
            train_history["train_loss"].append(mean_train_loss)
            train_history["val_loss"].append(mean_val_loss)
            if mean_val_loss < np.min(train_history["val_loss"]):
                torch.save(model.state_dict(), checkpoint_file)
            if earlystopper(mean_val_loss):
                print(f"Early stopped at epoch {epoch}!")
                exit()
            model.train()
            pbar_str = f"Epoch {epoch:02}/{epochs:02} | Loss (Train): {mean_train_loss:.3f} | Loss (Val): {mean_val_loss:.3f}"
            pbar.set_description(pbar_str)

# Save loss plot
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(train_history["train_loss"])
axes[0].title.set_text("MSE Training Loss")
axes[0].set_yscale('log')
axes[1].plot(train_history["val_loss"])
axes[1].title.set_text("MSE Validation Loss")
axes[1].set_yscale('log')
fig.tight_layout()
plt.savefig("figs/baseline_training_plot.png")
