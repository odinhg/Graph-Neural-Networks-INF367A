from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.profile import get_model_size, count_parameters
from torchinfo import summary
import matplotlib.pyplot as plt
from pathlib import Path
from os.path import join
import time

from .earlystopper import EarlyStopper

class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.loss_plot_file = config["loss_plot_file"]
        self.checkpoint_file = config["checkpoint_file"]
        self.prediction_plot_dir = config["prediction_plot_dir"]
        self.loss_function = loss_function
        self.device = device
        self.val_steps = len(train_dataloader) // config["val_per_epoch"]
        self.earlystopper = EarlyStopper(limit=config["earlystop_limit"])
        self.train_history = {"train_loss" : [], "val_loss" : []}
        self.test_results = {"predictions" : [], "ground_truth" : []}
        self.epoch_times = []

    def train_step(self, data):
        X, y = self.get_data_and_targets(data)
        self.optimizer.zero_grad()
        pred = self.model(X)
        loss = self.loss_function(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, optimizer, scheduler=None):
        print(f"Training model...")
        self.optimizer = optimizer
        self.scheduler = scheduler
        for epoch in range(self.epochs):
            time_start = time.time()
            train_losses = []
            for i, data in enumerate((pbar := tqdm(self.train_dataloader))):
                loss = self.train_step(data)
                train_losses.append(loss)

                if i % self.val_steps == self.val_steps - 1:
                    mean_train_loss = np.mean(train_losses)
                    train_losses = []
                    mean_val_loss = self.validate()
                    self.train_history["train_loss"].append(mean_train_loss)
                    self.train_history["val_loss"].append(mean_val_loss)

                    if mean_val_loss <= np.min(self.train_history["val_loss"], initial=np.inf):
                        torch.save(self.model.state_dict(), self.checkpoint_file)

                    if self.earlystopper(mean_val_loss):
                        print(f"Early stopped at epoch {epoch}!")
                        return

                    pbar_str = f"Epoch {epoch:02}/{self.epochs:02} | "
                    pbar_str += f"Loss (Train): {mean_train_loss:.4f} | "
                    pbar_str += f"Loss (Val): {mean_val_loss:.4f} | "
                    pbar_str += f"ES: {self.earlystopper.counter:02}/{self.earlystopper.limit:02} | "
                    if self.scheduler:
                        pbar_str += f"LR: {self.scheduler.get_last_lr()[0]} |"
                    pbar.set_description(pbar_str)
            if self.scheduler:
                self.scheduler.step()
            self.epoch_times.append(time.time() - time_start)

    def val_step(self, data):
        X, y = self.get_data_and_targets(data)
        preds = self.model(X)
        loss = self.loss_function(preds, y)
        return loss.item()

    def validate(self):
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for data in self.val_dataloader:
                val_loss = self.val_step(data)
                val_losses.append(val_loss)
        mean_val_loss = np.mean(val_losses)
        self.model.train()
        return mean_val_loss

    def get_data_and_targets(self, data):
        pass

    def print_model_size(self):
        pass

    def save_loss_plot(self, use_log_scale=True):
        print(f"Saving loss plot to {self.loss_plot_file}...")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.train_history["train_loss"], label="Training")
        ax.plot(self.train_history["val_loss"], label="Validation")
        if use_log_scale:
            ax.set_yscale('log')
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean loss value")
        ax.legend(loc="upper right")
        fig.suptitle("Loss")
        fig.tight_layout()
        plt.savefig(self.loss_plot_file, dpi=100)

    def summarize_training(self):
        self.save_loss_plot()
        print(f"Total training time: {np.sum(self.epoch_times):.2f}s.")
        print(f"Mean epoch time: {np.mean(self.epoch_times):.2f}s.")

    def evaluate(self):
        print("Loading checkpoint...")
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        self.model.eval()
        print("Evaluating model on test data...")
        with torch.no_grad():
            test_losses = []        # L1 losses
            test_losses_rmse = []    # Square root of L2 losses
            for data in tqdm(self.test_dataloader): 
                X, y = self.get_data_and_targets(data) 
                preds = self.model(X)
                test_loss = self.loss_function(preds, y)
                test_losses.append(test_loss.item())
                test_loss_rmse = np.sqrt(torch.nn.functional.mse_loss(preds, y).detach().cpu())
                test_losses_rmse.append(test_loss_rmse)

                # Save ground truth and predictions for later use
                gt_list, pred_list = self.batch_to_list(data, preds)
                self.test_results["ground_truth"].extend(gt_list)
                self.test_results["predictions"].extend(pred_list)

        for key in self.test_results.keys():
            self.test_results[key] = np.asarray(self.test_results[key])

        mean_test_loss = np.mean(test_losses)
        mean_test_loss_rmse = np.mean(test_losses_rmse)
        print(f"Test Loss (L1/MAE): {mean_test_loss:.4f}")
        print(f"Test Loss (L2/RMSE): {mean_test_loss_rmse:.4f}")

    def batch_to_list(self, data, preds):
        pass

    def save_prediction_plot(self, from_index, length):
        # Plots predictions and ground truth for stations with indices in station_indices
        print(f"Saving prediction plots to directory {self.prediction_plot_dir}...")
        preds = self.test_results["predictions"][from_index:from_index+length, :]
        truth = self.test_results["ground_truth"][from_index:from_index+length, :]
        timestamps = self.test_dataloader.dataset.timestamps[from_index:from_index+length]

        # Which traffic stations to save prediction plots for
        station_indices = [1, 20, 36, 56, 62, 65, 71, 74, 79, 84]
        station_ids = self.test_dataloader.dataset.column_names[station_indices]

        Path(self.prediction_plot_dir).mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(12,6))
        for i, station_id in zip(station_indices, station_ids): 
            ax.plot(timestamps, truth[:, i], label="Ground truth", c="blue", alpha=0.7)
            ax.plot(timestamps, preds[:, i], label="Predicted", c="red", alpha=0.7)
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Traffic volume")
            ax.legend(loc="upper right")
            fig.suptitle(f"Traffic station {station_id}")
            fig.tight_layout()
            filename = join(self.prediction_plot_dir, f"{i:03}_{station_id}.png")
            plt.savefig(filename, dpi=80)
            plt.cla()

class BaselineTrainer(Trainer):
    def get_data_and_targets(self, data):
        return data[0].to(self.device), data[1].to(self.device) 

    def print_model_size(self):
        summary(self.model, (self.batch_size, 98))

    def batch_to_list(self, data, preds):
        # Return ground truth and predictions for single mini-batch
        return data[1].tolist(), preds.detach().cpu().tolist()

class GNNTrainer(Trainer):
    def get_data_and_targets(self, data):
        return data.to(self.device), data.y.to(self.device)

    def print_model_size(self):
        print(f"Model size: {get_model_size(self.model)/2**20:.2f} MB")
        print(f"Parameters: {count_parameters(self.model)}")

    def batch_to_list(self, data, preds):
        # Return ground truth and predictions for single mini-batch
        # Note: we need to take extra care because PyG merges mini-batches into one large graph.
        gt_list = [data.y[data.ptr[j]:data.ptr[j+1]].tolist() for j in range(len(data.ptr) - 1)]
        pred_list = [preds[data.ptr[j]:data.ptr[j+1]].tolist() for j in range(len(data.ptr) - 1)]
        return gt_list, pred_list
