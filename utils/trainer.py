from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.profile import get_model_size, count_parameters
from torchinfo import summary
import matplotlib.pyplot as plt

from .earlystopper import EarlyStopper

class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader, config, loss_function, optimizer, device):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.loss_plot_file = config["loss_plot_file"]
        self.checkpoint_file = config["checkpoint_file"]
        self.prediction_plot_file = config["prediction_plot_file"]
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.val_steps = len(train_dataloader) // config["val_per_epoch"]
        self.earlystopper = EarlyStopper(limit=config["earlystop_limit"])
        self.train_history = {"train_loss" : [], "val_loss" : []}
        self.predictions = [] 
        self.ground_truth = []

    def train_step(self, data):
        X, y = self.get_data_and_targets(data)
        self.optimizer.zero_grad()
        pred = self.model(X)
        loss = self.loss_function(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        print(f"Training model...")
        for epoch in range(self.epochs):
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
                    pbar_str += f"ES: {self.earlystopper.counter:02}/{self.earlystopper.limit:02}"
                    pbar.set_description(pbar_str)

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

    def save_loss_plot(self):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].plot(self.train_history["train_loss"])
        axes[0].title.set_text("Training Loss")
        axes[0].set_yscale('log')
        axes[1].plot(self.train_history["val_loss"])
        axes[1].title.set_text("Validation Loss")
        axes[1].set_yscale('log')
        fig.tight_layout()
        plt.savefig(self.loss_plot_file, dpi=200)

    def evaluate(self, test_dataloader, store_predictions=True):
        print("Loading checkpoint...")
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        self.model.eval()
        print("Evaluating model on test data...")
        with torch.no_grad():
            test_losses = []
            for data in tqdm(test_dataloader): 
                X, y = self.get_data_and_targets(data) 
                preds = self.model(X)
                test_loss = self.loss_function(preds, y)
                test_losses.append(test_loss.item())

                # Save ground truth and predictions for later use
                gt_list, pred_list = self.batch_to_numpy(data, preds)
                self.ground_truth.extend(gt_list)
                self.predictions.extend(pred_list)

            self.ground_truth = np.array(self.ground_truth)
            self.predictions = np.array(self.predictions)
            mean_test_loss = np.mean(test_losses)
        print(f"Test Loss: {mean_test_loss:.4f}")

    def batch_to_numpy(self, data, preds):
        pass

    def save_prediction_plot(self, station_indices, from_index, length):
        preds = self.predictions[from_index:from_index+length, :]
        truth = self.ground_truth[from_index:from_index+length, :]

        fig, axes = plt.subplots(nrows=len(station_indices), ncols=1, figsize=(20,10))

        for i,j in enumerate(station_indices):
            axes[i].plot(truth[:, i], label="True", c="blue", alpha=0.5)
            axes[i].plot(preds[:, i], label="Predicted", c="red", alpha=0.5)
            #axes[i].title.set_text(f"Traffic station {station_names[i]}")
            axes[i].legend()
        fig.tight_layout()
        plt.savefig(self.prediction_plot_file, dpi=100)


class BaselineTrainer(Trainer):
    def get_data_and_targets(self, data):
        return data[0].to(self.device), data[1].to(self.device) 

    def print_model_size(self):
        summary(self.model, (self.batch_size, 98))

    def batch_to_numpy(self, data, preds):
        # Return ground truth and predictions for single mini-batch
        return data[1].tolist(), preds.detach().cpu().tolist()

class GNNTrainer(Trainer):
    def get_data_and_targets(self, data):
        return data.to(self.device), data.y.to(self.device)

    def print_model_size(self):
        print(f"Model size: {get_model_size(self.model)/2**20:.2f} MB")
        print(f"Parameters: {count_parameters(self.model)}")

    def batch_to_numpy(self, data, preds):
        # Return ground truth and predictions for single mini-batch
        # Note: we need to take extra care because PyG merges mini-batches into one large graph.
        for i in range(len(data.ptr) - 1):
            batch = data.x[data.ptr[i]:data.ptr[i+1]]
            print(batch.shape)
            # TODO: Change node features to only contain traffic volume data
            # by using MetaLayer so we can use a global model to take in the time/date data
            # Then make this function return predicted traffic volumes from the entire batch
