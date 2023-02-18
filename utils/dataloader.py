import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
from os.path import isfile 
import pandas as pd
import numpy as np

class TrafficVolumeDataSet(Dataset):
    def __init__(self, datafile):
        self.datafile = datafile
        assert isfile(self.datafile), f"Error: Data file {self.datafile} not found! Please run preprocess_data.py first."
        print(f"Loading traffic data from {self.datafile}...")
        self.df = pd.read_pickle(self.datafile)
        self.len = len(self.df) - 1 # Since there is no row after the last row.
        self.column_names = self.df.columns

    def __getitem__(self, index):
        # Return two consecutive rows of traffic data (and date / time)
        # Replace NaNs with -1 
        data_now = self.df.iloc[index].replace(np.nan, -1)
        data_next = self.df.iloc[index + 1].replace(np.nan, -1)

        month = getattr(data_now, "name").month
        weekday = getattr(data_now, "name").weekday()
        hour = getattr(data_now, "name").hour

        data_now = torch.Tensor(data_now.values)
        data_next = torch.Tensor(data_next.values)
        return ((month, weekday, hour, data_now), data_next)

    def __len__(self):
        return self.len

def mean_std(subset):
    indices = subset.indices
    dataset = subset.dataset.df.iloc[indices, :]
    return (dataset.mean(), dataset.std())

def normalize(subset, mean, std):
    indices = subset.indices
    subset.dataset.df.iloc[indices, :] = (subset.dataset.df.iloc[indices, :] - mean) / std
    return subset

def TrafficVolumeDataLoader(datafile, val=0.1, test=0.1, batch_size=32, num_workers=4, random_seed=0, normalize_data=True):
    dataset = TrafficVolumeDataSet(datafile)
    
    # Split the dataset into training, validation and testing data
    val_size = int(val * len(dataset))
    test_size = int(test * len(dataset))
    train_size = len(dataset) - val_size - test_size
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # Normalize dataset with statistics computed on the training daga (to prevent data leakage)
    if normalize_data:
        print("Normalizing datasets...")
        mean, std = mean_std(train_dataset)
        train_dataset = normalize(train_dataset, mean, std)
        val_dataset = normalize(val_dataset, mean, std)
        test_dataset = normalize(test_dataset, mean, std)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Training data: {len(train_dataloader)*batch_size} rows.")
    print(f"Validation data: {len(val_dataloader)*batch_size} rows.")
    print(f"Test data: {len(test_dataloader)*batch_size} rows.")

    return (train_dataloader, val_dataloader, test_dataloader)

TrafficVolumeDataLoader("../data/time_series_data.pkl")
