import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split, Subset
from os.path import isfile 
import pandas as pd
import numpy as np

class TrafficVolumeDataSet(Dataset):
    def __init__(self, datafile):
        self.datafile = datafile
        assert isfile(self.datafile), f"Error: Data file {self.datafile} not found! Please run preprocess_data.py first."
        self.df = pd.read_pickle(self.datafile)
        self.len = len(self.df) - 1 # Since there is no row after the last row.
        self.column_names = self.df.columns
        print(f"Loaded datafile {self.datafile} with {self.len} rows...")

    def __getitem__(self, index):
        # Return two consecutive rows of traffic data (and date / time)
        # Replace NaNs with -1 
        data_now = self.df.iloc[index].replace(np.nan, -1)
        data_next = self.df.iloc[index + 1].replace(np.nan, -1)

        timestamp = getattr(data_now, "name")
        month = timestamp.month
        weekday = timestamp.weekday()
        hour = timestamp.hour
        datetime = torch.Tensor([month, weekday, hour])

        volumes_now = torch.Tensor(data_now.to_numpy(dtype=np.float32))
        target = torch.Tensor(data_next.to_numpy(dtype=np.float32))
        data_now = torch.cat((datetime, volumes_now))

        return (data_now, target, index) 

    def __len__(self):
        return self.len

def TrafficVolumeDataLoader(datafile, batch_size=32, num_workers=4, random_seed=0, shuffle=False, drop_last=False):
    dataset = TrafficVolumeDataSet(datafile)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return dataloader
