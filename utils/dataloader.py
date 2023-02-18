import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
#from PIL import Image
#from tqdm import tqdm
#from itertools import permutations
#from random import sample, choice

class TrafficVolumeDataSet(Dataset):
    def __init__(self, datafile):
        self.datafile = datafile
        self.df = pd.read_pickle(self.datafile)
        self.len = len(self.df) - 1 # Since there is no row after the last row.
        self.column_names = self.df.columns

    def __getitem__(self, index):
        # Return two consecutive rows of traffic data (and date / time)
        # Replace NaNs with -1 hoping that the model will learn that this means missing data
        data_now = self.df.iloc[index].replace(np.nan, -1)
        data_next = self.df.iloc[index + 1].replace(np.nan, -1)
        # Extracting month, weekday and hour
        month = getattr(data_now, "name").month
        weekday = getattr(data_now, "name").weekday()
        hour = getattr(data_now, "name").hour

        data_now = torch.Tensor(data_now.values)
        data_next = torch.Tensor(data_next.values)
        return ((month, weekday, hour, data_now), data_next)

    def __len__(self):
        return self.len

dataset = TrafficVolumeDataSet("../data/time_series_data.pkl")
print(len(dataset))
for i in range(796,800):
    print(dataset[i])

"""
def FlowCamDataLoader(class_names, image_size = (300, 300), val = 0.1, test = 0.2, batch_size = 32, split=True):
    dataset = FlowCamDataSet(class_names, image_size)
    num_workers = 8
    if not split:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return dataloader

    #Split into train and test data
    val_size = int(val * len(dataset))
    test_size = int(test * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(420))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Training data: {len(train_dataloader)*batch_size} images.")
    print(f"Validation data: {len(val_dataloader)*batch_size} images.")
    print(f"Test data: {len(test_dataloader)*batch_size} images.")

    return  {
            "train_dataloader"  : train_dataloader, 
            "val_dataloader"    : val_dataloader, 
            "test_dataloader"   : test_dataloader, 
            "train_dataset"     : train_dataset, 
            "val_dataset"       : val_dataset, 
            "test_dataset"      : test_dataset
            }
"""
