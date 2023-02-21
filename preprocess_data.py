import pandas as pd
from os.path import isfile
from tqdm import tqdm
from config import *

assert isfile(data_file_pkl), "Error: Pickled data not found. Please run unpack_data.py first."

traffic_data = pd.read_pickle(data_file_pkl)
station_df = pd.read_csv(stations_data_file)
unique_stations = traffic_data["station_id"].unique()

first_timestamp = traffic_data["time_from"].min()
last_timestamp = traffic_data["time_from"].max()

print(f"First timestamp: {first_timestamp}")
print(f"Last timestamp: {last_timestamp}")
print("Building time series dataframe... Please grab a coffee!")

time_series_data = pd.DataFrame(index=pd.date_range(first_timestamp, last_timestamp, freq="1H"), columns=unique_stations)

# This is painfully slow, but it works for now.
#for row in tqdm(traffic_data.itertuples(), total=traffic_data.shape[0]):
#    time_series_data.loc[getattr(row, "time_from"), getattr(row, "station_id")] = getattr(row, "volume")

for station_id in tqdm(unique_stations):
    #station_id = getattr(station, "id")
    #timestamps = traffic_data.loc[traffic_data["station_id"]==station_id, "time_from"]
    #volumes = traffic_data.loc[traffic_data["station_id"]==station_id, "volume"]
    df = traffic_data.loc[traffic_data["station_id"]==station_id, ["volume", "time_from"]]
    timestamps = df["time_from"]
    volumes = df["volume"]
    volumes.index = timestamps
    time_series_data.loc[timestamps, station_id] = volumes

# Drop stations with too many NaNs / too few observations
print(f"Dropping stations with too few observations (<{min_number_of_observations})...")
time_series_data.dropna(thresh=min_number_of_observations, axis=1, inplace=True)

# Replace all NaN rows with data from the previous  and next row (taking the average)
#print("Filling all NaN rows...")
#time_series_data.loc[time_series_data.isnull().all(axis=1), :] = (time_series_data.ffill() + time_series_data.bfill()) / 2

# Normalize dataset using statistics from training data (to prevent data leakage) 
# time_series_data = (time_series_data - time_series_data.mean()) / time_series_data.std()

print(f"Time series contain {len(time_series_data)} hours of data from {len(time_series_data.columns)} stations. (Missing observations have value NaN)")
time_series_data.to_pickle(time_series_file) 
print(f"Time series data saved to {time_series_file}.")
