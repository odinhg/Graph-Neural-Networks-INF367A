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
print("Building time series dataframe...")

time_series_data = pd.DataFrame(index=pd.date_range(first_timestamp, last_timestamp, freq="1H"), columns=unique_stations)

# This is painfully slow, but it works for now.
for row in tqdm(traffic_data.itertuples(), total=traffic_data.shape[0]):
    time_series_data.loc[getattr(row, "time_from"), getattr(row, "station_id")] = getattr(row, "volume")

# Drop columns with too many NaNs
time_series_data.dropna(thresh=max_number_of_nans, axis=1)

print(f"Time series contain {len(time_series_data)} hours of data from {len(time_series_data.columns)} stations. (Missing observations have value NaN)")
print(f"Saving time series data to {time_series_file}")
time_series_data.to_pickle(time_series_file) 
