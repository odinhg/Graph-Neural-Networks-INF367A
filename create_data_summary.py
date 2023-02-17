import pandas as pd
from os.path import isfile
from tqdm import tqdm
from config import *

assert isfile(data_file_pkl), "Error: Pickled data not found. Please run unpack_data.py first."

traffic_data = pd.read_pickle(data_file_pkl)

station_df = pd.read_csv(stations_data_file)

total_samples = traffic_data.shape[0]
unique_stations = traffic_data["station_id"].unique()

print(f"Total of {total_samples} observations from {len(unique_stations)} stations.")

stations = []
table = []
table.append("|ID|NAME|LAT|LON|First observation|Last observation|No. observations|Min volume|Max volume|Mean volume|Standard deviation|\n")
table.append("|-|-|-|-|-|-|-|-|-|-|-|\n")

print("Calculating statistics...")
for station in tqdm(unique_stations):
    station_info = station_df[station_df["id"] == station]
    station_data = traffic_data[traffic_data["station_id"] == station]
    volume = station_data["volume"]
    timestamps = station_data["time_from"]
    name = station_info["name"].item()
    lat = station_info["latitude"].item()
    lon = station_info["longitude"].item()
    table.append(f"|**{station}**|{name}|{lat}|{lon}|{timestamps.min()}|{timestamps.max()}|{volume.size}|{volume.min()}|{volume.max()}|{volume.mean():.3f}|{volume.std():.3f}|\n")
    stations.append({"id" : station, "volume" : volume, "timestamps" : timestamps, "observations" : len(volume), "name" : name, "coords" : (lat, lon)})

with open(summary_table_file, "w") as f:
    f.writelines(table)
print(f"Summary table written to {summary_table_file}")
