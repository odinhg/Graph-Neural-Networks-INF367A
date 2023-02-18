"""
This script unpacks the zipped datafile into a pickled dataframe.
"""
import pandas as pd
from os.path import isfile
from config import *

# Unpack compressed data file if csv file is not found
if not isfile(data_file_pkl):
    print(f"Extracing {data_file_zip}...")
    data = pd.read_pickle(data_file_zip, compression="gzip")
    # Convert strings to datetime datatype
    time_format = "%Y-%m-%d %H:%M:%S%z"
    data["time_from"] = pd.to_datetime(data["time_from"], format=time_format)
    data["time_to"] = pd.to_datetime(data["time_to"], format=time_format)
    print(f"Saving pickled dataframe to {data_file_pkl}...")
    data.to_pickle(data_file_pkl)

