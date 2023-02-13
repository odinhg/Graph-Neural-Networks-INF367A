"""
Pickle or unpickle data file with compression.
"""

import pandas as pd
from os.path import isfile

file = "traffic_data"
compression_method = "gzip"

filename_csv = file + ".csv"
filename_zip = file + ".zip"

if not isfile(filename_zip):
    print(f"Compressing {filename_csv} => {filename_zip}")
    data = pd.read_csv(filename_csv)
    data.to_pickle(filename_zip, compression=compression_method)
elif not isfile(filename_csv):
    print(f"Extracing {filename_zip} => {filename_csv}")
    data = pd.read_pickle(filename_zip, compression=compression_method)
    data.to_csv(filename_csv)
else:
    print("Nothing to do.")

