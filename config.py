from os.path import join

# Paths
data_path = "data"
figs_path = "figs"
docs_path = "docs"

# Filenames
data_file_zip = join(data_path, "traffic_data.zip")
data_file_pkl = join(data_path, "traffic_data.pkl")
stations_data_file = join(data_path, "traffic_stations.csv")
summary_table_file = join(docs_path, "data_summary_table.md")
time_series_file = join(data_path, "time_series_data.pkl")

# Drop stations having too few observations
max_number_of_nans = 68000 

# How much data to use for training, validation and testing/evaluation
train_fraction, val_fraction, test_fraction = 0.8, 0.1, 0.1
