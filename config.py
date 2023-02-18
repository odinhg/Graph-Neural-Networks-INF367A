from os.path import join

# Paths
data_path = "data"
figs_path = "figs"
docs_path = "docs"
checkpoints_path = "checkpoints"

# Filenames
data_file_zip = join(data_path, "traffic_data.zip")
data_file_pkl = join(data_path, "traffic_data.pkl")
stations_data_file = join(data_path, "traffic_stations.csv")
summary_table_file = join(docs_path, "data_summary_table.md")
time_series_file = join(data_path, "time_series_data.pkl")


# Globals
min_number_of_observations = 1500   # Drop stations having too few observations
num_workers = 4                     # Number of workers to use with dataloader
device = "cpu"                      # Device for PyTorch to use
val_fraction = 0.1                  # Fraction of data to use for validation data
test_fraction = 0.1                 # Fraction of data to use for test data
normalize_data = True               # Normalize dataset using mean and std for training data
validations_per_epoch = 10          # How many time to do validation per epoch

# Baseline model
config_baseline = {}
config_baseline["batch_size"] = 64 
config_baseline["lr"] = 0.001
config_baseline["epochs"] = 3 
config_baseline["checkpoint_file"] = join(checkpoints_path, "baseline.pth") 
