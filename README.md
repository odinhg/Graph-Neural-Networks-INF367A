# Traffic data and graph neural networks
## Project 1 in INF367A : Geometric / Topological Deep Learning
### Odin Hoff Gardå, 2023 

In the following project we are given the traffic volumes registred at different traffic stations at a given hour and we aim to predict the traffic volumes for the next hour. There are two models up for comparison. Namely, one basic fully connected neural network (baseline) and one GNN (graph neural network). The dataset comes from Statens Vegvesen.

## Quickstart

Here are the steps to follow if you want to train and evaluate the models from scratch.

**Data pre-processing:**

1. Run `unpack_data.py` to extract zipped datafile to a pickled dataframe.
2. Run `create_data_summary.py` to create a table summarizing the data (saved as `docs/data_summary_table.md`).
3. 

**Baseline model:**

1. Train
2. Evaluate on test data

**Graph neural network model:**

1. Train
2. Evaluate on test data

## Overview of data and model architectures

### Data exploration and pre-processing
The dataset contains observations from 109 unique traffic stations. Some of them are built pretty recently and have less than 1300 observations. For simplicity, we will not include these stations in the final dataset. In total we then have 95 stations with the number of observations ranging from 37963 to 67077. The timestamps of the observations ranges from 2014-12-31 (23:00) to 2022-12-31 (21:00).

### The baseline model


### The graph nerural network model


