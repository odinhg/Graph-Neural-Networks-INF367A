# Traffic data and graph neural networks
### Project 1 in INF367A : Topological Deep Learning
**Odin Hoff Gardå, March 2023**

![Toad](docs/toad.png)

# Introduction

We train and compare four machine learning models, one fully connected neural network and three graph neural networks. The objective is to predict traffic volumes for all traffic stations (nodes) for the next hour given the current traffic volumes, month, weekday and hour.

## Quick start

1. Run `unpack_data.py` to unpack compressed data file.
2. Run `preprocess_data.py` to pre-process data.
3. Run `create_data_summary.py` to generate a table summarizing the dataset (optional).
4. Run `train_and_evaluate.py` to train and evaluate any of the four models.

Configurations for pre-processing, training and the different models can be set in `config.py`. Plots of training and validation losses are saved in `figs/`. Plots of predictions and ground truth for some selected traffic stations are saved in `figs/<model name>_predictions/`. The data summary table is saved in `docs/` as a Markdown file.

# Pre-processing and data summary

The dataset, provided by Statens Vegvesen, contains observed traffic volumes from 109 traffic stations at Vestlandet in Norway. The observations are collected every hour from 2014-12-31 23:00 to 2022-12-31 21:00. We also have GPS coordinates (latitude and longitude) for each traffic station in the dataset. A table summarizing the dataset can be found in `docs/data_summary_table.md` (created by running `create_data_summary.py`).

![Stations on map](./docs/traffic_stations_map_1.png)
*Figure: Geographic positions of the traffic stations.*

## Pre-processing of data

### Stations included

Stations with less than $1500$ observation in total are dropped from the dataset. This limit can be set in `config.py` before running the pre-processing script. After dropping stations with too few observations, we are left with 95 stations.

### Missing values

Some observations are missing and are given the value NaN. These NaN values are later replaced by $-1$ in the custom PyTorch dataset before being fed to the model. For an unknown reason, all stations are missing observations at 22:00 every day. We fill these rows with the mean value of the row before and row after the affected row.

### Normalization

No normalization was used in the end. The L1-loss (MAE) seems to handle unnormalized data well, and normalization did not improve the validation loss. Normalization by scaling the data to $[0,1]$ or by computing z-scores can be selected in `config.py` before running the pre-processing script. The statistics (minimum, maximum, mean and standard deviation) are computed on the training data to prevent data leakage.

### Dataset split

The dataset is split into training, validation and test data in chronological order since we want to test the model's performances on data newer than the training data. We use the following ratios when splitting the dataset:

| Dataset | Fraction | Rows |
|-|-|-|
|Train|0.7|49088|
|Validation|0.15|10518|
|Test|0.15|10518|

The above ratios can be changed in `config.py` before running the pre-processing script. Note that the training, validation and test data are saved to separate files in the `data/` directory.

# Model descriptions

Four models, named Baseline, GNN, GNN_NE and GNN_KNN, respectively were trained and evaluated on the pre-processed dataset.

|Baseline||
|-|-|
|**Description**|Fully connected NN with five linear layers. Batch normalization and ReLU activation functions between layers. The input layer consists of 98 nodes (95 traffic stations, month, weekday and hour). The model is defined in `models/baseline.py`.|
|**Parameters**|494015 (1.98MB)|

|GNN||
|-|-|
|**Description**|Graph NN with edge, node and global models. We follow the approach described in the paper [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261). The edge features are updated first, followed by the node features and then the global/graph features.<br />![Full GN block](https://github.com/odinhg/Graph-Neural-Networks-INF367A/blob/main/docs/full_gn_block.png)<br />*Figure: Diagram showing the structure of one `GNNLayer` instance. There are three MLPs* $\phi^e, \phi^v$ *and* $\phi^u$*, and three aggregation functions* $\rho^{e\to v}, \rho^{e\to u}$ *and* $\rho^{v\to u}$.<br /><br />![Full GN block algorithm](https://github.com/odinhg/Graph-Neural-Networks-INF367A/blob/main/docs/full_gn_block_algorithm.png)<br />*Figure: The algorithm implemented in `GNNLayer`.*<br /><br />We use the arithmetic mean as aggregation in all models. Using the `MetaLayer` class in PyG, we construct the `GNNLayer` class. Stacking five of these layers we create the `GNNModel` class found in `models/gnn.py`. <br /><br/>The graph (edges) used was hand-crafted by looking at a map showing the roads between stations. The adjacency matrix for this graph is stored in the file `data/graph.pkl` as a pickled DataFrame. <br />![Hand-crafted graph](https://github.com/odinhg/Graph-Neural-Networks-INF367A/blob/main/docs/traffic_stations_map_with_edges_1.png)<br />*Figure: Hand-crafted graph.* <br /><br />Both node and edge features are 1-dimensional and consist of traffic volume and $e^{-d_{ij}}$ (where $d_{ij}$ is the geodesic distance between nodes $i$ and $j$), respectively. The idea behind the edge feature is that edges between nearby traffic stations are more important than those far apart. The global graph features are 3-dimensional and consist of month number, weekday number and hour of day.|
|**Parameters**|113999 (0.51MB)|

|GNN_NE||
|-|-|
|**Description**|The GNN_NE is the same model as GNN but without the edge model. That is, we do not update the edge features in this model. Furthermore, all edges are weighted equally with value $1.0$.|
|**Parameters**|83914 (0.37MB)|

|GNN_KNN||
|-|-|
|**Description**|The GNN_KNN model is the same model as GNN. The only difference is the graph used. Instead of the manually crafted graph, we create the graph by connecting each node to its 10 nearest neighbours (using geodesic distances). The number of neighbours can be changed in the function `create_edge_index_and_features()` found in `utils/dataloader.py`. |
|**Parameters**|113999 (0.51MB)|

# Model comparisons

All four models were trained using the following configuration:

|Configuration details||
|-|-|
|**Batch size**|128|
|**Learning rate**|0.001|
|**Optimizer**|Adam with default parameters|
|**LR scheduler**|No (but implemented in code)|
|**Validation steps per epoch**|4|
|**Earlystopping rule**|Stop if 20 consecutive steps have validation loss worse than the so far lowest validtion loss + 0.5|
|**Loss function**| $L_1$-loss (MAE)|

**Note:** We report the generalization error as both the mean absolute error (MAE) and the root mean squared error (RMSE). Using $L_1$-loss during training gave better stability and faster convergence compared to using $L_2$-loss. 

The following table summarizes the results from training and evaluation of the models on the test dataset:

|**Model**|**Epochs trained**|**Epoch model saved at**|**Mean epoch time**|**Total training time**|**Test MAE**|**Test RMSE**|
|-|-|-|-|-|-|-|
|**Baseline**|88|80|7.7s|674s|54.29|**117.69**|
|**GNN**|49|43|14.8s|723s|28.10|**53.00**|
|**GNN_NE**|31|26|11.7s|363s|33.19|**65.30**|
|**GNN_KNN**|61|54|15s|914s|28.96|**55.14**|

## Plots of training and validation losses

The following plots show the training and validation losses for each of the models during training. Notice that the GNN models have an earlier and steeper drop in loss than the fully connected Baseline model. Note that the loss plots uses log scale on vertical axis.

### Baseline
![Loss plot for Baseline](figs/baseline_loss_plot.png)
*Figure: Training and validation loss for the Baseline model.*

### GNN
![Loss plot for GNN](figs/gnn_loss_plot.png)
*Figure: Training and validation loss for the GNN model.*

### GNN_NE
![Loss plot for GNN_NE](figs/gnn_ne_loss_plot.png)
*Figure: Training and validation loss for the GNN_NE model.*

### GNN_KNN
![Loss plot for GNN_KNN](figs/gnn_knn_loss_plot.png)
*Figure: Training and validation loss for the GNN_KNN model.*

## GNN vs Baseline model
There is a huge improvement in training time and test accuracy. All GNN based models performs much better than the baseline model even though they have far fewer parameters. Some advantages of using a FCNN includes ease of implementation, fast training (time per epoch) and fast evaluation (forward pass). But there is likely room for optimizing the GNN implementations more than what was done in this project.

### Differences in predictions
The following plots demonstrate the differences in predictions between the FCNN and GNN networks.

|Baseline|GNN|
|-|-|
|![Baseline predictions](figs/baseline_predictions/001_78845V804838.png)|![GNN predictions](figs/gnn_predictions/001_78845V804838.png)|

*Figure: During one period in the plot, there is no reported traffic at this station (possible due to road work, accident or equipment fault/maintenance). The Baseline FCNN model seems to have learned the traffic as a function of time data and predicts traffic as normal. The GNN on the other hand, correctly predicts that there is no traffic during this period. This suggests that the GNN uses data from the node itself and its neighbours and does not solely rely on the time data.*

|Baseline|GNN|
|-|-|
|![Baseline predictions](figs/baseline_predictions/065_03016V805614.png)|![GNN predictions](figs/gnn_predictions/065_03016V805614.png)|

*Figure: It is not clear why the Baseline model performs so poorly in this example, but it might be because of abnormaly high traffic volume at the station. The GNN performs a lot better in this case too. Well done GNN!*

|Baseline|GNN|
|-|-|
|![Baseline predictions](figs/baseline_predictions/020_65182V384013.png)|![GNN predictions](figs/gnn_predictions/020_65182V384013.png)|

*Figure: Here, we are missing values for the entire time period shown (recall that missing values are given the value -1.0). The Baseline model's predictions are all over the place, whereas the GNN's predictions are far more stable and close to -1.0.*

## GNN vs GNN_NE
The GNN model with edge feature updates performs better than the GNN_NE model which only updates node and graph features. But this comparison might be somewhat unfair since the GNN_NE model has fewer parameters in total.

## GNN vs GNN_KNN
The performance of the models GNN and GNN_KNN are similar with the GNN slightly better. The GNN_KNN model uses about 26% *more time to complete training*! This is possibly because it also has to learn which edges are important. Also, we have more edges, so a bit more data needs to be fed through the model resulting in a small increase in mean epoch time. One advantage to using the kNN based graph is that we save time and effort by outsourcing some work to the model.

### Remark: Expert opinion vs. auto-generated priors 
It is interesting to test how the GNN and GNN_KNN models compare if we restrict the size of the training dataset. Given enough data, it seems likely that the network can learn which edges in the kNN graph are important. But with less training data, will giving geometric priors of "higher quality" give even better results? By higher quality, we mean restricting our hypothesis class based on deeper knowledge about the domain.

We restrict the amount of training data to 50%, 40%, 30% and 20%, and compare the performance of the two models. The remaining data is equally split between validation and test data.

|**Data used for training**|GNN Test MAE | GNN Test RMSE | GNN_KNN Test MAE | GNN_KNN Test RMSE | $\Delta\text{RMSE}$|
|-|-|-|-|-|-|
|70%|28.10|53.00|28.96|55.14|2.14|
|50%|28.54|53.86|29.47|56.02|2.16|
|40%|30.82|59.61|33.14|65.34|5.73|
|30%|33.14|65.34|35.42|70.92|5.58|
|20%|37.68|74.88|37.58|75.92|1.04|

It is difficult to derive a certain conclusion from the above results. The GNN model using a hand-crafted graph always performs a bit better than the one using a kNN based graph (with $k=10$ in these experiments). It could be interesting to redo the experiments with a higher value for $k$.

### Remark: An idea for the future
Another thing that would be interesting to try: setting $k$ to a higher number (or using the complete graph) we could try to predict the edges which are most important.

# Concluding remarks
Knowing that the traffic volume at a geographic position is heavily correlated to the traffic along the same road and at nearby positions, we take advantage of this by providing geometric priors (in form of a graph). This improves accuracies, training times and the size of the model drastically.

Of course, there are many choices when it comes to the exact architecture (number layers, hidden features and hidden dimension in the MLPs in the edge, node and global models) of the GNN models. Adding more parameters to the models did not significantly improve the performance on validation data. One could also consider adding residual/skip connection to the edge, node and/or global model.
