import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def distance(s1, s2):
    return geodesic(s1, s2).km

# Only consider traffic stations included in our dataset (where some stations are excluded)
#included_ids = pd.read_pickle("data/time_series_test.pkl").columns
stations = pd.read_csv("data/traffic_stations.csv")
#stations = stations.loc[stations["id"].isin(included_ids), :]

coordinates = stations.loc[:, ["latitude", "longitude"]]
distances = pdist(coordinates, metric=distance)



fig = go.Figure(
        go.Scattermapbox(
            lat = stations.loc[:, "latitude"],
            lon = stations.loc[:, "longitude"],
            mode = "text+markers",
            text = list(range(0,len(stations))), #stations.loc[:, "id"],
            textposition = "top center",
            marker_size = 12,
        )
    )

# Add edges to graph
graph_df = pd.read_pickle("data/graph.pkl")
for i in range(0, len(graph_df)):
    for j in range(i, len(graph_df)):
        if graph_df.iloc[i,j] == 1:
            #station_1 = graph_df.columns[i]
            #station_2 = graph_df.columns[j]
            fig.add_trace(
                    go.Scattermapbox(
                        lat = [stations.iloc[i]["latitude"], stations.iloc[j]["latitude"]],
                        lon = [stations.iloc[i]["longitude"], stations.iloc[j]["longitude"]],
                        mode = "lines",
                    )
                )

"""
fig.add_trace(
        go.Scattermapbox(
            lat = [stations.iloc[0]["latitude"], stations.iloc[2]["latitude"]],
            lon = [stations.iloc[0]["longitude"], stations.iloc[2]["longitude"]],
            mode = "lines",
        )
    )
"""

fig.update_layout(title_text = "Traffic Stations", title_x = 0.5, showlegend=False)
fig.update_layout(mapbox_style="open-street-map")
fig.show()

"""
fig = px.scatter_mapbox(stations, lat='latitude', lon='longitude', hover_name="id", zoom=10, size_max=50)
fig.update_traces(marker={'size': 15})
fig.update_layout(title = 'Traffic Stations', title_x=0.5)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
"""

