import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def distance(s1, s2):
    # Compute distance between two stations
    # Only needed because *some* people don't believe the earth is flat
    return geodesic(s1, s2).km

stations = pd.read_csv("data/traffic_stations.csv")
coordinates = stations.loc[:, ["latitude", "longitude"]]
distances = pdist(coordinates, metric=distance)

fig = go.Figure(
        go.Scattermapbox(
            lat = stations.loc[:, "latitude"],
            lon = stations.loc[:, "longitude"],
            mode = "text+markers",
            text = stations.loc[:, "id"],
            textposition = "top center",
            marker_size = 12,
        )
    )
fig.update_layout(title_text = "Traffic Stations", title_x = 0.5)
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

