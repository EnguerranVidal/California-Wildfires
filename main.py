import numpy as np
import matplotlib.pyplot as plt
import h5py
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

import geopandas as gpd
from shapely.geometry import Point, Polygon
import shapely
import pandas as pd

from collections import namedtuple


def test_loader():
    n_level_pressure = 1
    days_average = 10
    extent = [-125, -115, 30, 45]
    size = [41, 61]
    DATAFIELD_NAME = '/HDFEOS/SWATHS/MOP02/Data Fields/RetrievedCOMixingRatioProfile'
    GEO_DATA = '/HDFEOS/SWATHS/MOP02/Geolocation Fields'
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    n_snaps = int(len(files) / days_average)
    for i in range(n_snaps):
        values = []
        longitudes = []
        latitudes = []
        for j in range(days_average):
            index = i * days_average + j
            path = os.path.join(data_dir, files[index])
            with h5py.File(path, mode='r') as file:
                # Extract Datasets
                data_var = file[DATAFIELD_NAME]
                data_lat = file[GEO_DATA + '/Latitude']
                data_lon = file[GEO_DATA + '/Longitude']
                data_lev = file[GEO_DATA + '/Pressure']
                data_date = file[GEO_DATA + '/Time']
                # Read Values
                val_lat = data_lat[:]
                val_lon = data_lon[:]
                val_lev = data_lev[n_level_pressure]
                val_time = data_date[0]
                val_data = data_var[:, n_level_pressure, 0]
                # Get units
                data_units = data_var.attrs.get('units', default=None)
                lat_units = data_lat.attrs.get('units', default=None)
                lev_units = data_lev.attrs.get('units', default=None)
                lon_units = data_lon.attrs.get('units', default=None)
                lev_title = 'Pressure'
                lat_title = 'Latitude'
                lon_title = 'Longitude'
                data_title = 'RetrievedCOMixingRatioProfile'
                longitudes += val_lon.tolist()
                latitudes += val_lat.tolist()
                values += val_data.tolist()
        long = np.linspace(extent[0], extent[1], size[0])
        lat = np.linspace(extent[2], extent[3], size[1])
        average, count = average_grid(values, longitudes, latitudes, long, lat)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent)
        ax.coastlines(color='white')
        ax.add_feature(cfeature.STATES, zorder=1, linewidth=1.5, edgecolor='white')
        ax.imshow(average, transform=ccrs.PlateCarree(), extent=extent, cmap='inferno')
        plt.show()


def average_grid(val_data, val_long, val_lat, long, lat):
    count = np.zeros((lat.shape[0] - 1, long.shape[0] - 1))
    average = np.zeros((lat.shape[0] - 1, long.shape[0] - 1))
    n = len(val_data)
    for i in range(n):
        if val_data[i] != np.nan and val_data[i] > 0:
            if long[0] < val_long[i] < long[-1] and lat[0] < val_lat[i] < lat[-1]:
                index_long = np.digitize(val_long[i], long) - 1
                index_lat = np.digitize(val_lat[i], lat) - 1
                average[index_lat, index_long] += val_data[i]
                count[index_lat, index_long] += 1
    valid = count != 0
    average[valid] = average[valid] / count[valid]
    test = np.where(count == 0)
    average[test] = np.nan
    return average, count


if __name__ == '__main__':
    test_loader()
