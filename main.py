import numpy as np
import matplotlib.pyplot as plt
import h5py
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

import datetime as dt
import pandas as pd
import imageio


def average_grid(val_data, val_long, val_lat, long, lat, flipped=True, cropped=True):
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
    if cropped:
        cropping = np.where(count == 0)
        average[cropping] = np.nan
    if flipped:
        return np.flip(average, axis=0), np.flip(count, axis=0)
    else:
        return average, count


def h5_MOPITT_loader(dir_path, extent, size, averaging=1, n_pressure=2):
    averages, counts = [], []
    long = np.linspace(extent[0], extent[1], size[0])
    lat = np.linspace(extent[2], extent[3], size[1])
    DATAFIELD_NAME = '/HDFEOS/SWATHS/MOP02/Data Fields/RetrievedCOMixingRatioProfile'
    GEO_DATA = '/HDFEOS/SWATHS/MOP02/Geolocation Fields'
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    n_snaps = int(len(files) / averaging)
    for i in range(n_snaps):
        values, longitudes, latitudes = [], [], []
        for j in range(averaging):
            index = i * averaging + j
            path = os.path.join(dir_path, files[index])
            with h5py.File(path, mode='r') as file:
                # Extract Datasets
                data_var = file[DATAFIELD_NAME]
                data_lat = file[GEO_DATA + '/Latitude']
                data_lon = file[GEO_DATA + '/Longitude']
                # Read Values
                val_lat = data_lat[:]
                val_lon = data_lon[:]
                val_data = data_var[:, n_pressure, 0]
                longitudes += val_lon.tolist()
                latitudes += val_lat.tolist()
                values += val_data.tolist()
        average, count = average_grid(values, longitudes, latitudes, long, lat)
        averages.append(average)
        counts.append(count)
    return averages, counts


def simple_plot_map(matrix, extent, borderlines="white"):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(color=borderlines)
    ax.add_feature(cfeature.STATES, zorder=1, linewidth=1.5, edgecolor=borderlines)
    ax.imshow(matrix, transform=ccrs.PlateCarree(), extent=extent, cmap='inferno')
    plt.show()


def csv_MODIS_loader(file_path, extent, size, averaging=1, beginning="2020-08-01"):
    long = np.linspace(extent[0], extent[1], size[0])
    lat = np.linspace(extent[2], extent[3], size[1])
    date_format = "%Y-%m-%d"
    df = pd.read_csv(file_path)
    df["acq_date"] = pd.to_datetime(df["acq_date"], format=date_format)
    df["acq_date"] = pd.to_datetime(df["acq_date"], format=date_format)
    n_snaps = int(92 / averaging)
    beginning = dt.datetime.strptime(beginning, date_format)
    averages, counts, times = [], [], []
    for i in range(n_snaps):
        start_time = beginning + dt.timedelta(days=averaging * i)
        end_time = beginning + dt.timedelta(days=averaging * (i + 1))
        if averaging == 1:
            times.append(start_time.strftime("%Y-%m-%d"))
        else:
            times.append(start_time.strftime("%Y-%m-%d") + " to " + end_time.strftime("%Y-%m-%d"))
        mask = (df["acq_date"] >= start_time) & (df["acq_date"] <= end_time) & (df["confidence"] > 80)
        data = df.loc[mask]
        latitude = np.array(data["latitude"])
        longitude = np.array(data["longitude"])
        brightness = np.array(data["brightness"])
        average, count = average_grid(brightness, longitude, latitude, long, lat, cropped=False)
        averages.append(average)
        counts.append(count)
    return averages, counts, times


def create_fires_gif_map():
    average = 1
    MODIS_filename = "fire_archive_M-C61_245017.csv"
    extent = [-125, -115, 30, 45]
    size = [201, 401]
    averages, counts, times = csv_MODIS_loader(MODIS_filename, extent, size, averaging=average)
    images_files = []
    print("Start of Loop - Creating Images")
    n = len(counts)
    for i in range(n):
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title(times[i])
        ax.set_extent(extent)
        ax.coastlines(color="white")
        ax.add_feature(cfeature.STATES, zorder=1, linewidth=1.5, edgecolor="white")
        ax.imshow(averages[i], transform=ccrs.PlateCarree(), extent=extent, cmap='inferno')
        filename = f'Day_{i}.png'
        images_files.append(filename)
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f'{i * 100 / n:.2f} %')
    print("End of Loop - Creating GIF")
    # Creating the GIF
    with imageio.get_writer('Fire.gif', mode='I', fps=12) as writer:
        for filename in images_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    writer.close()
    # Delete Old Images
    for filename in set(images_files):
        os.remove(filename)
    print("Finished Creating GIF")


def plot_fire_levels():
    average = 1
    MODIS_filename = "fire_archive_M-C61_245017.csv"
    extent = [-125, -115, 30, 45]
    size = [201, 401]
    averages, counts, times = csv_MODIS_loader(MODIS_filename, extent, size, averaging=average)
    counts_list = []
    n = len(counts)
    for i in range(n):
        counts_list.append(np.nansum(counts[i]))
    times = np.arange(n)
    fig, ax = plt.subplots(1, 1)
    ax.plot(times, counts_list)
    ax.set_xlabel("Times")
    ax.set_ylabel("Number of Fire Pixels")
    ax.axvline(x=counts_list.index(max(counts_list)), color='red', linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_CO_levels():
    average = 1
    MOPPIT_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    extent = [-125, -115, 30, 45]
    size = [201, 401]
    P = 2
    averages, counts = h5_MOPITT_loader(MOPPIT_data_directory, extent, size, averaging=average, n_pressure=P)
    counts_list = []
    n = len(counts)
    for i in range(n):
        counts_list.append(np.nansum(averages[i]) / np.nansum(counts[i]))
    times = np.arange(n)
    fig, ax = plt.subplots(1, 1)
    ax.plot(times, counts_list)
    ax.set_xlabel("Times")
    ax.set_ylabel("Average CO Levels")
    ax.axvline(x=counts_list.index(max(counts_list)), color='red', linestyle='--')
    plt.tight_layout()
    plt.show()


def confront_CO_fire():
    average = 2
    MODIS_filename = "fire_archive_M-C61_245017.csv"
    MOPPIT_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    extent = [-125, -115, 30, 45]
    size = [201, 401]
    P = 2
    averages_CO, counts_CO = h5_MOPITT_loader(MOPPIT_data_directory, extent, size, averaging=average, n_pressure=P)
    averages_fires, counts_fires, times_fires = csv_MODIS_loader(MODIS_filename, extent, size, averaging=average)
    CO_list = []
    Fires_list = []
    n = len(counts_CO)
    for i in range(n):
        CO_list.append(np.nansum(averages_CO[i]) / np.nansum(counts_CO[i]))
        Fires_list.append(np.nansum(counts_fires[i]))
    times = np.arange(n)
    fig, ax = plt.subplots(1, 1)
    ax.plot(times, np.array(CO_list) / max(CO_list), label="CO Levels")
    ax.plot(times, np.array(Fires_list) / max(Fires_list), label="Fires' Numbers")
    ax.set_xlabel("Times")
    # ax.axvline(x=Fires_list.index(max(Fires_list)), color='red', linestyle='--')
    plt.tight_layout()
    plt.legend()
    plt.show()
    # Add Correlation Coefficients between CO and fire data
    corr = np.corrcoef(np.array(CO_list) / max(CO_list), np.array(Fires_list) / max(Fires_list))
    print(corr)


def plot_weeks():
    average = 7
    MODIS_filename = "fire_archive_M-C61_245017.csv"
    MOPPIT_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    extent = [-125, -115, 30, 45]
    size = [201, 401]
    P = 2
    averages_CO, counts_CO = h5_MOPITT_loader(MOPPIT_data_directory, extent, size, averaging=average, n_pressure=P)
    averages_fires, counts_fires, times_fires = csv_MODIS_loader(MODIS_filename, extent, size, averaging=average)
    CO_list = []
    Fires_list = []
    n = len(counts_CO)
    for i in range(n):
        CO_list.append(averages_CO[i])
        Fires_list.append(averages_fires[i])
    # Fires Weekly Plots
    plt.figure(figsize=(8, 10))
    axes = []
    for i in range(12):
        axes.append(plt.subplot(4, 3, i + 1, projection=ccrs.PlateCarree()))
        axes[i].set_extent(extent)
        axes[i].coastlines(color="white")
        axes[i].add_feature(cfeature.STATES, zorder=1, linewidth=0.25, edgecolor="white")
        axes[i].imshow(Fires_list[i], transform=ccrs.PlateCarree(), extent=extent, cmap='inferno')
    plt.show()
    # CO Weekly Plots
    plt.figure(figsize=(8, 10))
    axes = []
    for i in range(12):
        axes.append(plt.subplot(4, 3, i + 1, projection=ccrs.PlateCarree()))
        axes[i].set_extent(extent)
        axes[i].coastlines(color="black")
        axes[i].add_feature(cfeature.STATES, zorder=1, linewidth=0.25, edgecolor="black")
        axes[i].imshow(CO_list[i], transform=ccrs.PlateCarree(), extent=extent, cmap='inferno')
    plt.show()


if __name__ == '__main__':
    # plot_fire_levels()
    # plot_CO_levels()
    confront_CO_fire()
    # create_fires_gif_map()
    # simple_plot_map(averages_fires[0], extent)
    # simple_plot_map(averages_CO[0], extent)
    # plot_weeks()
    pass
