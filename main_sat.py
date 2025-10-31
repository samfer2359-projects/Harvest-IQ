import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pystac_client import Client
import planetary_computer
import stackstac

def compute_ndvi(bbox, datetime):
    """
    Compute NDVI using Sentinel-2 data based on the provided bounding box and time range.
    This function does not retrain any model, it simply computes NDVI using the provided data.

    :param bbox: List of bounding box coordinates [lon_min, lat_min, lon_max, lat_max]
    :param datetime: Time range string (e.g., "2025-01-01/2025-01-10")
    :return: xarray.DataArray containing the computed NDVI values
    """
    # Connect to Planetary Computer STAC catalog
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # Search for Sentinel-2 L2A images with low cloud cover
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": 10}}  # Low cloud cover
    )

    # Retrieve the search results
    items = list(search.get_items())
    if len(items) == 0:
        raise ValueError("No Sentinel-2 images found for the given area/time range.")

    # Sign items to authorize asset access via Planetary Computer
    signed_items = [planetary_computer.sign(item) for item in items]

    # Stack Red (B04) and NIR (B08) bands from the signed items
    ds = stackstac.stack(
        signed_items,
        assets=["B04", "B08"],
        bounds_latlon=bbox,
        epsg=4326,
        resolution=0.00025
    )

    # Convert to xarray dataset and extract Red and NIR bands
    ds = ds.to_dataset("band")
    red = ds["B04"]
    nir = ds["B08"]

    # Check the coordinates and rename to 'latitude' and 'longitude' if necessary
    coords = list(ds.coords)
    if 'lat' in coords and 'lon' in coords:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    elif 'latitude' in coords and 'longitude' in coords:
        pass  # Already correct, do nothing
    else:
        raise ValueError("The dataset does not contain 'lat' or 'longitude' dimensions")

    # Compute NDVI
    ndvi = (nir - red) / (nir + red)
    
    # Take the median of NDVI over time (this reduces the time dimension)
    ndvi_median = ndvi.median(dim="time")

    return ndvi_median

def get_ndvi_paths(bbox, datetime):
    """
    Computes NDVI, saves a plot and NetCDF file, and returns their paths.
    This function uses the computed NDVI data and saves it as both an image and NetCDF file.

    :param bbox: List of bounding box coordinates [lon_min, lat_min, lon_max, lat_max]
    :param datetime: Time range string (e.g., "2025-01-01/2025-01-10")
    :return: Dictionary with file paths for the NDVI plot image and NetCDF file
    """
    # Ensure the output directory exists
    output_dir = 'static/ndvi_output'
    os.makedirs(output_dir, exist_ok=True)

    # Compute NDVI data using the given bounding box and time range
    ndvi = compute_ndvi(bbox, datetime)

    # Save NDVI plot as an image
    plot_path = os.path.join(output_dir, 'ndvi_plot.png')
    plt.figure(figsize=(8, 6))
    ndvi.plot(cmap='YlGn')  # Use a green-yellow colormap for NDVI
    plt.title("NDVI Plot")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Save NDVI data to NetCDF file (excluding unnecessary metadata)
    netcdf_path = os.path.join(output_dir, 'ndvi_output.nc')
    # Save without including extra attributes that might cause issues
    ndvi.to_netcdf(netcdf_path, encoding={'latitude': {'dtype': 'float32'}, 'longitude': {'dtype': 'float32'}})

    # Return the paths to the plot and NetCDF file
    return {
        "ndvi_plot": plot_path,      # Path to NDVI plot image
        "ndvi_netcdf": netcdf_path   # Path to NDVI NetCDF file
    }
