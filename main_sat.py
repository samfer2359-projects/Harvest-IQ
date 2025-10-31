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

    items = list(search.get_items())
    if len(items) == 0:
        raise ValueError("No Sentinel-2 images found for the given area/time range.")

    # ✅ Sign items to authorize asset access via Planetary Computer
    signed_items = [planetary_computer.sign(item) for item in items]

    # Stack Red (B04) and NIR (B08) bands from the signed items
    ds = stackstac.stack(
        signed_items,
        assets=["B04", "B08"],
        bounds_latlon=bbox,
        epsg=4326,
        resolution=0.00025
    )

    # Debug: Print out the dataset dimensions to understand its structure
    print("Dataset dimensions:", ds.dims)

    # If the dataset does not contain 'lat' or 'lon', rename the dimensions
    if 'latitude' in ds.dims and 'longitude' in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    elif 'y' in ds.dims and 'x' in ds.dims:
        ds = ds.rename({"y": "lat", "x": "lon"})

    # Convert to dataset and extract Red and NIR bands
    ds = ds.to_dataset("band")
    red = ds["B04"]
    nir = ds["B08"]

    # Remove any non-data variables (e.g., projection or shape metadata)
    ds = ds.drop_vars(['proj:shape', 'proj:transform', 'spatial_ref'], errors='ignore')

    # Compute NDVI
    ndvi = (nir - red) / (nir + red)
    
    # Take the median of NDVI over time
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
    # Ensure output directory exists
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

    # Save NDVI data to NetCDF file
    netcdf_path = os.path.join(output_dir, 'ndvi_output.nc')
    ndvi.to_netcdf(netcdf_path)

    # Return the paths to the plot and NetCDF file
    return {
        "ndvi_plot": plot_path,      # Path to NDVI plot image
        "ndvi_netcdf": netcdf_path   # Path to NDVI NetCDF file
    }
