# Satellite NDVI Processing Script

from pystac_client import Client
import planetary_computer
import stackstac
import xarray as xr
import matplotlib.pyplot as plt
import os

# Connect to Planetary Computer STAC catalog
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

# Define area of interest (change this to the area you want to process)
bbox = [78.0, 20.0, 79.0, 21.0]  # Example region (India)
datetime = "2025-01-01/2025-01-10"

# Search for Sentinel-2 L2A images with low cloud cover
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=datetime,
    query={"eo:cloud_cover": {"lt": 10}}
)

items = list(search.get_items())
print(f"Images found: {len(items)}")

if len(items) == 0:
    raise ValueError("No Sentinel-2 images found for the given area/time range.")

# Sign items to authorize asset access
signed_items = [planetary_computer.sign(item) for item in items]

# Stack Red (B04) and NIR (B08) bands
ds = stackstac.stack(
    signed_items,
    assets=["B04", "B08"],
    bounds_latlon=bbox,
    epsg=4326,
    resolution=0.00025
)

# Convert to dataset and extract bands
ds = ds.to_dataset("band")
red = ds["B04"]
nir = ds["B08"]

# Compute NDVI (use median across time if multiple time steps)
ndvi = (nir - red) / (nir + red)
ndvi_median = ndvi.median(dim="time")  # Get median NDVI across time

print("✅ NDVI computed successfully!")

# Visualize NDVI
fig, ax = plt.subplots(figsize=(8, 8))
ndvi_median.plot(ax=ax, cmap="YlGn")
ax.set_title("NDVI Median (Jan 2025)")

# Save the plot as a PNG image (instead of downloading it)
output_folder = 'static/ndvi_output'
os.makedirs(output_folder, exist_ok=True)
plot_path = os.path.join(output_folder, 'ndvi_plot.png')
fig.savefig(plot_path)
plt.close(fig)

# Optionally, return or print the path where the plot is saved
print(f"✅ NDVI plot saved as '{plot_path}'")

# Also, return the NDVI dataset as a NetCDF file
netcdf_path = os.path.join(output_folder, 'ndvi_output.nc')
ndvi_median = ndvi_median.drop_vars(['proj:shape', 'spatial_ref'], errors='ignore')
ndvi_median.to_netcdf(netcdf_path)
print(f"✅ NDVI dataset saved as '{netcdf_path}'")

# Return the paths to be used in Flask or other scripts
def get_ndvi_paths():
    return {
        "ndvi_plot": plot_path,
        "ndvi_netcdf": netcdf_path
    }
