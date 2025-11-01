# main_sat.py

# Set Matplotlib backend to Agg (suitable for non-GUI environments)
import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import xarray as xr

def get_ndvi_paths():
    # Ensure output directory exists
    output_dir = 'static/ndvi_output'
    os.makedirs(output_dir, exist_ok=True)

    # Your NDVI calculation logic here
    # For example, using some sample logic:
    ndvi = compute_ndvi_somehow()  # Replace with your actual NDVI computation logic

    # Save NDVI plot
    plot_path = os.path.join(output_dir, 'ndvi_plot.png')
    plt.figure(figsize=(8, 6))
    ndvi.plot(cmap='YlGn')
    plt.title("NDVI Plot")
    plt.savefig(plot_path)

    # Save NDVI data to NetCDF file
    netcdf_path = os.path.join(output_dir, 'ndvi_output.nc')
    ndvi.to_netcdf(netcdf_path)

    return {
        "ndvi_plot": plot_path,  # Path to the NDVI plot image
        "ndvi_netcdf": netcdf_path  # Path to the NDVI NetCDF file
    }

def compute_ndvi_somehow():
    # Replace with actual logic to compute NDVI (using satellite data, etc.)
    # This is just a placeholder function.
    return xr.DataArray([1, 2, 3])  # Example placeholder