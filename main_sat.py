import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Set Matplotlib backend to Agg (suitable for non-GUI environments)
import matplotlib
matplotlib.use('Agg')

def get_ndvi_paths():
    """
    Main function to calculate NDVI, generate the plot, and save the data as NetCDF.
    """
    # Ensure output directory exists
    output_dir = 'static/ndvi_output'
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Compute NDVI data
    ndvi_data = compute_ndvi_somehow()  # Replace with your actual NDVI computation logic

    # Step 2: Plot and save the NDVI image
    plot_path = os.path.join(output_dir, 'ndvi_plot.png')
    save_ndvi_plot(ndvi_data, plot_path)

    # Step 3: Save the NDVI data to a NetCDF file
    netcdf_path = os.path.join(output_dir, 'ndvi_output.nc')
    save_ndvi_to_netcdf(ndvi_data, netcdf_path)

    # Return paths to the generated plot and NetCDF file
    return {
        "ndvi_plot": plot_path,  # Path to the NDVI plot image
        "ndvi_netcdf": netcdf_path  # Path to the NDVI NetCDF file
    }


def compute_ndvi_somehow():
    """
    Compute the NDVI (Normalized Difference Vegetation Index) using satellite data or sample logic.
    This function should be replaced with actual NDVI computation logic using appropriate data.

    Returns:
        np.ndarray: A 2D numpy array representing the NDVI values.
    """
    # Example: Generate random NDVI data (replace with actual NDVI computation)
    ndvi_data = np.random.rand(100, 100)  # Random data for illustration purposes
    return ndvi_data


def save_ndvi_plot(ndvi_data, plot_path):
    """
    Save the NDVI plot to a file as a PNG image.

    Args:
        ndvi_data (np.ndarray): 2D array of NDVI values.
        plot_path (str): Path where the NDVI plot image will be saved.
    """
    plt.figure(figsize=(8, 6))

    # Use imshow for 2D array visualization with colormap
    im = plt.imshow(ndvi_data, cmap='YlGn')  # 'YlGn' colormap (Green shades for NDVI)
    plt.colorbar(im)  # Optional: Add a colorbar to show NDVI scale
    plt.title("NDVI Plot")
    
    # Save the plot
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free resources


def save_ndvi_to_netcdf(ndvi_data, netcdf_path):
    """
    Save the NDVI data to a NetCDF file.

    Args:
        ndvi_data (np.ndarray): 2D array of NDVI values.
        netcdf_path (str): Path where the NetCDF file will be saved.
    """
    # Convert the NDVI data into an xarray.DataArray
    ndvi_xr = xr.DataArray(ndvi_data, dims=["lat", "lon"], 
                            coords={"lat": np.arange(ndvi_data.shape[0]), 
                                    "lon": np.arange(ndvi_data.shape[1])})

    # Save the NDVI DataArray to a NetCDF file
    ndvi_xr.to_netcdf(netcdf_path)


# Main entry point when running the script
if __name__ == "__main__":
    ndvi_paths = get_ndvi_paths()
    print(f"NDVI plot saved at: {ndvi_paths['ndvi_plot']}")
    print(f"NDVI data saved in NetCDF format at: {ndvi_paths['ndvi_netcdf']}")
