import os
import csv
import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject

def resample_raster(src_path, ref_raster):
    """Resamples the input raster to match the resolution and extent of the reference raster using bilinear interpolation."""
    with rasterio.open(src_path) as src:
        data = src.read(1, masked=True)  # Read as a masked array

        # Create empty array for resampled data
        resampled_data = np.empty(ref_raster.shape, dtype=np.float32)

        # Perform reprojection/resampling with bilinear
        reproject(
            source=data,
            destination=resampled_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_raster.transform,
            dst_crs=ref_raster.crs,
            resampling=Resampling.bilinear
        )

        # Convert to masked array (ignoring NaN values)
        resampled_masked = np.ma.masked_invalid(resampled_data)

    return resampled_masked

def compute_nrmse(reference, prediction, normalization="range"):
    """Computes the Normalized RMSE between reference and prediction rasters, expressed as a percentage."""
    mask = ~reference.mask & ~prediction.mask  # Only compare overlapping valid data
    ref_values = reference[mask]
    pred_values = prediction[mask]

    # Compute RMSE
    rmse = np.sqrt(np.mean((ref_values - pred_values) ** 2))

    # Choose normalization factor
    if normalization == "range":
        norm_factor = ref_values.max() - ref_values.min()
    elif normalization == "mean":
        norm_factor = np.mean(ref_values)
    elif normalization == "std":
        norm_factor = np.std(ref_values)
    else:
        raise ValueError("Invalid normalization method. Choose 'range', 'mean', or 'std'.")

    return (rmse / norm_factor) * 100  # Convert to percentage

def process_nrmse(data_folder, variable_name, normalization_method="range"):
    """Processes NRMSE for a given dataset and returns results as a list of dicts."""
    reference_filename = f"Max_{variable_name}_N10m.tif"
    prediction_filenames = [
        f"Max_{variable_name}_N15m.tif",
        f"Max_{variable_name}_N20m.tif",
        f"Max_{variable_name}_N30m.tif"
    ]

    reference_path = os.path.join(data_folder, reference_filename)
    results = []

    if not os.path.exists(reference_path):
        print(f"Error: Reference file not found: {reference_path}")
        return results

    # Read the reference raster
    with rasterio.open(reference_path) as ref_src:
        ref_data = ref_src.read(1, masked=True)

        for pred_filename in prediction_filenames:
            pred_path = os.path.join(data_folder, pred_filename)

            if not os.path.exists(pred_path):
                print(f"Warning: Prediction file not found: {pred_path}")
                continue

            resampled_pred = resample_raster(pred_path, ref_src)
            nrmse = compute_nrmse(ref_data, resampled_pred, normalization=normalization_method)

            results.append({
                'variable': variable_name,
                'prediction_file': pred_filename,
                'nrmse': nrmse,
                'normalization_method': normalization_method
            })

            print(f"NRMSE for {pred_filename} (bilinear interpolation): {nrmse:.2f}% (Normalization: {normalization_method})")

    return results

# Define datasets with their respective normalization methods
datasets = [
    {"var_folder": "depth", "file_prefix": "depth", "normalization": "range"},
    {"var_folder": "speed", "file_prefix": "speed", "normalization": "range"},
    {"var_folder": "solid_frac", "file_prefix": "solids_frac", "normalization": "range"},
    {"var_folder": "time", "file_prefix": "Inundation_time", "normalization": "range"},
    {"var_folder": "erosion", "file_prefix": "erosion", "normalization": "range"},
    {"var_folder": "IP", "file_prefix": "impact_pressure", "normalization": "range"}
]

# Set base folder path where datasets are stored
base_folder = "NRMSE"

# Collect all results
all_results = []
for dataset in datasets:
    var_folder = dataset["var_folder"]
    file_prefix = dataset["file_prefix"]
    normalization_method = dataset["normalization"]
    data_folder_path = os.path.join(base_folder, var_folder)

    print(f"\n===== Processing {var_folder.capitalize()} Dataset =====")
    results = process_nrmse(data_folder_path, file_prefix, normalization_method)
    all_results.extend(results)

# Save results to CSV
csv_filename = "nrmse_results.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['variable', 'prediction_file', 'nrmse', 'normalization_method']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for result in all_results:
        # Format NRMSE to 2 decimal places with percentage sign
        formatted_result = {
            'variable': result['variable'],
            'prediction_file': result['prediction_file'],
            'nrmse': f"{result['nrmse']:.2f}%",
            'normalization_method': result['normalization_method']
        }
        writer.writerow(formatted_result)

print(f"\nResults saved to {csv_filename}")