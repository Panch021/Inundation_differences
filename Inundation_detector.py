import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import csv

# Configuration
REFERENCE_PATH = 'reference_10m.tif'  # 10m reference raster
COARSE_PATHS = [
    'prediction_15m.tif',  # 15m prediction raster
    'prediction_20m.tif',  # 20m prediction raster
    'prediction_30m.tif'  # 30m prediction raster
]
OUTPUT_DIR = 'output'  # Directory to save results
INUNDATED_VALUE = 1  # Pixel value indicating inundation
NODATA_CLASS = 255  # NoData value for output classification rasters


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open reference raster and read data
    with rasterio.open(REFERENCE_PATH) as ref_dst:
        ref_profile = ref_dst.profile
        ref_array = ref_dst.read(1)
        ref_nodata = ref_dst.nodata if ref_dst.nodata is not None else -9999

    # Mask reference array to only include inundation pixels
    ref_mask = (ref_array == INUNDATED_VALUE)
    total_inundated_pixels = np.sum(ref_mask)
    pixel_area = abs(ref_profile['transform'][0] * ref_profile['transform'][4])  # m² per pixel

    stats = []

    for coarse_path in COARSE_PATHS:
        with rasterio.open(coarse_path) as coarse_dst:
            res = os.path.basename(coarse_path).split('.')[0]
            output_path = os.path.join(OUTPUT_DIR, f'classification_{res}.tif')

            # Resample coarse raster to match reference raster's grid
            resampled_coarse = np.empty_like(ref_array, dtype=np.float32)
            reproject(
                source=coarse_dst.read(1),
                destination=resampled_coarse,
                src_transform=coarse_dst.transform,
                src_crs=coarse_dst.crs,
                dst_transform=ref_profile['transform'],
                dst_crs=ref_profile['crs'],
                src_nodata=coarse_dst.nodata if coarse_dst.nodata is not None else -9999,
                dst_nodata=-9999,
                resampling=Resampling.nearest
            )

            # Mask valid pixels (only compare pixels that are inundated in reference)
            valid_mask = (ref_array == INUNDATED_VALUE) & (resampled_coarse == INUNDATED_VALUE)
            false_negatives = (ref_array == INUNDATED_VALUE) & (resampled_coarse != INUNDATED_VALUE)
            false_positives = (ref_array != INUNDATED_VALUE) & (resampled_coarse == INUNDATED_VALUE)

            # Classification raster
            classification = np.full_like(ref_array, NODATA_CLASS, dtype=np.uint8)
            classification[valid_mask] = 1  # True Positive
            classification[false_negatives] = 2  # False Negative
            classification[false_positives] = 3  # False Positive

            # Save classification raster
            profile = ref_profile.copy()
            profile.update(dtype=rasterio.uint8, nodata=NODATA_CLASS, count=1)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(classification, 1)

            # Calculate statistics
            tp_count = np.sum(valid_mask)
            fn_count = np.sum(false_negatives)
            fp_count = np.sum(false_positives)

            stats.append({
                'Resolution': res,
                'True Positives': tp_count,
                'False Negatives': fn_count,
                'False Positives': fp_count,
                'TP Area (m²)': tp_count * pixel_area,
                'FN Area (m²)': fn_count * pixel_area,
                'FP Area (m²)': fp_count * pixel_area,
                'TP (%)': (tp_count / total_inundated_pixels) * 100 if total_inundated_pixels > 0 else 0,
                'FN (%)': (fn_count / total_inundated_pixels) * 100 if total_inundated_pixels > 0 else 0,
                'FP (%)': (fp_count / total_inundated_pixels) * 100 if total_inundated_pixels > 0 else 0
            })

    # Write statistics to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'inundation_stats.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Resolution', 'True Positives', 'False Negatives', 'False Positives',
                      'TP Area (m²)', 'FN Area (m²)', 'FP Area (m²)',
                      'TP (%)', 'FN (%)', 'FP (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in stats:
            writer.writerow(entry)


if __name__ == '__main__':
    main()
