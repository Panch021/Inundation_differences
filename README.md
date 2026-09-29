# Inundation_differences

Python codes to quantify pixel-by-pixel inundation discrepancies between simulation results:

1. **NRMSE.py** – Normalized Root Mean Square Error (NRMSE) of Kestrel `Maximums.nc` runs against a reference run (max depth, speed, solid fraction, impact pressure, erosion and inundation time). Outputs a CSV and a radar chart.
2. **Inundation_detector.py** – Observed vs. predicted inundation (TP / FN / FP, precision, recall, F1) with comparison maps.

## 1. Installation

Create the conda environment (all packages from conda-forge):

```bash
conda create -n inundation_validation -c conda-forge python=3.11 numpy pandas xarray netcdf4 rasterio pyproj matplotlib dash geopandas shapely -y
```

Activate it:

```bash
conda activate inundation_validation
```

## 2. Working directory

Create a working directory and place the Python scripts and your data in it (e.g. the `Maximums.nc` files of each run and, for the Inundation detector, the observed layer: `.tif`, `.nc`, `.shp`, `.zip`, `.gpkg` or `.geojson`).
Relative paths typed in the dashboards are resolved from the folder that contains the scripts.

```bash
cd path/to/working_directory
```

## 3. Run NRMSE

Dashboard (open http://127.0.0.1:8040 in your browser):

```bash
python NRMSE.py
```

Command line, using a JSON configuration file:

```bash
python NRMSE.py --cli --config my_case.json
```

Example `my_case.json`:

```json
{
  "reference_path": "Maximums_CotoN_10m.nc",
  "prediction_paths": ["Maximums_CotoN_15m.nc", "Maximums_CotoN_20m.nc", "Maximums_CotoN_30m.nc"],
  "depth_threshold": 0.0,
  "mask_mode": "intersection",
  "normalization": "range",
  "chart_title": "Cotopaxi - NRMSE"
}
```

## 4. Run Inundation detector

Dashboard (open http://127.0.0.1:8040 in your browser):

```bash
python Inundation_detector.py
```

Command line, using a JSON configuration file:

```bash
python Inundation_detector.py --cli --config my_case.json
```

## 5. Running on a remote server

Start the script on the server with `--port`, then open an SSH tunnel from your computer and browse to the same port locally:

```bash
ssh -L 8040:127.0.0.1:8040 user@server
```
