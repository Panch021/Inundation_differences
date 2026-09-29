#!/usr/bin/env python3
"""
Inundation_detector.py — observed vs. predicted inundation comparison
=====================================================================

Compares ONE observed inundation footprint against ONE OR MORE model
predictions (e.g. Kestrel/LaharFlow runs with different topographies) and
reports how well each prediction reproduces the observed area.

Inputs
------
* Observed (the area that was really inundated):
    - GeoTIFF (.tif/.tiff)  any variable, band 1
    - NetCDF  (.nc)         any 2D variable (chosen in the dashboard /
                            OBSERVED_VARIABLE; default max_depth or the
                            first 2D variable)
    - Vector  (.shp, .zip with a shapefile, .gpkg, .geojson) — polygons
      are rasterised (1 = inundated, 0 = not inundated)
* Predictions (one or many, always compared against the observed):
    - GeoTIFF (.tif/.tiff)  any variable (depth, binary mask, ...), band 1
    - NetCDF  (.nc)         must be a Kestrel Maximums.nc -> max_depth

Every layer is converted to a BINARY MASK (1 = inundated, 0 = dry):
    raster/NetCDF : value > THRESHOLD  (one value for observed and
                    predictions; NaN / NoData / outside extent = 0)
    vector        : inside a polygon (rasterised, optional all_touched);
                    the threshold is not used
TP/FN/FP, precision, recall and F1 are computed inside the OBSERVED extent.
Masks are placed on one common grid covering observed + predictions
(CRS of the observed layer if it is projected; resolution of the observed
raster, or the finest prediction when the observed layer is a vector).

Outputs (in the output folder)
------------------------------
* inundation_metrics.csv — one row per prediction:
    comparison                 "observed vs. prediction"
    true_positive_pct          TP area / observed area x 100
    false_negative_pct         FN area / observed area x 100
    false_positive_pct         FP area / observed area x 100
    precision                  TP / (TP + FP)
    recall                     TP / (TP + FN)
    f1_score                   2 x P x R / (P + R)
    p95_max_depth_m            95th percentile of the prediction variable over
                               the inundated cells (value > THRESHOLD); NaN
                               for binary 0/1 rasters
    max_inundation_area_m2     predicted inundated area on the prediction's own
                               grid: n cells with value > THRESHOLD × cell area
                               (whole file, not clipped, not resampled)
    ratio_to_observed_area     max_inundation_area_m2 / observed area (observed
                               area = cells > THRESHOLD, or polygon area)
    + observed_area_m2, tp/fn/fp areas, grid resolution, thresholds
* classification_<label>.tif — 1 TP, 2 FN, 3 FP (0 = dry in both, NoData),
                               with an embedded colour table (blue/red/yellow)
* map_<label>.png            — one map per prediction
* comparison_maps.png / .pdf — all predictions stacked (panels A, B, ...)
* run_config.json            — settings used

Paths typed in the dashboard (or given in --config) are relative to the
folder that contains this script.

Usage
-----
    python Inundation_detector.py                 # dashboard (starts empty)
    python Inundation_detector.py --port 8070
    python Inundation_detector.py --cli           # uses USER CONFIG below
    python Inundation_detector.py --cli --config my_case.json

Remote server: ssh -L 8070:127.0.0.1:8070 user@server, then open
http://127.0.0.1:8070 locally.

Requirements: numpy, pandas, rasterio, xarray, netCDF4 (or h5netcdf),
matplotlib, dash; geopandas + shapely for vector observed layers.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import math
import os
import re
import sys
import tempfile
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from rasterio.warp import (Resampling, calculate_default_transform, reproject,
                           transform_bounds)
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import ListedColormap          # noqa: E402
from matplotlib.patches import FancyArrow, Patch, Polygon, Rectangle  # noqa: E402

try:
    import geopandas as gpd
except ImportError:  # vector observed layers will not be available
    gpd = None


# =============================================================================
# USER CONFIG  (used by --cli; the dashboard starts with these values too)
# =============================================================================
# The dashboard always starts empty; these values are only used by --cli.
OBSERVED_PATH = None                     # tif / nc / shp / zip / gpkg / geojson
OBSERVED_LABEL = None                    # None -> file name
OBSERVED_VARIABLE = None                 # NetCDF only (None -> auto)

PREDICTION_PATHS = []                    # tif (any variable) or Maximums.nc, e.g.
#   ["prediction_DSM_2cm.tif", "Gasca/DSM/Maximums.nc"]
PREDICTION_LABELS = None                 # e.g. ["DSM", "DTM", ...]; None -> auto
PREDICTION_NC_VARIABLE = "max_depth"     # variable read from Maximums.nc

THRESHOLD = 0.02                         # inundated if value > THRESHOLD, applied to
                                         # every raster/NetCDF layer (observed and
                                         # predictions); ignored for a vector observed
GRID_RESOLUTION = None                   # m; None -> observed raster / finest pred
MASK_RESAMPLING = "nearest"              # "nearest" | "max" (any wet sub-cell)
ALL_TOUCHED = False                      # rasterise vector: all touched cells
DEFAULT_EPSG = 32717                     # for NetCDF / vector without CRS

OUTPUT_DIR = None                        # None -> <observed folder>/Results_inundation
FLOW_DIRECTION_AZIMUTH = None            # deg from north (e.g. 100); None = no arrow
SHOW_METRICS_ON_MAP = True
MAP_MARGIN = 0.08                        # blank margin around the flooded area (fraction)
MAP_DPI = 300

HOST = "127.0.0.1"
PORT = 8070
MAX_GRID_CELLS = 300_000_000             # safety limit for the common grid
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RASTER_EXT = {".tif", ".tiff", ".gtiff", ".geotiff"}
NC_EXT = {".nc", ".nc4", ".cdf"}
VECTOR_EXT = {".shp", ".zip", ".gpkg", ".geojson", ".json"}
X_NAMES = ("x", "lon", "longitude", "easting", "X", "Lon", "Longitude")
Y_NAMES = ("y", "lat", "latitude", "northing", "Y", "Lat", "Latitude")

CLASS_COLORS = {1: "#1020f0", 2: "#ee1c1c", 3: "#f4f71c"}   # TP, FN, FP
CLASS_LABELS = {1: "True Positive\n(overlapping)",
                2: "False Negative\n(underestimation)",
                3: "False Positive\n(overestimation)"}
MAP_BG = "#a8a4a5"
FLOW_COLOR = "#2a1790"


# -----------------------------------------------------------------------------
# Configuration object
# -----------------------------------------------------------------------------
@dataclass
class Config:
    observed_path: str | None = OBSERVED_PATH
    prediction_paths: list = field(default_factory=lambda: list(PREDICTION_PATHS))
    prediction_labels: list | None = PREDICTION_LABELS
    observed_label: str | None = OBSERVED_LABEL
    observed_variable: str | None = OBSERVED_VARIABLE
    threshold: float = THRESHOLD
    grid_resolution: float | None = GRID_RESOLUTION
    mask_resampling: str = MASK_RESAMPLING
    all_touched: bool = ALL_TOUCHED
    default_epsg: int = DEFAULT_EPSG
    output_dir: str | None = OUTPUT_DIR
    flow_direction_azimuth: float | None = FLOW_DIRECTION_AZIMUTH
    show_metrics_on_map: bool = SHOW_METRICS_ON_MAP
    map_dpi: int = MAP_DPI


# -----------------------------------------------------------------------------
# Layer readers
# -----------------------------------------------------------------------------
@dataclass
class Layer:
    path: str
    fmt: str                      # "tif" | "nc" | "vector"
    crs: CRS | None
    data: np.ndarray | None = None
    transform: Affine | None = None
    gdf: object = None
    variable: str | None = None

    @property
    def is_raster(self) -> bool:
        return self.fmt in ("tif", "nc")

    @property
    def bounds(self):
        if self.is_raster:
            h, w = self.data.shape
            t = self.transform
            xs = [t.c, t.c + t.a * w]
            ys = [t.f, t.f + t.e * h]
            return min(xs), min(ys), max(xs), max(ys)
        return tuple(self.gdf.total_bounds)


def file_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in RASTER_EXT:
        return "tif"
    if ext in NC_EXT:
        return "nc"
    if ext in VECTOR_EXT:
        return "vector"
    raise ValueError(f"Unsupported file type '{ext}' ({Path(path).name}). "
                     "Use .tif, .nc, .shp (+.shx/.dbf/.prj), .zip, .gpkg or .geojson.")


def _to_crs(value) -> CRS | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, np.integer)):
            return CRS.from_epsg(int(value))
        s = str(value).strip()
        if s.isdigit():
            return CRS.from_epsg(int(s))
        return CRS.from_user_input(s.upper() if s.lower().startswith("epsg:") else s)
    except Exception:
        return None


def read_tif(path: str) -> Layer:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float64").filled(np.nan)
        arr[arr < -1e30] = np.nan
        return Layer(path, "tif", src.crs, arr, src.transform,
                     variable=src.descriptions[0] or "band1")


def _nc_2d_variables(ds: xr.Dataset) -> list[str]:
    out = []
    for name, da in ds.data_vars.items():
        dims = set(da.dims)
        if any(x in dims for x in X_NAMES) and any(y in dims for y in Y_NAMES):
            out.append(name)
    return out


def _nc_crs(ds: xr.Dataset, da: xr.DataArray, xname: str, default_epsg):
    for key in ("crs_epsg", "epsg", "EPSG", "crs", "spatial_ref", "crs_wkt"):
        if key in ds.attrs:
            c = _to_crs(ds.attrs[key])
            if c:
                return c, f"global attr '{key}'"
    gm = da.attrs.get("grid_mapping")
    for vname in [gm, "spatial_ref", "crs"]:
        if vname and vname in ds.variables:
            att = ds[vname].attrs
            for key in ("crs_wkt", "spatial_ref", "epsg_code", "epsg"):
                if key in att:
                    c = _to_crs(att[key])
                    if c:
                        return c, f"{vname}:{key}"
            try:
                import pyproj
                return CRS.from_wkt(pyproj.CRS.from_cf(att).to_wkt()), f"{vname} (CF)"
            except Exception:
                pass
    if xname.lower().startswith("lon"):
        return CRS.from_epsg(4326), "lon/lat coordinates"
    return CRS.from_epsg(int(default_epsg)), f"DEFAULT EPSG:{default_epsg} (no CRS in file)"


def read_nc(path: str, role: str, variable: str | None = None,
            default_epsg: int = DEFAULT_EPSG) -> Layer:
    ds = xr.open_dataset(path, mask_and_scale=True)
    try:
        vars2d = _nc_2d_variables(ds)
        if role == "prediction":
            if PREDICTION_NC_VARIABLE not in ds.data_vars:
                raise ValueError(
                    f"{Path(path).name}: prediction NetCDF files must be Kestrel "
                    f"Maximums.nc files with variable '{PREDICTION_NC_VARIABLE}' "
                    f"(found: {', '.join(vars2d) or 'none'}).")
            var = PREDICTION_NC_VARIABLE
        else:
            if variable and variable in ds.data_vars:
                var = variable
            elif PREDICTION_NC_VARIABLE in vars2d:
                var = PREDICTION_NC_VARIABLE
            elif vars2d:
                var = vars2d[0]
            else:
                raise ValueError(f"{Path(path).name}: no 2D (y, x) variable found.")
        da = ds[var]
        xname = next(d for d in da.dims if d in X_NAMES)
        yname = next(d for d in da.dims if d in Y_NAMES)
        extra = [d for d in da.dims if d not in (xname, yname)]
        if extra:                       # e.g. time -> "ever inundated"
            da = da.max(dim=extra, skipna=True)
        da = da.transpose(yname, xname)
        a = np.asarray(da.values, dtype="float64")
        fill = ds.attrs.get("_FillValue", da.attrs.get("_FillValue"))
        if fill is not None:
            try:
                a[np.isclose(a, float(fill), rtol=0, atol=1e-3)] = np.nan
            except (TypeError, ValueError):
                pass
        a[a < -1e30] = np.nan
        x = np.asarray(ds[xname].values, dtype="float64")
        y = np.asarray(ds[yname].values, dtype="float64")
        if x.size > 1 and x[1] < x[0]:
            a, x = a[:, ::-1], x[::-1]
        if y.size > 1 and y[1] > y[0]:           # Kestrel: south -> north
            a, y = a[::-1, :], y[::-1]
        dx = abs(float(np.mean(np.diff(x)))) if x.size > 1 else float(ds.attrs.get("deltaX", 1))
        dy = abs(float(np.mean(np.diff(y)))) if y.size > 1 else float(ds.attrs.get("deltaY", dx))
        transform = from_origin(x[0] - dx / 2, y[0] + dy / 2, dx, dy)
        crs, _src = _nc_crs(ds, da, xname, default_epsg)
        return Layer(path, "nc", crs, a, transform, variable=var)
    finally:
        ds.close()


def read_vector(path: str, default_epsg: int = DEFAULT_EPSG) -> Layer:
    if gpd is None:
        raise ImportError("geopandas is required for vector observed layers "
                          "(conda install -c conda-forge geopandas).")
    p = Path(path)
    gdf = gpd.read_file(f"zip://{p}" if p.suffix.lower() == ".zip" else p)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise ValueError(f"{p.name}: no geometries found.")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=int(default_epsg))
    return Layer(str(path), "vector", CRS.from_user_input(gdf.crs.to_wkt()), gdf=gdf)


def load_layer(path: str, role: str, variable=None, default_epsg=DEFAULT_EPSG) -> Layer:
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    fmt = file_format(path)
    if fmt == "tif":
        return read_tif(path)
    if fmt == "nc":
        return read_nc(path, role, variable, default_epsg)
    if role == "prediction":
        raise ValueError(f"{Path(path).name}: predictions must be .tif or Maximums.nc "
                         "(vector files are only accepted for the observed layer).")
    return read_vector(path, default_epsg)


def describe_file(path: str, role: str, default_epsg=DEFAULT_EPSG) -> dict:
    """Quick summary for the dashboard (format, variables, CRS, size)."""
    fmt = file_format(path)
    info = {"format": fmt, "variables": [], "summary": ""}
    if fmt == "nc":
        with xr.open_dataset(path) as ds:
            info["variables"] = _nc_2d_variables(ds)
            if role == "prediction" and PREDICTION_NC_VARIABLE not in ds.data_vars:
                raise ValueError(f"{Path(path).name}: not a Maximums.nc "
                                 f"(no '{PREDICTION_NC_VARIABLE}' variable).")
            sizes = ", ".join(f"{k}={v}" for k, v in ds.sizes.items() if k in X_NAMES + Y_NAMES)
            crs = next((str(ds.attrs[k]) for k in ("crs_epsg", "epsg") if k in ds.attrs), "?")
            info["summary"] = f"NetCDF · {sizes} · CRS {crs}"
    elif fmt == "tif":
        with rasterio.open(path) as src:
            info["summary"] = (f"GeoTIFF · {src.width}×{src.height} · "
                               f"{src.res[0]:g} m · {src.crs.to_string() if src.crs else 'no CRS'}")
        if role == "prediction":
            lyr = read_tif(path)
            if is_binary(lyr):
                info["warning"] = ("Binary raster (0/1): P95 max depth cannot be computed. "
                                   "To get it, load a raster with maximum flow depth "
                                   "values (m) or the Maximums.nc.")
            else:
                info["summary"] += f" · max {np.nanmax(lyr.data):.2f}"
    else:
        lyr = read_vector(path, default_epsg)
        types = ", ".join(sorted(set(lyr.gdf.geom_type)))
        info["summary"] = (f"Vector · {len(lyr.gdf)} features ({types}) · "
                           f"{lyr.gdf.crs.to_string() if lyr.gdf.crs else 'no CRS'}")
    return info


# -----------------------------------------------------------------------------
# Common grid and masks
# -----------------------------------------------------------------------------
@dataclass
class Grid:
    crs: CRS
    transform: Affine
    width: int
    height: int
    res: float
    eval_window: tuple            # (r0, r1, c0, c1): observed extent inside the grid

    @property
    def cell_area(self) -> float:
        return self.res * self.res

    @property
    def eval_slice(self):
        r0, r1, c0, c1 = self.eval_window
        return slice(r0, r1), slice(c0, c1)

    def eval_mask(self) -> np.ndarray:
        m = np.zeros((self.height, self.width), dtype=bool)
        m[self.eval_slice] = True
        return m

    def eval_bounds(self):
        r0, r1, c0, c1 = self.eval_window
        t = self.transform
        return (t.c + c0 * self.res, t.f - r1 * self.res,
                t.c + c1 * self.res, t.f - r0 * self.res)


def layer_bounds_in(lyr: Layer, crs: CRS):
    if not lyr.is_raster:
        return tuple(lyr.gdf.to_crs(crs.to_wkt()).total_bounds)
    if lyr.crs == crs:
        return lyr.bounds
    return transform_bounds(lyr.crs, crs, *lyr.bounds, densify_pts=21)


def layer_res_in(lyr: Layer, crs: CRS) -> float | None:
    if not lyr.is_raster:
        return None
    if lyr.crs == crs:
        return min(abs(lyr.transform.a), abs(lyr.transform.e))
    h, w = lyr.data.shape
    t, _, _ = calculate_default_transform(lyr.crs, crs, w, h, *lyr.bounds)
    return min(abs(t.a), abs(t.e))


def choose_crs(obs: Layer, preds: list[Layer], log) -> CRS:
    for lyr in [obs] + preds:
        if lyr.crs is not None and lyr.crs.is_projected:
            return lyr.crs
    b = transform_bounds(obs.crs, CRS.from_epsg(4326), *layer_bounds_in(obs, obs.crs))
    lon, lat = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    log(f"All layers are geographic -> using UTM EPSG:{epsg} for area calculations.")
    return CRS.from_epsg(epsg)


def build_grid(obs: Layer, preds: list[Layer], cfg: Config, log) -> Grid:
    """Grid covering observed + predictions (so maps show every flooded cell);
    the metrics are computed only inside the observed extent (eval_window)."""
    crs = choose_crs(obs, preds, log)
    res = cfg.grid_resolution
    if not res:
        res = layer_res_in(obs, crs) if obs.is_raster else min(layer_res_in(p, crs) for p in preds)
    res = float(res)
    # align to a raster's cells when possible (avoids a half-cell shift)
    anchor = next((l for l in [obs] + preds if l.is_raster and l.crs == crs), None)
    ox, oy = (anchor.transform.c, anchor.transform.f) if anchor else (0.0, 0.0)

    def snap(b):
        return (ox + math.floor((b[0] - ox) / res) * res, oy + math.floor((b[1] - oy) / res) * res,
                ox + math.ceil((b[2] - ox) / res) * res, oy + math.ceil((b[3] - oy) / res) * res)

    ob = snap(layer_bounds_in(obs, crs))
    boxes = [ob] + [snap(layer_bounds_in(p, crs)) for p in preds]
    minx = min(b[0] for b in boxes); miny = min(b[1] for b in boxes)
    maxx = max(b[2] for b in boxes); maxy = max(b[3] for b in boxes)
    width = int(round((maxx - minx) / res)); height = int(round((maxy - miny) / res))
    if width * height > MAX_GRID_CELLS:
        raise MemoryError(f"Common grid would be {width}×{height} cells at {res:g} m. "
                          "Set a coarser grid resolution.")
    win = (int(round((maxy - ob[3]) / res)), int(round((maxy - ob[1]) / res)),
           int(round((ob[0] - minx) / res)), int(round((ob[2] - minx) / res)))
    log(f"Grid: {width}×{height} cells, {res:g} m, {crs.to_string()}; metrics over the "
        f"observed extent ({win[3] - win[2]}×{win[1] - win[0]} cells).")
    return Grid(crs, from_origin(minx, maxy, res, res), width, height, res, win)


def layer_to_mask(lyr: Layer, grid: Grid, threshold: float, cfg: Config) -> np.ndarray:
    """Binary mask on the common grid: True = inundated."""
    shape = (grid.height, grid.width)
    if not lyr.is_raster:
        gdf = lyr.gdf.to_crs(grid.crs.to_wkt())
        shapes = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
        return rasterize(shapes, out_shape=shape, transform=grid.transform, fill=0,
                         all_touched=cfg.all_touched, dtype="uint8").astype(bool)
    src = (np.isfinite(lyr.data) & (lyr.data > threshold)).astype("uint8")
    dst = np.zeros(shape, dtype="uint8")
    src_res = layer_res_in(lyr, grid.crs)
    rs = (Resampling.max if (cfg.mask_resampling == "max" and src_res < grid.res * 0.999)
          else Resampling.nearest)
    reproject(source=src, destination=dst,
              src_transform=lyr.transform, src_crs=lyr.crs,
              dst_transform=grid.transform, dst_crs=grid.crs,
              src_nodata=None, dst_nodata=None, resampling=rs)
    return dst.astype(bool)


def native_wet_area(lyr: Layer, threshold: float) -> float:
    """Inundated area on the layer's OWN grid/geometry (no resampling, no clipping):
    raster -> number of cells with value > threshold × cell area;
    vector -> area of the dissolved polygons. NaN if it cannot be computed."""
    try:
        if lyr.is_raster:
            if lyr.crs is None or not lyr.crs.is_projected:
                return float("nan")
            n = np.count_nonzero(np.isfinite(lyr.data) & (lyr.data > threshold))
            return float(n * abs(lyr.transform.a * lyr.transform.e))
        g = lyr.gdf
        if g.crs is None or not g.crs.is_projected:
            g = g.to_crs(g.estimate_utm_crs())
        geom = g.geometry
        u = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
        return float(u.area)
    except Exception:
        return float("nan")


def is_binary(lyr: Layer) -> bool:
    if not lyr.is_raster:
        return True
    vals = lyr.data[np.isfinite(lyr.data)]
    return vals.size > 0 and bool(np.all(np.isin(vals, (0.0, 1.0))))


def layer_p95_value(lyr: Layer, threshold: float) -> float:
    """P95 of the prediction variable over inundated cells (NaN for binary rasters)."""
    if not lyr.is_raster or is_binary(lyr):
        return float("nan")
    vals = lyr.data[np.isfinite(lyr.data)]
    wet = vals[vals > threshold]
    return float(np.percentile(wet, 95)) if wet.size else float("nan")


MAX_DEPTH_TIP = ("binary raster (0/1): P95 max depth cannot be computed — load a raster with "
                 "maximum flow depth values in m (e.g. max_depth of Maximums.nc) "
                 "or the Maximums.nc itself")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def compare_masks(obs: np.ndarray, pred: np.ndarray, cell_area: float) -> dict:
    tp = int(np.count_nonzero(obs & pred))
    fn = int(np.count_nonzero(obs & ~pred))
    fp = int(np.count_nonzero(~obs & pred))
    n_obs = tp + fn
    pct = (lambda n: 100.0 * n / n_obs) if n_obs else (lambda n: float("nan"))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0
          else float("nan"))
    return {
        "true_positive_pct": pct(tp),
        "false_negative_pct": pct(fn),
        "false_positive_pct": pct(fp),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp_area_m2": tp * cell_area,
        "fn_area_m2": fn * cell_area,
        "fp_area_m2": fp * cell_area,
    }


def classify(obs: np.ndarray, pred: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """1 TP, 2 FN, 3 FP inside the observed extent (cells outside it are left 0)."""
    cls = np.zeros(obs.shape, dtype="uint8")
    cls[ev & obs & pred] = 1
    cls[ev & obs & ~pred] = 2
    cls[ev & ~obs & pred] = 3
    return cls


def _hex_rgba(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def write_classification(path: Path, cls: np.ndarray, grid: Grid):
    profile = dict(driver="GTiff", dtype="uint8", count=1, width=grid.width,
                   height=grid.height, crs=grid.crs, transform=grid.transform,
                   nodata=0, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(cls, 1)
        dst.update_tags(1, classes="1=True Positive, 2=False Negative, 3=False Positive")
        dst.set_band_description(1, "1 TP | 2 FN | 3 FP")
        cmap = {0: (0, 0, 0, 0)}
        cmap.update({k: _hex_rgba(CLASS_COLORS[k]) for k in (1, 2, 3)})
        dst.write_colormap(1, cmap)          # opens coloured in QGIS/ArcGIS


# -----------------------------------------------------------------------------
# Maps
# -----------------------------------------------------------------------------
def _nice(value: float) -> float:
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    for m in (1, 2, 2.5, 5, 10):
        if m * 10 ** exp >= value:
            return m * 10 ** exp
    return 10 ** (exp + 1)


def _nice_floor(value: float) -> float:
    exp = math.floor(math.log10(value))
    best = 10 ** exp
    for m in (1, 2, 5):
        if m * 10 ** exp <= value:
            best = m * 10 ** exp
    return best


def _union(classes):
    wet = np.zeros(classes[0].shape, dtype=bool)
    for c in classes:
        wet |= c > 0
    return wet


def view_extent(classes, grid: Grid, margin_frac=MAP_MARGIN):
    """Bounding box of every flooded cell + a blank margin (may go beyond the grid)."""
    wet = _union(classes)
    t, res = grid.transform, grid.res
    rows = np.flatnonzero(wet.any(axis=1)); cols = np.flatnonzero(wet.any(axis=0))
    if rows.size == 0:
        r0, r1, c0, c1 = 0, grid.height, 0, grid.width
    else:
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    x0, x1 = t.c + c0 * res, t.c + c1 * res
    y0, y1 = t.f - r1 * res, t.f - r0 * res
    m = max(margin_frac * max(x1 - x0, y1 - y0), 10 * res)
    return [x0 - m, x1 + m, y0 - m, y1 + m]


def wet_points(classes, grid: Grid, max_px=900):
    """Coarse centres of flooded blocks (x, y, half block size) for collision tests."""
    wet = _union(classes)
    f = max(1, int(math.ceil(max(wet.shape) / max_px)))
    h, w = wet.shape
    H, W = int(math.ceil(h / f)) * f, int(math.ceil(w / f)) * f
    pad = np.zeros((H, W), dtype=bool); pad[:h, :w] = wet
    blk = pad.reshape(H // f, f, W // f, f).any(axis=(1, 3))
    r, c = np.nonzero(blk)
    t, res = grid.transform, grid.res
    return (t.c + (c + 0.5) * f * res, t.f - (r + 0.5) * f * res, 0.5 * f * res)


def downsample_classes(cls: np.ndarray, max_px: int) -> np.ndarray:
    """Block-reduce a class raster keeping the most frequent wet class per block,
    so thin channels survive when the map has fewer pixels than the grid."""
    f = int(math.ceil(max(cls.shape) / max_px))
    if f <= 1:
        return cls
    h, w = cls.shape
    H, W = int(math.ceil(h / f)) * f, int(math.ceil(w / f)) * f
    pad = np.zeros((H, W), dtype=cls.dtype); pad[:h, :w] = cls
    blocks = pad.reshape(H // f, f, W // f, f)
    counts = np.stack([(blocks == k).sum(axis=(1, 3)) for k in (1, 2, 3)])
    out = (counts.argmax(axis=0) + 1).astype("uint8")
    out[counts.sum(axis=0) == 0] = 0
    return out


# --- decoration sizes (inches, before scaling) --------------------------------
_DECOR_IN = {"letter": (0.50, 0.30), "north": (0.40, 0.80), "metrics": (1.95, 0.26)}
_PREF = {   # preferred anchors (fx, fy) of the box inside the map, tried first
    "letter": [(1, 1), (0, 1), (1, 0), (0, 0)],
    "north": [(0, 1), (0, 0), (1, 1), (1, 0)],
    "scale": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "flow": [(0.5, 1), (0.5, 0), (0.3, 1), (0.7, 1), (0.3, 0), (0.7, 0)],
    "metrics": [(1, 0), (0, 0), (1, 1), (0, 1)],
}


def _decor_sizes(extent, k, s, cfg: Config):
    """Box size (data units) of every decoration; k = data units per inch."""
    W = extent[1] - extent[0]
    sizes = {n: (w * s * k, h * s * k) for n, (w, h) in _DECOR_IN.items()}
    L = _nice_floor(W / 4.5)
    sizes["scale"] = (L + 0.55 * s * k, 0.40 * s * k)
    if cfg.flow_direction_azimuth is not None:
        az = math.radians(float(cfg.flow_direction_azimuth))
        Lf = 1.8 * s
        sizes["flow"] = ((abs(math.sin(az)) * Lf + 0.42 * s) * k,
                         (abs(math.cos(az)) * Lf + 0.42 * s) * k)
    if not cfg.show_metrics_on_map:
        sizes.pop("metrics")
    return sizes, L


def place_decorations(extent, k, s, pts, cfg: Config):
    """Put letter, north arrow, scale bar, flow arrow and metrics where they do
    not touch any flooded cell (nor each other). Returns None if impossible."""
    x0, x1, y0, y1 = extent
    px, py, half = pts
    sizes, L = _decor_sizes(extent, k, s, cfg)
    inset, clear = 0.07 * k, 0.05 * k + half
    placed = {}
    sweep = [(f, 1) for f in np.linspace(0, 1, 21)] + [(f, 0) for f in np.linspace(0, 1, 21)] + \
            [(0, f) for f in np.linspace(0, 1, 11)] + [(1, f) for f in np.linspace(0, 1, 11)]
    for name in ("letter", "north", "scale", "flow", "metrics"):
        if name not in sizes:
            continue
        bw, bh = sizes[name]
        if bw > (x1 - x0) - 2 * inset or bh > (y1 - y0) - 2 * inset:
            return None
        ok = None
        for fx, fy in _PREF[name] + sweep:
            bx = x0 + inset + fx * ((x1 - x0) - 2 * inset - bw)
            by = y0 + inset + fy * ((y1 - y0) - 2 * inset - bh)
            box = (bx, by, bx + bw, by + bh)
            hit = np.any((px > box[0] - clear) & (px < box[2] + clear) &
                         (py > box[1] - clear) & (py < box[3] + clear))
            if hit:
                continue
            if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                   for b in placed.values()):
                continue
            ok = box
            break
        if ok is None:
            return None
        placed[name] = ok
    placed["_scale_len"] = L
    return placed


def _panel_geometry(aspect, n, ncols=None):
    if ncols is None:
        ncols = 1 if aspect <= 0.8 or n == 1 else (2 if aspect <= 1.6 else min(n, 4))
        ncols = min(ncols, n)
    gap = 0.10 if ncols == 1 else 0.45
    pw = (9.0 - 0.42 - 0.08 - (ncols - 1) * gap) / ncols
    ph = pw * aspect
    if math.ceil(n / ncols) * ph > 22:
        ph = 22 / math.ceil(n / ncols); pw = ph / aspect
    return ncols, pw, ph


def compute_layout(classes, grid: Grid, cfg: Config) -> dict:
    """Map extent (with blank margin) + decoration boxes shared by every map.
    The extent is enlarged until the decorations fit outside the flooded area."""
    ext = view_extent(classes, grid)
    pts = wet_points(classes, grid)
    n = len(classes)
    for _ in range(14):
        W, H = ext[1] - ext[0], ext[3] - ext[2]
        ncols, pw, ph = _panel_geometry(H / W, n)
        k = W / pw
        s = max(0.55, min(1.0, 0.36 * ph / 0.80))
        boxes = place_decorations(ext, k, s, pts, cfg)
        if boxes is not None:
            break
        dx, dy = 0.03 * W, 0.07 * H                      # grow, mostly vertically
        ext = [ext[0] - dx, ext[1] + dx, ext[2] - dy, ext[3] + dy]
    else:                                                # fallback: fixed corners
        boxes = None
    return {"extent": ext, "ncols": ncols, "pw": pw, "ph": ph, "k": k, "s": s,
            "boxes": boxes}


def _default_boxes(extent, k, s, cfg):
    x0, x1, y0, y1 = extent
    sizes, L = _decor_sizes(extent, k, s, cfg)
    e = 0.07 * k
    pos = {"letter": (1, 1), "north": (0, 1), "scale": (0, 0), "flow": (0.5, 1),
           "metrics": (1, 0)}
    out = {}
    for name, (bw, bh) in sizes.items():
        fx, fy = pos[name]
        bx = x0 + e + fx * (x1 - x0 - 2 * e - bw); by = y0 + e + fy * (y1 - y0 - 2 * e - bh)
        out[name] = (bx, by, bx + bw, by + bh)
    out["_scale_len"] = L
    return out


def _draw_north(ax, box, s):
    bx0, by0, bx1, by1 = box
    lab_h = 0.2 * (by1 - by0)
    sh = (by1 - by0) - lab_h
    cx, w = (bx0 + bx1) / 2, 0.36 * sh
    top, base = by0 + sh, by0
    notch = (cx, base + 0.28 * sh)
    ax.add_patch(Polygon([(cx, top), (cx - w, base), notch], closed=True,
                         fc="white", ec="black", lw=0.9, zorder=6))
    ax.add_patch(Polygon([(cx, top), (cx + w, base), notch], closed=True,
                         fc="#9a9a9a", ec="black", lw=0.9, zorder=6))
    ax.text(cx, by1, "N", ha="center", va="top", fontsize=9 * s + 1, fontweight="bold",
            zorder=6)


def _draw_scale(ax, box, L, s):
    bx0, by0, bx1, by1 = box
    bh = 0.28 * (by1 - by0)
    ax.add_patch(Rectangle((bx0, by0), L / 2, bh, fc="black", ec="black", lw=0.8, zorder=6))
    ax.add_patch(Rectangle((bx0 + L / 2, by0), L / 2, bh, fc="white", ec="black", lw=0.8,
                           zorder=6))
    unit, kk = ("km", 1000.0) if L >= 5000 else ("m", 1.0)
    fmt = lambda v: f"{v / kk:g}"
    for v, txt in ((0, "0"), (L / 2, fmt(L / 2)), (L, f"{fmt(L)} {unit}")):
        ax.text(bx0 + v, by0 + bh * 1.35, txt, ha="left" if v == L else "center",
                va="bottom", fontsize=9 * s + 0.5, zorder=6)


def _draw_flow(ax, box, azimuth, k, s):
    bx0, by0, bx1, by1 = box
    az = math.radians(azimuth)
    dx, dy = math.sin(az), math.cos(az)
    L = 1.8 * s * k
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    width = 0.22 * s * k
    ax.add_patch(FancyArrow(cx - dx * L / 2, cy - dy * L / 2, dx * L, dy * L,
                            width=width, head_width=width * 1.8,
                            head_length=min(L * 0.25, width * 1.6),
                            length_includes_head=True, fc=FLOW_COLOR, ec="none", zorder=6))
    ang = math.degrees(math.atan2(dy, dx))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    ax.text(cx - dx * L * 0.08, cy - dy * L * 0.08, "flow direction", rotation=ang,
            rotation_mode="anchor", ha="center", va="center", color="white",
            fontsize=8.5 * s, fontweight="bold", zorder=7)


def draw_panel(ax, cls, grid: Grid, layout, label, letter, metrics, cfg: Config, max_px):
    x0, x1, y0, y1 = layout["extent"]
    W, H = x1 - x0, y1 - y0
    k, s = layout["k"], layout["s"]
    ax.set_facecolor(MAP_BG)
    t = grid.transform
    gext = (t.c, t.c + grid.width * grid.res, t.f - grid.height * grid.res, t.f)
    cmap = ListedColormap([(0, 0, 0, 0)] + [CLASS_COLORS[i] for i in (1, 2, 3)])
    ax.imshow(downsample_classes(cls, max_px), cmap=cmap, vmin=-0.5, vmax=3.5,
              extent=gext, origin="upper", interpolation="nearest", zorder=2)
    step = _nice(max(W, H) / 7)
    ax.set_xticks(np.arange(math.ceil(x0 / step) * step, x1, step))
    ax.set_yticks(np.arange(math.ceil(y0 / step) * step, y1, step))
    ax.grid(True, color="#4d4d4d", ls=(0, (5, 4)), lw=0.5, zorder=3)
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    ax.set_ylabel(label, fontsize=12, fontweight="bold", labelpad=6)
    boxes = layout["boxes"] or _default_boxes(layout["extent"], k, s, cfg)
    b = boxes["letter"]
    ax.text((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, f"({letter})", ha="center", va="center",
            fontsize=14 * s, fontweight="bold", zorder=8)
    _draw_north(ax, boxes["north"], s)
    _draw_scale(ax, boxes["scale"], boxes["_scale_len"], s)
    if "flow" in boxes:
        _draw_flow(ax, boxes["flow"], float(cfg.flow_direction_azimuth), k, s)
    if "metrics" in boxes and metrics:
        b = boxes["metrics"]
        txt = (f"P {metrics['precision']:.2f} · R {metrics['recall']:.2f} · "
               f"F1 {metrics['f1_score']:.2f}")
        ax.text((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, txt, ha="center", va="center",
                fontsize=8 * s + 0.5, zorder=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))


def make_figure(classes, labels, metrics_list, grid: Grid, layout, cfg: Config, single=False):
    n = len(classes)
    ncols = 1 if single else layout["ncols"]
    pw, ph = layout["pw"], layout["ph"]
    nrows = math.ceil(n / ncols)
    left, right, top = 0.42, 0.08, 0.08
    gap = 0.10 if ncols == 1 else 0.45
    fig_w = left + ncols * pw + (ncols - 1) * gap + right
    present = [1, 2, 3]
    leg_cols = len(present) if fig_w >= 2.25 * len(present) else 2
    legend_h = 0.85 if leg_cols == len(present) else 1.45
    fig_h = top + nrows * ph + (nrows - 1) * 0.10 + legend_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    max_px = int(max(pw, ph) * cfg.map_dpi)
    for i, (cls, lab, met) in enumerate(zip(classes, labels, metrics_list)):
        r, c = divmod(i, ncols)
        ax = fig.add_axes([(left + c * (pw + gap)) / fig_w,
                           (fig_h - top - (r + 1) * ph - r * 0.10) / fig_h,
                           pw / fig_w, ph / fig_h])
        letter = chr(ord("A") + i) if i < 26 else str(i + 1)
        draw_panel(ax, cls, grid, layout, lab, letter, met, cfg, max_px)
    handles = [Patch(fc=CLASS_COLORS[k], ec="black", lw=0.8, label=CLASS_LABELS[k])
               for k in present]
    fig.legend(handles=handles, loc="center", ncol=leg_cols, frameon=False,
               bbox_to_anchor=(0.5, (legend_h * 0.5) / fig_h), handlelength=3.0,
               handleheight=2.0, prop={"size": 10, "weight": "bold"}, columnspacing=1.6)
    return fig


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "prediction"


def default_label(path: str, used: set | None = None) -> str:
    p = Path(path)
    lab = p.stem
    if lab.lower().startswith("maximums"):
        parent = p.parent.name
        if parent and not re.fullmatch(r"[0-9a-f]{32}", parent):
            lab = parent
    if used is not None:
        base, k = lab, 2
        while lab in used:
            lab = f"{base}_{k}"; k += 1
        used.add(lab)
    return lab


def resolve_path(p: str | None) -> str:
    """Relative paths are resolved from the folder that contains this script."""
    p = (p or "").strip().strip('"').strip("'")
    if not p:
        return ""
    q = Path(p).expanduser()
    return str(q if q.is_absolute() else (SCRIPT_DIR / q).resolve())


def run_comparison(cfg: Config, log=print, progress=None) -> dict:
    """progress(fraction 0-1, message) is called as the analysis advances."""
    if not cfg.observed_path:
        raise ValueError("Load an observed file first.")
    if not cfg.prediction_paths:
        raise ValueError("Add at least one prediction file.")
    n = len(cfg.prediction_paths)
    total = 4 + 3 * n + 2
    state = {"k": 0}

    def step(msg):
        state["k"] += 1
        if progress:
            progress(min(state["k"] / total, 0.99), msg)

    step("Reading observed layer")
    obs_path = resolve_path(cfg.observed_path)
    obs = load_layer(obs_path, "observed", cfg.observed_variable, cfg.default_epsg)
    obs_label = cfg.observed_label or Path(obs_path).stem
    log(f"Observed: {Path(obs_path).name} ({obs.fmt}"
        f"{', ' + obs.variable if obs.variable else ''}, {obs.crs.to_string()})")

    used = set()
    labels = list(cfg.prediction_labels or [])
    pred_paths = [resolve_path(p) for p in cfg.prediction_paths]
    labels = [(labels[i] if i < len(labels) and labels[i] else default_label(p, used))
              for i, p in enumerate(pred_paths)]
    preds = []
    for p, lab in zip(pred_paths, labels):
        step(f"Reading prediction '{lab}'")
        lyr = load_layer(p, "prediction", default_epsg=cfg.default_epsg)
        preds.append(lyr)
        log(f"Prediction '{lab}': {Path(p).name} ({lyr.fmt}, {lyr.variable}, "
            f"{lyr.crs.to_string()}, {abs(lyr.transform.a):g} m)")
        if lyr.fmt == "tif" and is_binary(lyr):
            log(f"  NOTE '{lab}': {MAX_DEPTH_TIP}.")

    step("Building common grid")
    grid = build_grid(obs, preds, cfg, log)
    ev = grid.eval_mask()
    step("Building observed mask")
    obs_mask = layer_to_mask(obs, grid, cfg.threshold, cfg) & ev
    obs_area = native_wet_area(obs, cfg.threshold)
    if not obs_area == obs_area:
        obs_area = float(obs_mask.sum() * grid.cell_area)
    log(f"Observed inundated area: {obs_area:,.0f} m² "
        f"({'polygon area' if not obs.is_raster else 'cells > threshold'})")
    if not obs_mask.any():
        log("WARNING: observed mask is empty — check threshold / variable / CRS.")

    out_dir = Path(resolve_path(cfg.output_dir)) if cfg.output_dir else (
        Path(obs_path).resolve().parent / "Results_inundation")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, classes, mets, written = [], [], [], []
    for lab, lyr in zip(labels, preds):
        step(f"Comparing '{lab}'")
        pm = layer_to_mask(lyr, grid, cfg.threshold, cfg)
        m = compare_masks(obs_mask, pm & ev, grid.cell_area)
        pred_area = native_wet_area(lyr, cfg.threshold)
        if not pred_area == pred_area:
            pred_area = float(pm.sum() * grid.cell_area)
        outside = float((pm & ~ev).sum() * grid.cell_area)
        cls = classify(obs_mask, pm, ev)
        tif = out_dir / f"classification_{_safe(lab)}.tif"
        write_classification(tif, cls, grid); written.append(tif)
        row = {"comparison": f"{obs_label} vs. {lab}", "observed": obs_label,
               "prediction": lab}
        row.update({k: m[k] for k in ("true_positive_pct", "false_negative_pct",
                                      "false_positive_pct", "precision", "recall", "f1_score")})
        row["p95_max_depth_m"] = layer_p95_value(lyr, cfg.threshold)
        row["max_inundation_area_m2"] = pred_area
        row["ratio_to_observed_area"] = pred_area / obs_area if obs_area else float("nan")
        row["observed_area_m2"] = obs_area
        row.update({k: m[k] for k in ("tp_area_m2", "fn_area_m2", "fp_area_m2")})
        row["predicted_area_outside_observed_extent_m2"] = outside
        row.update({"grid_resolution_m": grid.res, "threshold": cfg.threshold,
                    "max_depth_note": MAX_DEPTH_TIP if (lyr.fmt == "tif" and is_binary(lyr))
                    else "", "prediction_file": str(lyr.path)})
        rows.append(row); classes.append(cls); mets.append(m)
        log(f"  {lab}: TP {m['true_positive_pct']:.1f}% · FN {m['false_negative_pct']:.1f}% · "
            f"FP {m['false_positive_pct']:.1f}% · F1 {m['f1_score']:.3f} · "
            f"area {pred_area:,.0f} m²" + (f" ({outside:,.0f} m² outside observed extent)"
                                           if outside else ""))

    df = pd.DataFrame(rows)
    csv_path = out_dir / "inundation_metrics.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g"); written.append(csv_path)

    # maps: one layout (extent + decorations) shared by all maps
    step("Planning map layout")
    layout = compute_layout(classes, grid, cfg)
    for cls, lab, m in zip(classes, labels, mets):
        step(f"Drawing map '{lab}'")
        fig = make_figure([cls], [lab], [m], grid, layout, cfg, single=True)
        pth = out_dir / f"map_{_safe(lab)}.png"
        fig.savefig(pth, dpi=cfg.map_dpi); plt.close(fig); written.append(pth)
    step("Drawing comparison figure")
    fig = make_figure(classes, labels, mets, grid, layout, cfg)
    png = out_dir / "comparison_maps.png"; pdf = out_dir / "comparison_maps.pdf"
    fig.savefig(png, dpi=cfg.map_dpi); fig.savefig(pdf)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    written += [png, pdf]

    cfg_path = out_dir / "run_config.json"
    cfg_dump = asdict(cfg); cfg_dump["prediction_labels"] = labels
    cfg_dump["run_time"] = _dt.datetime.now().isoformat(timespec="seconds")
    cfg_path.write_text(json.dumps(cfg_dump, indent=2, default=str)); written.append(cfg_path)
    tifs = [str(p) for p in written if str(p).endswith(".tif")]
    step("Writing summary")
    log(f"Results written to {out_dir}")
    if progress:
        progress(1.0, "Done")
    return {"df": df, "out_dir": str(out_dir), "csv": str(csv_path), "png": str(png),
            "pdf": str(pdf), "files": [str(p) for p in written], "tifs": tifs,
            "preview_b64": base64.b64encode(buf.getvalue()).decode()}


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "inundation_detector_uploads"

CARD = {"background": "#ffffff", "border": "1px solid #dde1e6", "borderRadius": "10px",
        "padding": "16px 18px", "marginBottom": "14px"}
DROP = {"border": "2px dashed #8a94a6", "borderRadius": "8px", "padding": "18px",
        "textAlign": "center", "cursor": "pointer", "background": "#f6f8fb",
        "color": "#3b4252"}
LBL = {"fontWeight": 600, "fontSize": "13px", "marginTop": "8px", "display": "block"}
INP = {"width": "100%", "padding": "6px 8px", "border": "1px solid #c5cad3",
       "borderRadius": "6px", "boxSizing": "border-box"}
BTN = {"padding": "7px 14px", "border": "none", "borderRadius": "6px",
       "background": "#1f4e8c", "color": "white", "cursor": "pointer",
       "marginRight": "6px", "fontWeight": 600}
BTN2 = dict(BTN, background="#6b7280")


def _save_upload(contents, filenames) -> list[Path]:
    """Save a batch of uploaded files into its own folder; returns saved paths."""
    d = UPLOAD_ROOT / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for c, fn in zip(contents, filenames):
        data = base64.b64decode(c.split(",", 1)[1])
        p = d / Path(fn).name
        p.write_bytes(data)
        out.append(p)
    return out


def _observed_from_upload(paths: list[Path]) -> Path:
    shp = [p for p in paths if p.suffix.lower() == ".shp"]
    if shp:
        stem = shp[0].with_suffix("")
        missing = [e for e in (".shx", ".dbf") if not stem.with_suffix(e).exists()]
        if missing:
            raise ValueError("A shapefile needs its sidecar files: drop the .shp together "
                             f"with {', '.join(missing)} (and .prj), or a .zip.")
        return shp[0]
    ok = [p for p in paths if p.suffix.lower() in RASTER_EXT | NC_EXT | VECTOR_EXT]
    if not ok:
        raise ValueError("Unsupported file(s): " + ", ".join(p.name for p in paths))
    return ok[0]


def _clean_path(p: str | None) -> str:
    return resolve_path(p)


_SKIP_DIRS = {"__pycache__", ".git", ".idea", "renders", "Results_inundation", "node_modules"}


def _candidate_files(role: str, max_depth=4, limit=600) -> list[str]:
    """Files under the script folder offered as suggestions in the path boxes."""
    exts = (RASTER_EXT | NC_EXT | ({".shp", ".zip", ".gpkg", ".geojson"} if role == "observed"
                                   else set()))
    out = []
    base_depth = len(SCRIPT_DIR.parts)
    try:
        for root, dirs, files in os.walk(SCRIPT_DIR):
            rp = Path(root)
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
            if len(rp.parts) - base_depth >= max_depth:
                dirs[:] = []
            for f in sorted(files):
                ext = Path(f).suffix.lower()
                if ext not in exts:
                    continue
                if role == "prediction" and ext in NC_EXT and not f.lower().startswith("maximums"):
                    continue                      # timestep files are not valid predictions
                out.append(str((rp / f).relative_to(SCRIPT_DIR)))
                if len(out) >= limit:
                    return out
    except Exception:
        pass
    return out


# drag & drop reordering of the prediction boxes (HTML5 DnD, no extra packages)
DRAG_CSS_JS = """
<style>
.pred-slot .drag-handle { cursor: grab; user-select: none; }
.pred-slot.dragging { opacity: .45; }
.pred-slot.drop-before { box-shadow: 0 -3px 0 #1f4e8c; }
.pred-slot.drop-after  { box-shadow: 0 3px 0 #1f4e8c; }
</style>
<script>
(function () {
  let drag = null;
  const slotOf = el => (el && el.closest) ? el.closest('.pred-slot') : null;
  const clearMarks = () => document.querySelectorAll('.drop-before,.drop-after')
      .forEach(x => x.classList.remove('drop-before', 'drop-after'));
  const idx = el => JSON.parse(el.id).index;
  document.addEventListener('mousedown', e => {
    const h = e.target.closest && e.target.closest('.drag-handle');
    if (!h || e.target.closest('button')) return;
    const s = slotOf(h); if (s) s.setAttribute('draggable', 'true');
  }, true);
  document.addEventListener('mouseup', () => {
    document.querySelectorAll('.pred-slot[draggable="true"]')
      .forEach(s => s.setAttribute('draggable', 'false'));
  }, true);
  document.addEventListener('dragstart', e => {
    const s = slotOf(e.target);
    if (!s || s.getAttribute('draggable') !== 'true') return;
    drag = s; s.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    try { e.dataTransfer.setData('text/plain', 'slot'); } catch (_) {}
  }, true);
  document.addEventListener('dragend', () => {
    if (drag) { drag.classList.remove('dragging'); drag.setAttribute('draggable', 'false'); }
    drag = null; clearMarks();
  }, true);
  ['dragenter', 'dragleave'].forEach(ev => document.addEventListener(ev, e => {
    if (drag) { e.preventDefault(); e.stopPropagation(); }   // keep drop zones idle
  }, true));
  document.addEventListener('dragover', e => {
    if (!drag) return;                       // file drags go to the drop zones
    e.preventDefault(); e.stopPropagation();
    const t = slotOf(e.target); clearMarks();
    if (!t || t === drag) return;
    const r = t.getBoundingClientRect();
    t.classList.add((e.clientY - r.top) > r.height / 2 ? 'drop-after' : 'drop-before');
  }, true);
  document.addEventListener('drop', e => {
    if (!drag) return;
    e.preventDefault(); e.stopPropagation();
    const t = slotOf(e.target);
    if (t && t !== drag) {
      const r = t.getBoundingClientRect();
      const after = (e.clientY - r.top) > r.height / 2;
      const slots = [...document.querySelectorAll('#pred-slots .pred-slot')]
        .filter(x => x.style.display !== 'none')
        .sort((a, b) => (parseInt(a.style.order) || 0) - (parseInt(b.style.order) || 0));
      const order = slots.map(idx).filter(i => i !== idx(drag));
      let pos = order.indexOf(idx(t)); if (after) pos += 1;
      order.splice(pos, 0, idx(drag));
      window.dash_clientside.set_props('pred-order', {data: order});
    }
    clearMarks();
  }, true);
})();
</script>
"""


def build_app(cfg0: Config):
    import threading
    import time
    import traceback

    import dash
    from dash import (ALL, MATCH, Input, Output, Patch, State, ctx, dash_table, dcc,
                      html, no_update)

    app = dash.Dash(__name__, title="Inundation detector", suppress_callback_exceptions=True)
    app.index_string = app.index_string.replace("</head>", DRAG_CSS_JS + "</head>")
    JOBS: dict = {}

    SLOT = {"border": "1px solid #dde1e6", "borderRadius": "8px", "padding": "10px",
            "marginBottom": "10px", "background": "#fbfcfd"}
    DROP_SMALL = dict(DROP, padding="10px")
    SMALL = {"fontSize": "12px", "marginTop": "4px", "color": "#1f4e8c"}

    def pred_slot(i: int, pos: int):
        return html.Div(id={"type": "slot", "index": i}, className="pred-slot",
                        style=dict(SLOT, order=pos), children=[
            html.Div(className="drag-handle", title="Hold and drag to reorder",
                     style={"display": "flex", "justifyContent": "space-between",
                            "alignItems": "center", "marginBottom": "6px"}, children=[
                html.Span([html.Span("⠿ ", style={"color": "#8a94a6", "fontSize": "16px"}),
                           html.B(f"Prediction {pos + 1}", id={"type": "slot-title", "index": i},
                                  style={"fontSize": "13px"})]),
                html.Button("✕", id={"type": "slot-del", "index": i}, title="Remove",
                            style={"border": "none", "background": "none",
                                   "cursor": "pointer", "color": "#9aa1ad",
                                   "fontSize": "15px"})]),
            dcc.Upload(id={"type": "pred-upload", "index": i}, multiple=False,
                       style=DROP_SMALL, children=html.Div(
                           "Drop one .tif or Maximums.nc (for max_depth the .tif must "
                           "contain maximum flow depth in m)", style={"fontSize": "12.5px"})),
            html.Div(style={"display": "flex", "gap": "6px", "marginTop": "6px"}, children=[
                dcc.Input(id={"type": "pred-path", "index": i}, list="pred-files",
                          placeholder=f"…or path relative to {SCRIPT_DIR.name}/",
                          debounce=True, style=INP),
                html.Button("Load", id={"type": "pred-load", "index": i}, style=BTN)]),
            html.Div(id={"type": "pred-status", "index": i}, style=SMALL),
            dcc.Input(id={"type": "pred-label", "index": i}, placeholder="Label (e.g. DSM)",
                      style=dict(INP, marginTop="6px")),
            dcc.Store(id={"type": "pred-store", "index": i}),
        ])

    app.layout = html.Div(style={"fontFamily": "Inter, Helvetica, Arial, sans-serif",
                                 "background": "#eef1f5", "minHeight": "100vh",
                                 "padding": "18px"}, children=[
        html.Div(style={"maxWidth": "1280px", "margin": "0 auto"}, children=[
            html.H2("Inundation detector", style={"margin": "0 0 2px 0", "color": "#1f2d3d"}),
            html.Div("Observed vs. predicted inundation · TP / FN / FP, precision, recall, "
                     "F1 and comparison maps", style={"color": "#5b6573", "marginBottom": "14px"}),
            dcc.Store(id="obs-store"), dcc.Store(id="result-store"), dcc.Store(id="job-store"),
            dcc.Store(id="slot-count", data=1), dcc.Store(id="pred-order", data=[0]),
            html.Datalist(id="obs-files", children=[html.Option(value=f) for f in
                                                    _candidate_files("observed")]),
            html.Datalist(id="pred-files", children=[html.Option(value=f) for f in
                                                     _candidate_files("prediction")]),
            html.Div(f"Root folder for paths: {SCRIPT_DIR}",
                     style={"fontSize": "12px", "color": "#5b6573", "marginBottom": "10px"}),
            dcc.Interval(id="poll", interval=600, disabled=True),
            html.Div(style={"display": "grid", "gridTemplateColumns": "minmax(320px, 1fr) 2fr",
                            "gap": "14px"}, children=[
                # ---------------- left column: inputs ----------------
                html.Div([
                    html.Div(style=CARD, children=[
                        html.H4("1 · Observed inundation", style={"marginTop": 0}),
                        dcc.Upload(id="obs-upload", multiple=True, style=DROP, children=html.Div([
                            html.Div("⇩", style={"fontSize": "22px"}),
                            html.Div("Drop a .tif, .nc, .zip or a shapefile "
                                     "(.shp + .shx + .dbf + .prj together)",
                                     style={"fontSize": "13px"})])),
                        html.Div(style={"display": "flex", "gap": "6px", "marginTop": "8px"},
                                 children=[
                            dcc.Input(id="obs-path", list="obs-files",
                                      placeholder=f"…or path relative to {SCRIPT_DIR.name}/",
                                      style=INP, debounce=True),
                            html.Button("Load", id="obs-load", style=BTN)]),
                        html.Div(id="obs-status", style=SMALL),
                        html.Div(id="obs-var-wrap", style={"display": "none"}, children=[
                            html.Span("NetCDF variable", style=LBL),
                            dcc.Dropdown(id="obs-var", clearable=False)]),
                        html.Span("Observed label", style=LBL),
                        dcc.Input(id="obs-label", placeholder="Label (e.g. Observed)", style=INP),
                    ]),
                    html.Div(style=CARD, children=[
                        html.H4("2 · Predictions", style={"marginTop": 0}),
                        html.Div("Hold ⠿ and drag a box to change the order of the maps.",
                                 style={"fontSize": "12px", "color": "#5b6573",
                                        "marginBottom": "8px"}),
                        html.Div(id="pred-slots", children=[pred_slot(0, 0)],
                                 style={"display": "flex", "flexDirection": "column"}),
                        html.Button("+ Add prediction", id="pred-add",
                                    style=dict(BTN2, width="100%")),
                    ]),
                    html.Div(style=CARD, children=[
                        html.H4("3 · Settings", style={"marginTop": 0}),
                        html.Span("Threshold (inundated if value >)", style=LBL),
                        dcc.Input(id="thr", type="number", value=cfg0.threshold, style=INP),
                        html.Div(id="thr-hint", style={"fontSize": "11.5px", "color": "#5b6573",
                                                       "marginTop": "3px"},
                                 children="Applied to every raster / NetCDF input."),
                        html.Span("Grid resolution (m, blank = auto)", style=LBL),
                        dcc.Input(id="res", type="number", value=cfg0.grid_resolution, style=INP),
                        html.Span("Mask resampling (fine → coarse)", style=LBL),
                        dcc.Dropdown(id="resamp", value=cfg0.mask_resampling, clearable=False,
                                     options=[{"label": "nearest (cell centre)", "value": "nearest"},
                                              {"label": "max (any wet sub-cell)", "value": "max"}]),
                        dcc.Checklist(id="flags", style={"marginTop": "8px", "fontSize": "13px"},
                                      value=(["touched"] if cfg0.all_touched else []) +
                                            (["metrics"] if cfg0.show_metrics_on_map else []),
                                      options=[{"label": " rasterise vector with all_touched",
                                                "value": "touched"},
                                               {"label": " write P / R / F1 on maps",
                                                "value": "metrics"}]),
                        html.Span("Default EPSG (files without CRS)", style=LBL),
                        dcc.Input(id="epsg", type="number", value=cfg0.default_epsg, style=INP),
                        html.Span("Flow-direction arrow azimuth (°, blank = none)", style=LBL),
                        dcc.Input(id="flow-az", type="number", value=cfg0.flow_direction_azimuth,
                                  style=INP),
                        html.Span("Map DPI", style=LBL),
                        dcc.Input(id="dpi", type="number", value=cfg0.map_dpi, style=INP),
                        html.Span("Output folder (blank = <observed folder>/Results_inundation)",
                                  style=LBL),
                        dcc.Input(id="outdir", value=cfg0.output_dir or "", style=INP),
                        html.Button("▶ Run comparison", id="run", style=dict(
                            BTN, width="100%", marginTop="14px", padding="10px",
                            fontSize="15px", background="#0f7a4a")),
                    ]),
                ]),
                # ---------------- right column: results ----------------
                html.Div(style=CARD, children=[
                    html.H4("Results", style={"marginTop": 0}),
                    html.Div(id="run-status", style={"fontSize": "13px"}),
                    html.Div(id="progress-wrap", style={"display": "none"}, children=[
                        html.Div(style={"background": "#e3e7ee", "borderRadius": "6px",
                                        "height": "18px", "overflow": "hidden",
                                        "margin": "8px 0 4px 0"}, children=[
                            html.Div(id="progress-bar", style={
                                "width": "0%", "height": "100%", "background": "#0f7a4a",
                                "transition": "width 0.4s ease"})]),
                        html.Div(id="progress-text", style={"fontSize": "12.5px",
                                                            "color": "#3b4252"}),
                    ]),
                    html.Div(id="downloads", style={"display": "none", "margin": "8px 0"},
                             children=[
                        html.Button("⬇ CSV", id="dl-csv-btn", style=BTN),
                        html.Button("⬇ Maps PNG", id="dl-png-btn", style=BTN),
                        html.Button("⬇ Maps PDF", id="dl-pdf-btn", style=BTN),
                        html.Button("⬇ TP/FN/FP GeoTIFF (zip)", id="dl-tif-btn", style=BTN),
                        html.Button("⬇ All (zip)", id="dl-zip-btn", style=BTN2)]),
                    html.Div(id="res-notes", style={"fontSize": "12px", "color": "#9a5b00",
                                                    "margin": "6px 0"}),
                    dash_table.DataTable(
                        id="res-table", data=[], columns=[],
                        style_cell={"fontSize": "12px", "padding": "4px 6px"},
                        style_header={"fontWeight": 600, "background": "#f0f3f7",
                                      "whiteSpace": "normal", "height": "auto"},
                        style_table={"overflowX": "auto"}),
                    html.Img(id="map-img", style={"width": "100%", "marginTop": "12px"}),
                    html.Pre(id="log", style={"fontSize": "11px", "background": "#f6f8fb",
                                              "padding": "8px", "borderRadius": "6px",
                                              "whiteSpace": "pre-wrap", "marginTop": "10px"}),
                ]),
            ]),
            dcc.Download(id="dl-csv"), dcc.Download(id="dl-png"), dcc.Download(id="dl-pdf"),
            dcc.Download(id="dl-tif"), dcc.Download(id="dl-zip"),
        ])])

    # ---- observed -----------------------------------------------------------
    @app.callback(Output("obs-store", "data"), Output("obs-status", "children"),
                  Output("obs-var", "options"), Output("obs-var", "value"),
                  Output("obs-var-wrap", "style"), Output("obs-label", "value"),
                  Output("thr-hint", "children"),
                  Input("obs-upload", "contents"), Input("obs-load", "n_clicks"),
                  Input("obs-path", "value"),
                  State("obs-upload", "filename"), State("epsg", "value"),
                  prevent_initial_call=True)
    def load_observed(contents, _n, typed, filenames, epsg):
        path = resolve_path(typed)
        hint_all = "Applied to every raster / NetCDF input."
        try:
            if ctx.triggered_id == "obs-upload" and contents:
                path = str(_observed_from_upload(_save_upload(contents, filenames)))
            if not path:
                return None, "", [], None, {"display": "none"}, no_update, hint_all
            info = describe_file(path, "observed", epsg or DEFAULT_EPSG)
        except Exception as e:
            return None, f"⚠ {e}", [], None, {"display": "none"}, no_update, hint_all
        vars_ = info["variables"]
        val = (PREDICTION_NC_VARIABLE if PREDICTION_NC_VARIABLE in vars_
               else (vars_[0] if vars_ else None))
        style = {"display": "block"} if info["format"] == "nc" else {"display": "none"}
        hint = ("Observed is a vector (rasterised as 1 = inundated) → the threshold "
                "is applied to the predictions only." if info["format"] == "vector"
                else hint_all)
        return ({"path": path, "format": info["format"]},
                f"✓ {Path(path).name} — {info['summary']}",
                [{"label": v, "value": v} for v in vars_], val, style, Path(path).stem, hint)

    # ---- predictions: add a slot ---------------------------------------------
    @app.callback(Output("pred-slots", "children"), Output("slot-count", "data"),
                  Output("pred-order", "data"),
                  Input("pred-add", "n_clicks"), State("slot-count", "data"),
                  State("pred-order", "data"), prevent_initial_call=True)
    def add_slot(_n, count, order):
        p = Patch()
        p.append(pred_slot(count, len(order or [])))
        return p, count + 1, (order or []) + [count]

    # ---- predictions: remove a slot ----------------------------------------
    @app.callback(Output("pred-order", "data", allow_duplicate=True),
                  Input({"type": "slot-del", "index": ALL}, "n_clicks"),
                  State("pred-order", "data"), prevent_initial_call=True)
    def del_slot(_clicks, order):
        trig_val = ctx.triggered[0]["value"] if ctx.triggered else None
        if not ctx.triggered_id or not trig_val:
            return no_update
        i = ctx.triggered_id["index"]
        return [j for j in (order or []) if j != i]

    # ---- predictions: apply order (drag & drop / remove) --------------------
    @app.callback(Output({"type": "slot", "index": ALL}, "style"),
                  Output({"type": "slot-title", "index": ALL}, "children"),
                  Input("pred-order", "data"), prevent_initial_call=True)
    def apply_order(order):
        order = order or []
        ids = [o["id"]["index"] for o in ctx.outputs_list[0]]
        styles, titles = [], []
        for i in ids:
            if i in order:
                pos = order.index(i)
                styles.append(dict(SLOT, order=pos)); titles.append(f"Prediction {pos + 1}")
            else:
                styles.append(dict(SLOT, display="none")); titles.append("")
        return styles, titles

    # ---- predictions: load / remove one slot --------------------------------
    @app.callback(Output({"type": "pred-store", "index": MATCH}, "data"),
                  Output({"type": "pred-status", "index": MATCH}, "children"),
                  Output({"type": "pred-label", "index": MATCH}, "value"),
                  Input({"type": "pred-upload", "index": MATCH}, "contents"),
                  Input({"type": "pred-load", "index": MATCH}, "n_clicks"),
                  Input({"type": "pred-path", "index": MATCH}, "value"),
                  State({"type": "pred-upload", "index": MATCH}, "filename"),
                  State({"type": "pred-label", "index": MATCH}, "value"),
                  prevent_initial_call=True)
    def load_prediction(contents, _l, typed, filename, label):
        trig = ctx.triggered_id["type"] if ctx.triggered_id else None
        path = resolve_path(typed)
        try:
            if trig == "pred-upload" and contents:
                path = str(_save_upload([contents], [filename])[0])
            if not path:
                return None, "", no_update
            if not Path(path).exists():
                raise FileNotFoundError(f"not found: {path}")
            if file_format(path) == "vector":
                raise ValueError("predictions must be .tif or Maximums.nc")
            info = describe_file(path, "prediction")
        except Exception as e:
            return None, f"⚠ {e}", no_update
        new_label = label if label else default_label(path)
        status = [html.Div(f"✓ {Path(path).name} — {info['summary']}")]
        if info.get("warning"):
            status.append(html.Div(f"⚠ {info['warning']}", style={"color": "#9a5b00"}))
        return {"path": path}, status, new_label

    # ---- run (starts a worker thread) ---------------------------------------
    @app.callback(Output("job-store", "data"), Output("poll", "disabled"),
                  Output("run-status", "children"), Output("progress-wrap", "style"),
                  Output("run", "disabled"),
                  Output("res-table", "data"), Output("map-img", "src"),
                  Output("downloads", "style"), Output("log", "children"),
                  Input("run", "n_clicks"),
                  State("obs-store", "data"), State("obs-var", "value"),
                  State("obs-label", "value"),
                  State({"type": "pred-store", "index": ALL}, "data"),
                  State({"type": "pred-store", "index": ALL}, "id"),
                  State({"type": "pred-label", "index": ALL}, "value"),
                  State({"type": "pred-label", "index": ALL}, "id"),
                  State("pred-order", "data"),
                  State("thr", "value"), State("res", "value"), State("resamp", "value"),
                  State("flags", "value"), State("epsg", "value"), State("flow-az", "value"),
                  State("dpi", "value"), State("outdir", "value"),
                  prevent_initial_call=True)
    def start_run(_n, obs, obs_var, obs_label, pstores, pstore_ids, plabels, plabel_ids,
                  order, thr, res, resamp, flags, epsg, flow_az, dpi, outdir):
        hide = {"display": "none"}
        fail = lambda msg: (no_update, True, html.Span(msg, style={"color": "#b42318"}),
                            hide, False, [], None, hide, "")
        if not obs or not obs.get("path"):
            return fail("⚠ Load an observed file first.")
        st = {i["index"]: s for i, s in zip(pstore_ids, pstores)}
        lb = {i["index"]: l for i, l in zip(plabel_ids, plabels)}
        preds = [(st[i]["path"], (lb.get(i) or "").strip() or None)
                 for i in (order or []) if st.get(i) and st[i].get("path")]
        if not preds:
            return fail("⚠ Load at least one prediction.")
        obs_path = obs["path"]
        out = resolve_path(outdir) or None
        if out is None and str(obs_path).startswith(str(UPLOAD_ROOT)):
            out = str(SCRIPT_DIR / "Results_inundation")
        cfg = Config(observed_path=obs_path,
                     prediction_paths=[p for p, _ in preds],
                     prediction_labels=[l for _, l in preds],
                     observed_label=(obs_label or "").strip() or None,
                     observed_variable=obs_var,
                     threshold=float(thr if thr is not None else THRESHOLD),
                     grid_resolution=float(res) if res else None,
                     mask_resampling=resamp or "nearest",
                     all_touched="touched" in (flags or []),
                     default_epsg=int(epsg or DEFAULT_EPSG),
                     output_dir=out,
                     flow_direction_azimuth=float(flow_az) if flow_az not in (None, "") else None,
                     show_metrics_on_map="metrics" in (flags or []),
                     map_dpi=int(dpi or MAP_DPI))
        job_id = uuid.uuid4().hex
        job = {"pct": 0.0, "msg": "Starting…", "log": [], "done": False,
               "result": None, "error": None, "t0": time.time()}
        JOBS[job_id] = job

        def worker():
            def prog(f, msg):
                job["pct"], job["msg"] = f, msg
            try:
                job["result"] = run_comparison(cfg, log=job["log"].append, progress=prog)
            except Exception as e:
                job["error"] = str(e)
                job["log"].append(traceback.format_exc())
            finally:
                job["done"] = True

        threading.Thread(target=worker, daemon=True).start()
        return (job_id, False, "Running…", {"display": "block"}, True, [], None, hide, "")

    # ---- poll progress -------------------------------------------------------
    @app.callback(Output("progress-bar", "style"), Output("progress-text", "children"),
                  Output("log", "children", allow_duplicate=True),
                  Output("poll", "disabled", allow_duplicate=True),
                  Output("run", "disabled", allow_duplicate=True),
                  Output("run-status", "children", allow_duplicate=True),
                  Output("res-table", "data", allow_duplicate=True),
                  Output("res-table", "columns"),
                  Output("map-img", "src", allow_duplicate=True),
                  Output("result-store", "data"),
                  Output("downloads", "style", allow_duplicate=True),
                  Output("res-notes", "children"),
                  Input("poll", "n_intervals"), State("job-store", "data"),
                  prevent_initial_call=True)
    def poll(_n, job_id):
        job = JOBS.get(job_id)
        if not job:
            return (no_update,) * 3 + (True,) + (no_update,) * 8
        pct = 100 * job["pct"]
        el = int(time.time() - job["t0"])
        bar = {"width": f"{pct:.0f}%", "height": "100%", "transition": "width 0.4s ease",
               "background": "#b42318" if job["error"] else "#0f7a4a"}
        text = f"{pct:.0f} % — {job['msg']} · {el // 60:02d}:{el % 60:02d} elapsed"
        log = "\n".join(job["log"])
        if not job["done"]:
            return bar, text, log, False, True, no_update, no_update, no_update, \
                no_update, no_update, no_update, no_update
        if job["error"]:
            return (bar, f"Stopped at {pct:.0f} % — {job['msg']}", log, True, False,
                    html.Span(f"⚠ {job['error']}", style={"color": "#b42318"}),
                    [], [], None, None, {"display": "none"}, "")
        r = job["result"]
        show_cols = ["comparison", "true_positive_pct", "false_negative_pct",
                     "false_positive_pct", "precision", "recall", "f1_score", "p95_max_depth_m",
                     "max_inundation_area_m2", "ratio_to_observed_area"]
        names = {"comparison": "Observed vs. prediction", "true_positive_pct": "TP (%)",
                 "false_negative_pct": "FN (%)", "false_positive_pct": "FP (%)",
                 "precision": "Precision", "recall": "Recall", "f1_score": "F1 score",
                 "p95_max_depth_m": "P95 max depth (m)", "max_inundation_area_m2": "Max inund. area (m²)",
                 "ratio_to_observed_area": "Ratio to observed area"}
        df = r["df"][show_cols].copy()
        for c in show_cols[1:]:
            df[c] = df[c].astype(float).round(2)
        cols = [{"name": names[c], "id": c} if c == "comparison" else
             {"name": names[c], "id": c, "type": "numeric",
              "format": {"specifier": ",.2f"}} for c in show_cols]
        status = html.Span(f"✓ Done in {el // 60:02d}:{el % 60:02d} — results in {r['out_dir']}",
                           style={"color": "#0f7a4a"})
        store = {k: r[k] for k in ("csv", "png", "pdf", "files", "tifs", "out_dir")}
        notes = [html.Div(f"ℹ {row['prediction']}: {row['max_depth_note']}.")
                 for _, row in r["df"].iterrows() if row.get("max_depth_note")]
        JOBS.pop(job_id, None)
        return (bar, text, log, True, False, status, df.to_dict("records"), cols,
                "data:image/png;base64," + r["preview_b64"], store,
                {"display": "block", "margin": "8px 0"}, notes)

    # ---- downloads ----------------------------------------------------------
    for key in ("csv", "png", "pdf"):
        @app.callback(Output(f"dl-{key}", "data"), Input(f"dl-{key}-btn", "n_clicks"),
                      State("result-store", "data"), prevent_initial_call=True)
        def _dl(_n, store, key=key):
            return dcc.send_file(store[key]) if store else no_update

    def _zip(files, name):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, arcname=Path(f).name)
        return dcc.send_bytes(buf.getvalue(), name)

    @app.callback(Output("dl-tif", "data"), Input("dl-tif-btn", "n_clicks"),
                  State("result-store", "data"), prevent_initial_call=True)
    def dl_tif(_n, store):
        return _zip(store["tifs"], "classification_TP_FN_FP.zip") if store else no_update

    @app.callback(Output("dl-zip", "data"), Input("dl-zip-btn", "n_clicks"),
                  State("result-store", "data"), prevent_initial_call=True)
    def dl_zip(_n, store):
        return _zip(store["files"], "inundation_results.zip") if store else no_update

    return app


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", action="store_true", help="run without dashboard")
    ap.add_argument("--config", help="JSON file with Config fields (overrides USER CONFIG)")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    cfg = Config()
    if args.config:
        for k, v in json.loads(Path(args.config).read_text()).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.cli:
        if not cfg.observed_path or not cfg.prediction_paths:
            sys.exit("Set OBSERVED_PATH and PREDICTION_PATHS (USER CONFIG) or use --config.")
        base = Path(args.config).resolve().parent if args.config else SCRIPT_DIR
        fix = lambda p: str(p if Path(p).is_absolute() else base / p)
        cfg.observed_path = fix(cfg.observed_path)
        cfg.prediction_paths = [fix(p) for p in cfg.prediction_paths]
        r = run_comparison(cfg)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(r["df"].iloc[:, :13].to_string(index=False))
        return
    app = build_app(cfg)
    print(f"Inundation detector → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    sys.exit(main())
