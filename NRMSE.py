#!/usr/bin/env python3
"""
NRMSE.py — NRMSE of Kestrel Maximums.nc runs against a reference run
====================================================================

Compares ONE reference Kestrel run (its Maximums.nc) against ONE OR MORE
prediction runs (their Maximums.nc) and reports, for every variable, the
Normalised Root Mean Square Error (NRMSE, %) of each prediction. Results are
written to a CSV and drawn as a radar (spider) chart.

Variables (read from every Maximums.nc)
---------------------------------------
    max_depth         Max. depth             (m)
    max_speed         Max. speed             (m/s)
    max_solids_frac   Max. solid fraction    (-)
    impact_pressure   Max. impact pressure   (kPa)  <- computed, see below
    max_erosion       Max. erosion           (m)
    inundation_time   Inundation time        (s; -1 = never reached -> NaN)

Impact pressure is computed on each run's own grid from its maxima:

    P = (C (rho_s - rho_w) + rho_w) (g h + V^2) / 1000        [kPa]

    C = max_solids_frac, h = max_depth, V = max_speed,
    rho_s = 2000 kg/m3, rho_w = 1000 kg/m3, g = 9.81 m/s2 (editable)

(The three maxima are not necessarily simultaneous, so P is an upper bound —
the same convention as the impact-pressure maps.)

Mask
----
A cell is "inundated" in a run when max_depth > DEPTH_THRESHOLD (0.0 m by
default, i.e. any flow). The NRMSE of every variable is computed only over the cells chosen
by MASK_MODE:
    intersection  wet in the reference AND in the prediction (default)
    union         wet in the reference OR in the prediction (dry cells count
                  as 0 -> penalises missing and extra flooded area)
    reference     wet in the reference
Inundation time: negative values (-1 = never reached) are always discarded,
and it is compared only where BOTH runs have a valid time (>= 0). By default
the depth mask is applied to it too (TIME_DEPTH_MASK = True); set it to False
to compare every cell with a valid arrival time in both runs, regardless of
max_depth (Kestrel records inundation_time with its own heightThreshold).

Grid
----
Every run is resampled onto one common grid: the CRS and cell size of the
reference (or GRID_RESOLUTION), aligned to the reference cells and covering
the flooded area of all runs. Resampling method: nearest (default), bilinear,
average (recommended when a prediction is finer than the grid) or cubic.

NRMSE = RMSE / N x 100, with N computed from the REFERENCE values in the mask:
    1. range   max - min (default)
    2. p1p99   P99 - P1 (robust range: ignores the extreme 1 % at each end)
    3. p2p98   P98 - P2 (robust range: ignores the extreme 2 % at each end)
    4. iqr     P75 - P25 (central 50 %; unstable when most cells are ~0)
    5. mean    mean (inflated % for variables with mean ~0, e.g. erosion)
    6. std     standard deviation

Outputs (output folder, default <reference folder>/Results_NRMSE)
-----------------------------------------------------------------
* nrmse_results.csv  — long table: one row per prediction x variable
                       (nrmse_pct, rmse, mae, bias, pearson_r, n_cells,
                       normalisation factor, reference stats, settings)
* nrmse_matrix.csv   — wide table: predictions x variables (NRMSE %)
* radar_chart.png / .pdf / .svg
* run_config.json    — settings used

Usage
-----
    python NRMSE.py                    # dashboard  http://127.0.0.1:8040
    python NRMSE.py --port 8080
    python NRMSE.py --cli              # uses USER CONFIG below
    python NRMSE.py --cli --config my_case.json

Remote server: ssh -L 8040:127.0.0.1:8040 user@server, then open
http://127.0.0.1:8040 locally.

Requirements: numpy, pandas, rasterio, xarray, netCDF4 (or h5netcdf),
matplotlib, dash.
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
import rasterio  # noqa: F401  (GDAL environment for reproject)
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import is_color_like, to_hex    # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402


# =============================================================================
# USER CONFIG  (used by --cli; the dashboard starts empty with these settings)
# =============================================================================
REFERENCE_PATH = None                    # e.g. "Gasca/DSM/Maximums.nc"
REFERENCE_LABEL = None                   # None -> parent folder name
PREDICTION_PATHS = []                    # e.g. ["Gasca/DTM/Maximums.nc", ...]
PREDICTION_LABELS = None                 # e.g. ["DTM", "buildings"]; None -> auto
PREDICTION_COLORS = None                 # e.g. ["#1f3fd1", ...]; None -> palette

VARIABLES = ["max_depth", "max_speed", "max_solids_frac",
             "impact_pressure", "max_erosion", "inundation_time"]   # radar order
DEPTH_THRESHOLD = 0.0                    # m; wet if max_depth > threshold
TIME_DEPTH_MASK = True                   # apply the depth mask to inundation_time too
MASK_MODE = "intersection"               # "intersection" | "union" | "reference"
NORMALIZATION = "range"                  # "range" | "p1p99" | "p2p98" | "iqr" | "mean" | "std"
RESAMPLING = "nearest"                   # "nearest" | "bilinear" | "average" | "cubic"
GRID_RESOLUTION = None                   # m; None -> reference cell size
RHO_SOLID = 2000.0                       # kg/m3
RHO_WATER = 1000.0                       # kg/m3
GRAVITY = 9.81                           # m/s2
DEFAULT_EPSG = 32717                     # for NetCDF files without CRS

# radar chart
AXIS_MAX = None                          # % ; None -> auto (nice value above max)
TICK_STEP = None                         # % ; None -> auto (4-5 rings)
CENTER_OFFSET = 1.0                      # rings of empty space between the centre and 0 %
CHART_TITLE = ""
SHOW_LEGEND = True
CHART_DPI = 300

OUTPUT_DIR = None                        # None -> <reference folder>/Results_NRMSE
HOST = "127.0.0.1"
PORT = 8040
MAX_GRID_CELLS = 200_000_000             # safety limit for the common grid
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
NC_EXT = {".nc", ".nc4", ".cdf"}
X_NAMES = ("x", "lon", "longitude", "easting", "X")
Y_NAMES = ("y", "lat", "latitude", "northing", "Y")

# key -> (NetCDF variable or None if derived, axis label, table label, units)
VAR_INFO = {
    "max_depth":       ("max_depth",       "Max. depth",           "Max. depth",       "m"),
    "max_speed":       ("max_speed",       "Max. speed",           "Max. speed",       "m/s"),
    "max_solids_frac": ("max_solids_frac", "Max. solid\nfraction", "Max. solid frac.", "-"),
    "impact_pressure": (None,              "Max. impact pressure", "Max. impact P.",   "kPa"),
    "max_erosion":     ("max_erosion",     "Max. erosion",         "Max. erosion",     "m"),
    "inundation_time": ("inundation_time", "Inundation time",      "Inundation time",  "s"),
}
IP_INPUTS = ("max_depth", "max_speed", "max_solids_frac")
NORMALIZATIONS = {                       # dropdown order = recommended order
    "range": "max − min (default)",
    "p1p99": "P99 − P1 (robust range, ignores extreme 1 %)",
    "p2p98": "P98 − P2 (robust range, ignores extreme 2 %)",
    "iqr": "P75 − P25 (central 50 %; unstable if most cells ≈ 0)",
    "mean": "mean (large % when the mean is ≈ 0)",
    "std": "standard deviation",
}
RESAMPLING_MAP = {"bilinear": Resampling.bilinear, "nearest": Resampling.nearest,
                  "average": Resampling.average, "cubic": Resampling.cubic}
PALETTE = ["#1f3fd1", "#8fe36b", "#e3a33b", "#c0392b", "#8e44ad", "#17a2b8",
           "#6d4c41", "#e84393", "#2d3436", "#b8b814"]
# colour picker in the dashboard (rows of swatches, like a standard colour menu)
SWATCH_ROWS = [
    ("Default", PALETTE),
    ("Bright", ["#ff0000", "#ff8000", "#ffd700", "#7fff00", "#00b050",
                "#00bfff", "#0000ff", "#8000ff", "#ff00ff", "#8b4513"]),
    ("Standard", ["#000000", "#7f7f7f", "#a6cee3", "#1f78b4", "#b2df8a",
                  "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00"]),
]


def resolve_color(value, fallback: str) -> tuple[str, bool]:
    """Colour typed by the user (hex '#1f3fd1', name 'red', 'tab:blue', ...) ->
    (hex, ok). Blank or invalid -> (fallback, False)."""
    v = (value or "").strip()
    if v and not v.startswith("#") and re.fullmatch(r"[0-9a-fA-F]{6}", v):
        v = "#" + v                                    # 1f3fd1 -> #1f3fd1
    if v and is_color_like(v):
        return to_hex(v), True
    return fallback, False


# -----------------------------------------------------------------------------
# Configuration object
# -----------------------------------------------------------------------------
@dataclass
class Config:
    reference_path: str | None = REFERENCE_PATH
    reference_label: str | None = REFERENCE_LABEL
    prediction_paths: list = field(default_factory=lambda: list(PREDICTION_PATHS))
    prediction_labels: list | None = PREDICTION_LABELS
    prediction_colors: list | None = PREDICTION_COLORS
    variables: list = field(default_factory=lambda: list(VARIABLES))
    depth_threshold: float = DEPTH_THRESHOLD
    time_depth_mask: bool = TIME_DEPTH_MASK
    mask_mode: str = MASK_MODE
    normalization: str = NORMALIZATION
    resampling: str = RESAMPLING
    grid_resolution: float | None = GRID_RESOLUTION
    rho_solid: float = RHO_SOLID
    rho_water: float = RHO_WATER
    gravity: float = GRAVITY
    default_epsg: int = DEFAULT_EPSG
    axis_max: float | None = AXIS_MAX
    tick_step: float | None = TICK_STEP
    center_offset: float = CENTER_OFFSET
    chart_title: str = CHART_TITLE
    show_legend: bool = SHOW_LEGEND
    chart_dpi: int = CHART_DPI
    output_dir: str | None = OUTPUT_DIR


# -----------------------------------------------------------------------------
# Reading Maximums.nc
# -----------------------------------------------------------------------------
@dataclass
class Run:
    path: str
    label: str
    crs: CRS
    transform: Affine            # north-up, of the cropped arrays
    res: float
    data: dict                   # key -> 2D float32 array (cropped to flooded bbox)
    attrs: dict
    shape_full: tuple
    n_wet: int

    @property
    def bounds(self):
        h, w = self.data["max_depth"].shape
        t = self.transform
        return t.c, t.f + t.e * h, t.c + t.a * w, t.f


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


def _nc_crs(ds: xr.Dataset, da: xr.DataArray, default_epsg) -> tuple[CRS, str]:
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
    return CRS.from_epsg(int(default_epsg)), f"DEFAULT EPSG:{default_epsg} (no CRS in file)"


def _xy_names(da: xr.DataArray):
    xname = next((d for d in da.dims if d in X_NAMES), None)
    yname = next((d for d in da.dims if d in Y_NAMES), None)
    if xname is None or yname is None:
        raise ValueError(f"'{da.name}' has no (y, x) dimensions: {da.dims}")
    return xname, yname


def _clean(a: np.ndarray, fill) -> np.ndarray:
    a = np.asarray(a, dtype="float64")
    if fill is not None:
        try:
            a[np.isclose(a, float(fill), rtol=0, atol=1e-3)] = np.nan
        except (TypeError, ValueError):
            pass
    a[a < -1e30] = np.nan
    return a


def read_maximums(path: str, label: str, threshold: float,
                  default_epsg: int = DEFAULT_EPSG, pad: int = 2) -> Run:
    """Read a Kestrel Maximums.nc, cropped to the flooded area (max_depth > threshold)."""
    ds = xr.open_dataset(path, mask_and_scale=True)
    try:
        if "max_depth" not in ds.data_vars:
            raise ValueError(f"{Path(path).name}: not a Kestrel Maximums.nc "
                             "(variable 'max_depth' not found).")
        da = ds["max_depth"]
        xname, yname = _xy_names(da)
        fill = ds.attrs.get("_FillValue", da.attrs.get("_FillValue"))
        x = np.asarray(ds[xname].values, dtype="float64")
        y = np.asarray(ds[yname].values, dtype="float64")
        flip_x = x.size > 1 and x[1] < x[0]
        flip_y = y.size > 1 and y[1] > y[0]          # Kestrel: south -> north

        def grab(name, rows=None, cols=None):
            v = ds[name].transpose(yname, xname)
            if rows is not None:
                v = v.isel({yname: rows, xname: cols})
            a = _clean(v.values, fill)
            if flip_x:
                a = a[:, ::-1]
            if flip_y:
                a = a[::-1, :]
            return a

        if flip_x:
            x = x[::-1]
        if flip_y:
            y = y[::-1]
        depth = grab("max_depth")
        wet = np.isfinite(depth) & (depth > threshold)
        n_wet = int(wet.sum())
        if n_wet == 0:
            raise ValueError(f"{Path(path).name}: no cell with max_depth > {threshold:g} m.")
        keep = wet.copy()
        if "inundation_time" in ds.data_vars:          # keep every valid arrival time
            t = grab("inundation_time")
            keep |= np.isfinite(t) & (t >= 0)
            del t
        rr = np.flatnonzero(keep.any(axis=1)); cc = np.flatnonzero(keep.any(axis=0))
        r0, r1 = max(rr[0] - pad, 0), min(rr[-1] + pad + 1, depth.shape[0])
        c0, c1 = max(cc[0] - pad, 0), min(cc[-1] + pad + 1, depth.shape[1])
        # indices in the (possibly flipped) array -> original file indices
        ny, nx = depth.shape
        rows = (slice(ny - r1, ny - r0) if flip_y else slice(r0, r1))
        cols = (slice(nx - c1, nx - c0) if flip_x else slice(c0, c1))

        data = {"max_depth": depth[r0:r1, c0:c1].astype("float32")}
        for key, (ncvar, *_rest) in VAR_INFO.items():
            if ncvar and ncvar != "max_depth" and ncvar in ds.data_vars:
                data[key] = grab(ncvar, rows, cols).astype("float32")
        dx = abs(float(np.mean(np.diff(x)))) if x.size > 1 else float(ds.attrs.get("deltaX", 1))
        dy = abs(float(np.mean(np.diff(y)))) if y.size > 1 else float(ds.attrs.get("deltaY", dx))
        transform = from_origin(x[c0] - dx / 2, y[r0] + dy / 2, dx, dy)
        crs, _ = _nc_crs(ds, da, default_epsg)
        attrs = {k: ds.attrs[k] for k in ("solids density", "water density", "g",
                                          "deltaX", "crs_epsg") if k in ds.attrs}
        return Run(str(path), label, crs, transform, float(dx), data, attrs,
                   (ny, nx), n_wet)
    finally:
        ds.close()


def add_impact_pressure(run: Run, rho_s: float, rho_w: float, g: float) -> bool:
    """P = (C(rho_s - rho_w) + rho_w)(g h + V^2)/1000 [kPa], on the run's own grid."""
    if not all(k in run.data for k in IP_INPUTS):
        return False
    h = np.nan_to_num(run.data["max_depth"].astype("float64"), nan=0.0)
    v = np.nan_to_num(run.data["max_speed"].astype("float64"), nan=0.0)
    c = np.nan_to_num(run.data["max_solids_frac"].astype("float64"), nan=0.0)
    run.data["impact_pressure"] = ((c * (rho_s - rho_w) + rho_w) * (g * h + v * v)
                                   / 1000.0).astype("float32")
    return True


def describe_file(path: str, default_epsg=DEFAULT_EPSG) -> dict:
    """Quick summary for the dashboard."""
    if Path(path).suffix.lower() not in NC_EXT:
        raise ValueError(f"{Path(path).name}: load a Kestrel Maximums.nc file.")
    with xr.open_dataset(path) as ds:
        if "max_depth" not in ds.data_vars:
            raise ValueError(f"{Path(path).name}: not a Kestrel Maximums.nc "
                             "(no 'max_depth' variable).")
        da = ds["max_depth"]
        xname, yname = _xy_names(da)
        x = ds[xname].values
        res = abs(float(x[1] - x[0])) if x.size > 1 else float(ds.attrs.get("deltaX", float("nan")))
        crs, src = _nc_crs(ds, da, default_epsg)
        present = [k for k, (v, *_r) in VAR_INFO.items() if v and v in ds.data_vars]
        missing = [k for k, (v, *_r) in VAR_INFO.items() if v and v not in ds.data_vars]
        if all(k in present for k in IP_INPUTS):
            present.append("impact_pressure")
        summary = (f"{ds.sizes[xname]}×{ds.sizes[yname]} cells · {res:g} m · "
                   f"{crs.to_string()}")
    info = {"summary": summary, "present": present, "missing": missing}
    if missing:
        info["warning"] = "missing variables: " + ", ".join(missing)
    return info


# -----------------------------------------------------------------------------
# Common grid and resampling
# -----------------------------------------------------------------------------
@dataclass
class Grid:
    crs: CRS
    transform: Affine
    width: int
    height: int
    res: float


def build_grid(ref: Run, preds: list[Run], res: float | None, log) -> Grid:
    crs = ref.crs
    res = float(res or ref.res)
    ox, oy = ref.transform.c, ref.transform.f

    def bb(run):
        b = run.bounds
        return b if run.crs == crs else transform_bounds(run.crs, crs, *b, densify_pts=21)

    boxes = [bb(r) for r in [ref] + preds]
    minx = ox + math.floor((min(b[0] for b in boxes) - ox) / res) * res
    miny = oy + math.floor((min(b[1] for b in boxes) - oy) / res) * res
    maxx = ox + math.ceil((max(b[2] for b in boxes) - ox) / res) * res
    maxy = oy + math.ceil((max(b[3] for b in boxes) - oy) / res) * res
    width = int(round((maxx - minx) / res)); height = int(round((maxy - miny) / res))
    if width * height > MAX_GRID_CELLS:
        raise MemoryError(f"Common grid would be {width}×{height} cells at {res:g} m. "
                          "Set a coarser grid resolution.")
    log(f"Common grid: {width}×{height} cells, {res:g} m, {crs.to_string()} "
        "(flooded area of all runs, aligned to the reference cells).")
    return Grid(crs, from_origin(minx, maxy, res, res), width, height, res)


def resample(run: Run, key: str, grid: Grid, method: str) -> np.ndarray:
    """Variable `key` of `run` on the common grid (float64).
    Flow variables: NaN / outside the file = 0 (no flow).
    inundation_time: never reached (<0) / outside = NaN."""
    src = run.data[key].astype("float64")
    if key == "inundation_time":
        src = np.where(np.isfinite(src) & (src >= 0), src, np.nan)
        dst = np.full((grid.height, grid.width), np.nan)
        nodata = np.nan
    else:
        src = np.nan_to_num(src, nan=0.0)
        dst = np.zeros((grid.height, grid.width))
        nodata = None
    rs = RESAMPLING_MAP.get(method, Resampling.bilinear)
    if (method == "average" and run.res > grid.res * 1.001):
        rs = Resampling.bilinear             # average only makes sense fine -> coarse
    reproject(source=src, destination=dst,
              src_transform=run.transform, src_crs=run.crs,
              dst_transform=grid.transform, dst_crs=grid.crs,
              src_nodata=nodata, dst_nodata=nodata, resampling=rs)
    return dst


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def norm_factor(ref_vals: np.ndarray, method: str) -> float:
    if ref_vals.size == 0:
        return float("nan")
    if method == "range":
        return float(ref_vals.max() - ref_vals.min())
    if method == "mean":
        return float(np.mean(ref_vals))
    if method == "std":
        return float(np.std(ref_vals))
    if method == "iqr":
        q75, q25 = np.percentile(ref_vals, [75, 25])
        return float(q75 - q25)
    if method == "p1p99":
        p99, p1 = np.percentile(ref_vals, [99, 1])
        return float(p99 - p1)
    if method == "p2p98":
        p98, p2 = np.percentile(ref_vals, [98, 2])
        return float(p98 - p2)
    raise ValueError(f"Invalid normalisation '{method}'. Use: {', '.join(NORMALIZATIONS)}")


def compute_metrics(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray, method: str) -> dict:
    m = mask & np.isfinite(ref) & np.isfinite(pred)
    r, p = ref[m], pred[m]
    out = {"n_cells": int(m.sum())}
    if r.size == 0:
        out.update(nrmse_pct=float("nan"), rmse=float("nan"), mae=float("nan"),
                   bias=float("nan"), pearson_r=float("nan"), norm_factor=float("nan"),
                   ref_min=float("nan"), ref_max=float("nan"), ref_mean=float("nan"),
                   pred_mean=float("nan"))
        return out
    d = p - r
    rmse = float(np.sqrt(np.mean(d * d)))
    nf = norm_factor(r, method)
    with np.errstate(invalid="ignore", divide="ignore"):
        pr = (float(np.corrcoef(r, p)[0, 1]) if r.size > 1 and r.std() > 0 and p.std() > 0
              else float("nan"))
    out.update(nrmse_pct=(rmse / nf * 100.0) if nf and np.isfinite(nf) else float("nan"),
               rmse=rmse, mae=float(np.mean(np.abs(d))), bias=float(np.mean(d)),
               pearson_r=pr, norm_factor=nf, ref_min=float(r.min()), ref_max=float(r.max()),
               ref_mean=float(r.mean()), pred_mean=float(p.mean()))
    return out


# -----------------------------------------------------------------------------
# Radar chart
# -----------------------------------------------------------------------------
def _nice_step(vmax: float) -> float:
    raw = max(vmax, 1e-9) / 5.0
    exp = math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * 10 ** exp >= raw:
            return m * 10 ** exp
    return 10 ** (exp + 1)


def radar_axis(values: np.ndarray, axis_max=None, tick_step=None):
    finite = values[np.isfinite(values)]
    vmax = float(finite.max()) if finite.size else 1.0
    if axis_max:
        amax = float(axis_max)
        step = float(tick_step) if tick_step else _nice_step(amax)
    else:
        step = float(tick_step) if tick_step else _nice_step(vmax * 1.05)
        amax = step * max(1, math.ceil(vmax * 1.0001 / step))
        if vmax <= 0:
            amax = step
    ticks = list(np.arange(step, amax + step * 0.5, step))
    return amax, step, ticks


def _fmt_tick(v: float) -> str:
    return f"{v:g}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}"


def draw_radar(matrix: pd.DataFrame, colors: list[str], cfg: Config):
    """matrix: index = prediction labels, columns = variable keys (NRMSE %)."""
    keys = [k for k in matrix.columns if matrix[k].notna().any()]
    n = len(keys)
    if n < 3:
        raise ValueError("The radar chart needs at least 3 variables with results.")
    vals = matrix[keys].to_numpy(dtype=float)
    amax, step, ticks = radar_axis(vals, cfg.axis_max, cfg.tick_step)
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n          # clockwise from the top
    ux, uy = np.cos(ang), np.sin(ang)
    off = max(float(cfg.center_offset or 0), 0.0) * step     # empty core (in %)
    rad = lambda v: (np.asarray(v, dtype=float) + off) / (amax + off)
    r_zero = float(rad(0))

    fig = plt.figure(figsize=(8.0, 6.6 + (0.45 if cfg.show_legend else 0)))
    ax = fig.add_axes([0.08, 0.10 if cfg.show_legend else 0.04, 0.84, 0.84])
    ax.set_aspect("equal"); ax.axis("off")
    lim = 1.38
    ax.set_xlim(-lim, lim); ax.set_ylim(-1.2, 1.2)

    grid_kw = dict(color="black", lw=0.7, ls=(0, (1, 1.6)), zorder=1)
    for t in ([0.0] if r_zero > 0 else []) + ticks:         # polygonal rings
        r = float(rad(t))
        ax.plot(np.r_[ux, ux[0]] * r, np.r_[uy, uy[0]] * r, **grid_kw)
    for i in range(n):                                       # spokes (from the 0 ring)
        ax.plot([r_zero * ux[i], ux[i]], [r_zero * uy[i], uy[i]], **grid_kw)
    # tick labels along the first (top) spoke
    ax.text(-0.035 if r_zero > 0 else 0, r_zero + 0.012, "0",
            ha="right" if r_zero > 0 else "center", va="bottom", fontsize=11, zorder=5)
    for t in ticks:
        r = float(rad(t))
        lab = _fmt_tick(t) + ("%" if t == ticks[-1] else "")
        ax.text(-0.012 if t != ticks[-1] else 0.0, r + 0.012, lab,
                ha="center" if t == ticks[-1] else "right",
                va="bottom", fontsize=11, zorder=5)
    # axis labels
    for i, k in enumerate(keys):
        x, y = 1.1 * ux[i], 1.1 * uy[i]
        ha = "center" if abs(ux[i]) < 0.15 else ("left" if ux[i] > 0 else "right")
        va = "center" if abs(uy[i]) < 0.9 else ("bottom" if uy[i] > 0 else "top")
        if abs(ux[i]) >= 0.15 and abs(uy[i]) < 0.9:
            y += 0.02 * np.sign(uy[i])
        ax.text(x, y, VAR_INFO[k][1], ha=ha, va=va, fontsize=14, zorder=5)
    # series
    handles = []
    for j, (lab, row) in enumerate(zip(matrix.index, vals)):
        col = colors[j % len(colors)]
        r = rad(np.clip(row, 0, amax))
        xs, ys = np.r_[r * ux, r[0] * ux[0]], np.r_[r * uy, r[0] * uy[0]]
        ax.plot(xs, ys, color=col, lw=1.8, zorder=3 + j * 0.01, solid_joinstyle="round")
        ax.scatter(r * ux, r * uy, s=26, color=col, edgecolor="#333333", linewidth=0.6,
                   zorder=4 + j * 0.01)
        over = np.isfinite(row) & (row > amax)
        if over.any():
            ax.scatter(ux[over], uy[over], s=90, facecolor="none", edgecolor=col,
                       linewidth=1.2, zorder=4)
        handles.append(Line2D([0], [0], color=col, lw=1.8, marker="o", markersize=6,
                              markeredgecolor="#333333", label=str(lab)))
    if cfg.chart_title:
        fig.text(0.5, 0.975, cfg.chart_title, fontsize=14, ha="center", va="top",
                 fontweight="bold")
    if cfg.show_legend:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
                   frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.005),
                   title=None)
    return fig, amax


def save_radar(matrix: pd.DataFrame, colors, cfg: Config, out_dir: Path) -> dict:
    fig, amax = draw_radar(matrix, colors, cfg)
    paths = {ext: out_dir / f"radar_chart.{ext}" for ext in ("png", "pdf", "svg")}
    kw = dict(bbox_inches="tight", pad_inches=0.15)      # long labels never clipped
    fig.savefig(paths["png"], dpi=cfg.chart_dpi, **kw)
    fig.savefig(paths["pdf"], **kw); fig.savefig(paths["svg"], **kw)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, **kw); plt.close(fig)
    return {"png": str(paths["png"]), "pdf": str(paths["pdf"]), "svg": str(paths["svg"]),
            "axis_max": amax, "preview_b64": base64.b64encode(buf.getvalue()).decode()}


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "run"


def default_label(path: str, used: set | None = None) -> str:
    p = Path(path)
    lab = p.stem
    rest = re.sub(r"(?i)^maximums[_\-\s]*", "", lab)
    if rest and rest != lab:                       # Maximums_CotoN_15m -> CotoN_15m
        lab = rest
    elif lab.lower().startswith("maximums"):
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


def run_nrmse(cfg: Config, log=print, progress=None) -> dict:
    """progress(fraction 0-1, message) is called as the analysis advances."""
    if not cfg.reference_path:
        raise ValueError("Load the reference Maximums.nc first.")
    if not cfg.prediction_paths:
        raise ValueError("Add at least one prediction Maximums.nc.")
    keys = [k for k in VARIABLES if k in (cfg.variables or VARIABLES)]
    if not keys:
        raise ValueError("Select at least one variable.")
    if cfg.mask_mode not in ("intersection", "union", "reference"):
        raise ValueError(f"Invalid mask mode '{cfg.mask_mode}'.")
    norm_factor(np.array([0.0, 1.0]), cfg.normalization)       # validates the name
    thr = float(cfg.depth_threshold)
    n = len(cfg.prediction_paths)
    total = 3 + n + n * len(keys) + 3
    state = {"k": 0}

    def step(msg):
        state["k"] += 1
        if progress:
            progress(min(state["k"] / total, 0.99), msg)

    used = set()
    ref_path = resolve_path(cfg.reference_path)
    ref_label = (cfg.reference_label or "").strip() or default_label(ref_path, used)
    used.add(ref_label)
    pred_paths = [resolve_path(p) for p in cfg.prediction_paths]
    labels = list(cfg.prediction_labels or [])
    labels = [(labels[i].strip() if i < len(labels) and labels[i] and labels[i].strip()
               else default_label(p, used)) for i, p in enumerate(pred_paths)]
    colors_in = list(cfg.prediction_colors or [])
    colors = []
    for i in range(n):
        raw = colors_in[i] if i < len(colors_in) else None
        col, ok = resolve_color(raw, PALETTE[i % len(PALETTE)])
        if raw and not ok:
            log(f"WARNING: colour '{raw}' not recognised for prediction {i + 1} "
                f"→ using {col}")
        colors.append(col)

    def load(path, label):
        run = read_maximums(path, label, thr, cfg.default_epsg)
        if "impact_pressure" in keys and not add_impact_pressure(
                run, cfg.rho_solid, cfg.rho_water, cfg.gravity):
            log(f"  WARNING '{label}': impact pressure needs max_depth, max_speed and "
                "max_solids_frac.")
        fa = run.attrs
        diffs = [f"{name}={fa[k]:g}" for k, name, val in
                 (("solids density", "rho_s", cfg.rho_solid),
                  ("water density", "rho_w", cfg.rho_water), ("g", "g", cfg.gravity))
                 if k in fa and abs(float(fa[k]) - val) > 1e-6]
        log(f"'{label}': {Path(path).name} · {run.res:g} m · {run.crs.to_string()} · "
            f"{run.n_wet:,} wet cells (> {thr:g} m) · "
            f"{run.n_wet * run.res ** 2:,.0f} m²"
            + (f" · NOTE file attrs {', '.join(diffs)} differ from the IP constants"
               if diffs and "impact_pressure" in keys else ""))
        return run

    step("Reading reference")
    ref = load(ref_path, ref_label)
    preds = []
    for p, lab in zip(pred_paths, labels):
        step(f"Reading '{lab}'")
        preds.append(load(p, lab))

    step("Building common grid")
    grid = build_grid(ref, preds, cfg.grid_resolution, log)
    ref_on = {}
    ref_depth = resample(ref, "max_depth", grid, cfg.resampling)
    ref_wet = ref_depth > thr

    out_dir = Path(resolve_path(cfg.output_dir)) if cfg.output_dir else (
        Path(ref_path).resolve().parent / "Results_NRMSE")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Mask: max_depth > {thr:g} m · mode '{cfg.mask_mode}' · normalisation "
        f"'{cfg.normalization}' ({NORMALIZATIONS[cfg.normalization]}) · resampling "
        f"'{cfg.resampling}' · inundation time: "
        + ("depth mask + valid time in both" if cfg.time_depth_mask
           else "valid time in both (no depth mask)"))

    rows = []
    for lab, run in zip(labels, preds):
        pred_depth = resample(run, "max_depth", grid, cfg.resampling)
        pred_wet = pred_depth > thr
        mask = {"intersection": ref_wet & pred_wet, "union": ref_wet | pred_wet,
                "reference": ref_wet}[cfg.mask_mode]
        log(f"— {ref_label} vs. {lab}: mask {int(mask.sum()):,} cells "
            f"(ref wet {int(ref_wet.sum()):,}, pred wet {int(pred_wet.sum()):,}, "
            f"both {int((ref_wet & pred_wet).sum()):,})")
        for key in keys:
            step(f"{lab}: {VAR_INFO[key][2]}")
            if key not in ref.data or key not in run.data:
                log(f"   {key}: missing in {'reference' if key not in ref.data else lab} "
                    "→ skipped")
                met = compute_metrics(np.array([]), np.array([]), np.array([], bool),
                                      cfg.normalization)
            else:
                if key not in ref_on:
                    ref_on[key] = ref_depth if key == "max_depth" else resample(
                        ref, key, grid, cfg.resampling)
                pv = pred_depth if key == "max_depth" else resample(run, key, grid,
                                                                    cfg.resampling)
                if key == "inundation_time":
                    # -1 / never reached is NaN after resample -> only cells valid in both
                    m_use = mask if cfg.time_depth_mask else np.ones_like(mask)
                else:
                    m_use = mask
                met = compute_metrics(ref_on[key], pv, m_use, cfg.normalization)
                log(f"   {VAR_INFO[key][2]:<18} NRMSE {met['nrmse_pct']:7.2f} %  "
                    f"RMSE {met['rmse']:.4g} {VAR_INFO[key][3]}  bias {met['bias']:+.4g}  "
                    f"n={met['n_cells']:,}")
            rows.append({"reference": ref_label, "prediction": lab, "variable": key,
                         "variable_label": VAR_INFO[key][2], "units": VAR_INFO[key][3],
                         **met, "normalization": cfg.normalization,
                         "mask_mode": (cfg.mask_mode if key != "inundation_time"
                                       or cfg.time_depth_mask else "valid time in both"),
                         "depth_threshold_m": thr,
                         "resampling": cfg.resampling, "grid_resolution_m": grid.res,
                         "reference_file": ref_path, "prediction_file": run.path})

    step("Writing CSV")
    df = pd.DataFrame(rows)
    long_csv = out_dir / "nrmse_results.csv"
    df.to_csv(long_csv, index=False, float_format="%.6g")
    matrix = (df.pivot(index="prediction", columns="variable", values="nrmse_pct")
              .reindex(index=labels, columns=keys))
    wide = matrix.rename(columns={k: f"{k}_nrmse_pct" for k in keys})
    wide.insert(0, "reference", ref_label)
    wide_csv = out_dir / "nrmse_matrix.csv"
    wide.to_csv(wide_csv, index_label="prediction", float_format="%.4f")

    step("Drawing radar chart")
    chart = save_radar(matrix, colors, cfg, out_dir) if len(keys) >= 3 else None
    if chart is None:
        log("Radar chart skipped: select at least 3 variables.")

    cfg_path = out_dir / "run_config.json"
    dump = asdict(cfg)
    dump.update(reference_label=ref_label, prediction_labels=labels,
                prediction_colors=colors, reference_path=ref_path,
                prediction_paths=pred_paths,
                run_time=_dt.datetime.now().isoformat(timespec="seconds"))
    cfg_path.write_text(json.dumps(dump, indent=2, default=str))
    files = [long_csv, wide_csv, cfg_path] + (
        [Path(chart[e]) for e in ("png", "pdf", "svg")] if chart else [])
    log(f"Results written to {out_dir}")
    if progress:
        progress(1.0, "Done")
    return {"df": df, "matrix": matrix, "labels": labels, "colors": colors,
            "reference_label": ref_label, "out_dir": str(out_dir),
            "csv": str(long_csv), "matrix_csv": str(wide_csv),
            "png": chart["png"] if chart else None, "pdf": chart["pdf"] if chart else None,
            "svg": chart["svg"] if chart else None,
            "preview_b64": chart["preview_b64"] if chart else None,
            "files": [str(f) for f in files]}


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "nrmse_uploads"

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
HINT = {"fontSize": "11.5px", "color": "#5b6573", "marginTop": "3px"}


def _save_upload(contents, filename) -> Path:
    d = UPLOAD_ROOT / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    p = d / Path(filename).name
    p.write_bytes(base64.b64decode(contents.split(",", 1)[1]))
    return p


_SKIP_DIRS = {"__pycache__", ".git", ".idea", "renders", "Results_NRMSE", "node_modules",
              "backup"}


def _candidate_files(max_depth=4, limit=600) -> list[str]:
    """Maximums*.nc files under the script folder, offered in the path boxes."""
    out = []
    base_depth = len(SCRIPT_DIR.parts)
    try:
        for root, dirs, files in os.walk(SCRIPT_DIR):
            rp = Path(root)
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
            if len(rp.parts) - base_depth >= max_depth:
                dirs[:] = []
            for f in sorted(files):
                if Path(f).suffix.lower() in NC_EXT and f.lower().startswith("maximums"):
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
    if (drag) { e.preventDefault(); e.stopPropagation(); }
  }, true));
  document.addEventListener('dragover', e => {
    if (!drag) return;
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


def _num(v, default=None):
    try:
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def build_app(cfg0: Config):
    import threading
    import time
    import traceback

    import dash
    from dash import (ALL, MATCH, Input, Output, Patch, State, ctx, dcc,
                      html, no_update)

    app = dash.Dash(__name__, title="NRMSE · Kestrel", suppress_callback_exceptions=True)
    app.index_string = app.index_string.replace("</head>", DRAG_CSS_JS + "</head>")
    JOBS: dict = {}

    SLOT = {"border": "1px solid #dde1e6", "borderRadius": "8px", "padding": "10px",
            "marginBottom": "10px", "background": "#fbfcfd"}
    DROP_SMALL = dict(DROP, padding="10px")
    SMALL = {"fontSize": "12px", "marginTop": "4px", "color": "#1f4e8c"}
    files = _candidate_files()

    SWATCH = {"width": "18px", "height": "18px", "padding": 0, "cursor": "pointer",
              "border": "1px solid #9aa1ad", "borderRadius": "3px"}

    def preview_style(col, ok=True):
        return {"width": "30px", "minWidth": "30px", "borderRadius": "6px",
                "border": "1px solid #c5cad3" if ok else "2px solid #b42318",
                "background": col}

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
                           "Drop a Maximums.nc", style={"fontSize": "12.5px"})),
            html.Div(style={"display": "flex", "gap": "6px", "marginTop": "6px"}, children=[
                dcc.Input(id={"type": "pred-path", "index": i}, list="nc-files",
                          placeholder=f"…or path relative to {SCRIPT_DIR.name}/",
                          debounce=True, style=INP),
                html.Button("Load", id={"type": "pred-load", "index": i}, style=BTN)]),
            html.Div(id={"type": "pred-status", "index": i}, style=SMALL),
            html.Div(style={"display": "flex", "gap": "6px", "marginTop": "6px"}, children=[
                dcc.Input(id={"type": "pred-label", "index": i}, placeholder="Label (e.g. 15 m)",
                          style=dict(INP, flex="3")),
                dcc.Input(id={"type": "pred-color", "index": i},
                          placeholder="colour: red, blue, #1f3fd1…", debounce=True,
                          style=dict(INP, flex="2")),
                html.Div(id={"type": "color-preview", "index": i},
                         title="Colour used in the chart",
                         style=preview_style(PALETTE[pos % len(PALETTE)]))]),
            html.Details(style={"marginTop": "4px"}, children=[
                html.Summary("Pick a colour", style={"fontSize": "12px", "cursor": "pointer",
                                                     "color": "#1f4e8c"}),
                html.Div(style={"marginTop": "4px"}, children=[
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "3px", "marginBottom": "3px"}, children=[
                        html.Span(name, style={"fontSize": "10.5px", "color": "#5b6573",
                                               "width": "52px"})] + [
                        html.Button("", id={"type": "swatch", "index": i, "c": col},
                                    n_clicks=0, title=col, style=dict(SWATCH, background=col))
                        for col in cols])
                    for name, cols in SWATCH_ROWS])]),
            dcc.Store(id={"type": "pred-store", "index": i}),
        ])

    var_opts = [{"label": " " + VAR_INFO[k][2] + ("  (computed)" if k == "impact_pressure"
                                                    else ""), "value": k} for k in VARIABLES]

    app.layout = html.Div(style={"fontFamily": "Inter, Helvetica, Arial, sans-serif",
                                 "background": "#eef1f5", "minHeight": "100vh",
                                 "padding": "18px"}, children=[
        html.Div(style={"maxWidth": "1320px", "margin": "0 auto"}, children=[
            html.H2("NRMSE · Kestrel Maximums.nc",
                    style={"margin": "0 0 2px 0", "color": "#1f2d3d"}),
            html.Div("Reference run vs. prediction runs · NRMSE (%) per variable, "
                     "CSV and radar chart", style={"color": "#5b6573", "marginBottom": "14px"}),
            dcc.Store(id="ref-store"), dcc.Store(id="result-store"), dcc.Store(id="job-store"),
            dcc.Store(id="slot-count", data=1), dcc.Store(id="pred-order", data=[0]),
            html.Datalist(id="nc-files", children=[html.Option(value=f) for f in files]),
            html.Div(f"Root folder for paths: {SCRIPT_DIR}",
                     style={"fontSize": "12px", "color": "#5b6573", "marginBottom": "10px"}),
            dcc.Interval(id="poll", interval=600, disabled=True),
            html.Div(style={"display": "grid", "gridTemplateColumns": "minmax(330px, 1fr) 2fr",
                            "gap": "14px"}, children=[
                # ---------------- left column: inputs ----------------
                html.Div([
                    html.Div(style=CARD, children=[
                        html.H4("1 · Reference", style={"marginTop": 0}),
                        dcc.Upload(id="ref-upload", multiple=False, style=DROP, children=html.Div([
                            html.Div("⇩", style={"fontSize": "22px"}),
                            html.Div("Drop the reference Maximums.nc",
                                     style={"fontSize": "13px"})])),
                        html.Div(style={"display": "flex", "gap": "6px", "marginTop": "8px"},
                                 children=[
                            dcc.Input(id="ref-path", list="nc-files",
                                      placeholder=f"…or path relative to {SCRIPT_DIR.name}/",
                                      style=INP, debounce=True),
                            html.Button("Load", id="ref-load", style=BTN)]),
                        html.Div(id="ref-status", style=SMALL),
                        html.Span("Reference label", style=LBL),
                        dcc.Input(id="ref-label", placeholder="Label (e.g. 10 m)", style=INP),
                    ]),
                    html.Div(style=CARD, children=[
                        html.H4("2 · Predictions", style={"marginTop": 0}),
                        html.Div("Hold ⠿ and drag a box to change the order (legend and "
                                 "table follow it).", style={"fontSize": "12px",
                                                             "color": "#5b6573",
                                                             "marginBottom": "8px"}),
                        html.Div(id="pred-slots", children=[pred_slot(0, 0)],
                                 style={"display": "flex", "flexDirection": "column"}),
                        html.Button("+ Add prediction", id="pred-add",
                                    style=dict(BTN2, width="100%")),
                    ]),
                    html.Div(style=CARD, children=[
                        html.H4("3 · Settings", style={"marginTop": 0}),
                        html.Span("Variables (radar order, clockwise from the top)", style=LBL),
                        dcc.Checklist(id="vars", value=list(cfg0.variables), options=var_opts,
                                      style={"fontSize": "13px"},
                                      inputStyle={"marginRight": "4px"}),
                        html.Span("Depth threshold for the mask (m)", style=LBL),
                        dcc.Input(id="thr", type="number", value=cfg0.depth_threshold,
                                  min=0, step="any", style=INP),
                        html.Div("Wet cell = max_depth > threshold; applied to every variable.",
                                 style=HINT),
                        dcc.Checklist(id="tmask", value=["on"] if cfg0.time_depth_mask else [],
                                      options=[{"label": " apply the depth mask to inundation "
                                                         "time", "value": "on"}],
                                      style={"marginTop": "6px", "fontSize": "13px"}),
                        html.Div("Inundation time always discards −1 (never reached) and is "
                                 "compared only where both runs have a valid time. Unchecked: "
                                 "every cell with a valid time in both runs, whatever max_depth.",
                                 style=HINT),
                        html.Span("Mask mode", style=LBL),
                        dcc.Dropdown(id="mask", value=cfg0.mask_mode, clearable=False, options=[
                            {"label": "intersection — wet in reference AND prediction",
                             "value": "intersection"},
                            {"label": "union — wet in reference OR prediction",
                             "value": "union"},
                            {"label": "reference — wet in reference", "value": "reference"}]),
                        html.Span("Normalisation", style=LBL),
                        dcc.Dropdown(id="norm", value=cfg0.normalization, clearable=False,
                                     options=[{"label": f"{k} — {v}", "value": k}
                                              for k, v in NORMALIZATIONS.items()]),
                        html.Span("Resampling onto the reference grid", style=LBL),
                        dcc.Dropdown(id="resamp", value=cfg0.resampling, clearable=False,
                                     options=[{"label": k, "value": k} for k in RESAMPLING_MAP]),
                        html.Span("Grid resolution (m, blank = reference)", style=LBL),
                        dcc.Input(id="res", type="number", value=cfg0.grid_resolution, style=INP),
                        html.Span("Impact pressure constants", style=LBL),
                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr",
                                        "gap": "6px"}, children=[
                            html.Div([html.Div("ρ solid (kg/m³)", style=HINT),
                                      dcc.Input(id="rhos", inputMode="decimal", value=cfg0.rho_solid,
                                                style=INP)]),
                            html.Div([html.Div("ρ water (kg/m³)", style=HINT),
                                      dcc.Input(id="rhow", inputMode="decimal", value=cfg0.rho_water,
                                                style=INP)]),
                            html.Div([html.Div("g (m/s²)", style=HINT),
                                      dcc.Input(id="grav", inputMode="decimal", value=cfg0.gravity,
                                                style=INP)])]),
                        html.Div("P = (C(ρs−ρw)+ρw)(g·h+V²)/1000 kPa, from max_solids_frac, "
                                 "max_depth and max_speed.", style=HINT),
                        html.Span("Default EPSG (files without CRS)", style=LBL),
                        dcc.Input(id="epsg", type="number", value=cfg0.default_epsg, style=INP),
                        html.Span("Output folder (blank = <reference folder>/Results_NRMSE)",
                                  style=LBL),
                        dcc.Input(id="outdir", value=cfg0.output_dir or "", style=INP),
                    ]),
                    html.Div(style=CARD, children=[
                        html.H4("4 · Radar chart", style={"marginTop": 0}),
                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                                        "gap": "6px"}, children=[
                            html.Div([html.Span("Axis max (%, blank = auto)", style=LBL),
                                      dcc.Input(id="amax", type="number", value=cfg0.axis_max,
                                                style=INP)]),
                            html.Div([html.Span("Ring step (%, blank = auto)", style=LBL),
                                      dcc.Input(id="tstep", type="number", value=cfg0.tick_step,
                                                style=INP)]),
                            html.Div([html.Span("Centre offset (rings)", style=LBL),
                                      dcc.Input(id="coff", type="number", min=0, step="any",
                                                value=cfg0.center_offset, style=INP)]),
                            html.Div([html.Span("DPI", style=LBL),
                                      dcc.Input(id="dpi", type="number", value=cfg0.chart_dpi,
                                                style=INP)])]),
                        html.Div("Centre offset: empty space between the centre and the "
                                 "0 % ring, in ring widths (0 = values start at the centre).",
                                 style=HINT),
                        html.Span("Title (optional)", style=LBL),
                        dcc.Input(id="title", value=cfg0.chart_title, style=INP),
                        dcc.Checklist(id="legend", value=["on"] if cfg0.show_legend else [],
                                      options=[{"label": " show legend", "value": "on"}],
                                      style={"marginTop": "8px", "fontSize": "13px"}),
                        html.Button("▶ Compute NRMSE", id="run", style=dict(
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
                        html.Button("⬇ CSV (long)", id="dl-csv-btn", style=BTN),
                        html.Button("⬇ CSV (matrix)", id="dl-mat-btn", style=BTN),
                        html.Button("⬇ Radar PNG", id="dl-png-btn", style=BTN),
                        html.Button("⬇ PDF", id="dl-pdf-btn", style=BTN),
                        html.Button("⬇ SVG", id="dl-svg-btn", style=BTN),
                        html.Button("⬇ All (zip)", id="dl-zip-btn", style=BTN2)]),
                    html.Div(id="res-table", style={"overflowX": "auto"}),
                    html.Img(id="radar-img", style={"width": "100%", "maxWidth": "760px",
                                                    "display": "block",
                                                    "margin": "12px auto 0 auto"}),
                    html.Pre(id="log", style={"fontSize": "11px", "background": "#f6f8fb",
                                              "padding": "8px", "borderRadius": "6px",
                                              "whiteSpace": "pre-wrap", "marginTop": "10px"}),
                ]),
            ]),
            dcc.Download(id="dl-csv"), dcc.Download(id="dl-mat"), dcc.Download(id="dl-png"),
            dcc.Download(id="dl-pdf"), dcc.Download(id="dl-svg"), dcc.Download(id="dl-zip"),
        ])])

    # ---- reference -----------------------------------------------------------
    @app.callback(Output("ref-store", "data"), Output("ref-status", "children"),
                  Output("ref-label", "value"),
                  Input("ref-upload", "contents"), Input("ref-load", "n_clicks"),
                  Input("ref-path", "value"),
                  State("ref-upload", "filename"), State("ref-label", "value"),
                  State("epsg", "value"), prevent_initial_call=True)
    def load_reference(contents, _n, typed, filename, label, epsg):
        path = resolve_path(typed)
        try:
            if ctx.triggered_id == "ref-upload" and contents:
                path = str(_save_upload(contents, filename))
            if not path:
                return None, "", no_update
            if not Path(path).exists():
                raise FileNotFoundError(f"not found: {path}")
            info = describe_file(path, int(epsg or DEFAULT_EPSG))
        except Exception as e:
            return None, f"⚠ {e}", no_update
        status = [html.Div(f"✓ {Path(path).name} — {info['summary']}")]
        if info.get("warning"):
            status.append(html.Div(f"⚠ {info['warning']}", style={"color": "#9a5b00"}))
        return {"path": path}, status, (label or default_label(path))

    # ---- predictions: add / remove / order -----------------------------------
    @app.callback(Output("pred-slots", "children"), Output("slot-count", "data"),
                  Output("pred-order", "data"),
                  Input("pred-add", "n_clicks"), State("slot-count", "data"),
                  State("pred-order", "data"), prevent_initial_call=True)
    def add_slot(_n, count, order):
        p = Patch()
        p.append(pred_slot(count, len(order or [])))
        return p, count + 1, (order or []) + [count]

    @app.callback(Output("pred-order", "data", allow_duplicate=True),
                  Input({"type": "slot-del", "index": ALL}, "n_clicks"),
                  State("pred-order", "data"), prevent_initial_call=True)
    def del_slot(_clicks, order):
        trig_val = ctx.triggered[0]["value"] if ctx.triggered else None
        if not ctx.triggered_id or not trig_val:
            return no_update
        i = ctx.triggered_id["index"]
        return [j for j in (order or []) if j != i]

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

    @app.callback(Output({"type": "pred-store", "index": MATCH}, "data"),
                  Output({"type": "pred-status", "index": MATCH}, "children"),
                  Output({"type": "pred-label", "index": MATCH}, "value"),
                  Input({"type": "pred-upload", "index": MATCH}, "contents"),
                  Input({"type": "pred-load", "index": MATCH}, "n_clicks"),
                  Input({"type": "pred-path", "index": MATCH}, "value"),
                  State({"type": "pred-upload", "index": MATCH}, "filename"),
                  State({"type": "pred-label", "index": MATCH}, "value"),
                  State("epsg", "value"), prevent_initial_call=True)
    def load_prediction(contents, _l, typed, filename, label, epsg):
        trig = ctx.triggered_id["type"] if ctx.triggered_id else None
        path = resolve_path(typed)
        try:
            if trig == "pred-upload" and contents:
                path = str(_save_upload(contents, filename))
            if not path:
                return None, "", no_update
            if not Path(path).exists():
                raise FileNotFoundError(f"not found: {path}")
            info = describe_file(path, int(epsg or DEFAULT_EPSG))
        except Exception as e:
            return None, f"⚠ {e}", no_update
        status = [html.Div(f"✓ {Path(path).name} — {info['summary']}")]
        if info.get("warning"):
            status.append(html.Div(f"⚠ {info['warning']}", style={"color": "#9a5b00"}))
        return {"path": path}, status, (label or default_label(path))

    # ---- colour picker ---------------------------------------------------------
    @app.callback(Output({"type": "pred-color", "index": MATCH}, "value"),
                  Input({"type": "swatch", "index": MATCH, "c": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def pick_swatch(clicks):
        if not ctx.triggered_id or not any(clicks or []):
            return no_update
        return ctx.triggered_id["c"]

    @app.callback(Output({"type": "color-preview", "index": MATCH}, "style"),
                  Input({"type": "pred-color", "index": MATCH}, "value"),
                  State("pred-order", "data"), prevent_initial_call=True)
    def show_color(value, order):
        i = ctx.triggered_id["index"] if ctx.triggered_id else 0
        pos = (order or []).index(i) if i in (order or []) else 0
        col, ok = resolve_color(value, PALETTE[pos % len(PALETTE)])
        return preview_style(col, ok or not (value or "").strip())

    # ---- helpers --------------------------------------------------------------
    def chart_cfg(amax, tstep, coff, title, legend, dpi) -> Config:
        return Config(axis_max=_num(amax), tick_step=_num(tstep),
                      center_offset=_num(coff, 0.0), chart_title=(title or "").strip(),
                      show_legend="on" in (legend or []), chart_dpi=int(_num(dpi, CHART_DPI)))

    TH = {"fontWeight": 600, "background": "#f0f3f7", "padding": "5px 8px",
          "border": "1px solid #dde1e6", "fontSize": "12.5px", "textAlign": "center"}
    TD = {"padding": "4px 8px", "border": "1px solid #dde1e6", "fontSize": "12.5px",
          "textAlign": "right"}

    def table_of(matrix: pd.DataFrame):
        """Read-only HTML table: predictions x variables (NRMSE %) + mean."""
        keys = list(matrix.columns)
        fmt = lambda v: f"{v:.2f}" if np.isfinite(v) else "–"
        head = html.Tr([html.Th("Prediction", style=TH)] +
                       [html.Th(f"{VAR_INFO[k][2]} (%)", style=TH) for k in keys] +
                       [html.Th("Mean (%)", style=TH)])
        body = []
        for lab, row in matrix.iterrows():
            fin = [v for v in row.values if np.isfinite(v)]
            body.append(html.Tr([html.Td(str(lab), style=dict(TD, textAlign="left"))] +
                                [html.Td(fmt(float(v)), style=TD) for v in row.values] +
                                [html.Td(fmt(float(np.mean(fin))) if fin else "–",
                                         style=dict(TD, fontWeight=600))]))
        return html.Table([html.Thead(head), html.Tbody(body)],
                          style={"borderCollapse": "collapse", "width": "100%"})

    # ---- run (worker thread) ----------------------------------------------------
    @app.callback(Output("job-store", "data"), Output("poll", "disabled"),
                  Output("run-status", "children"), Output("progress-wrap", "style"),
                  Output("run", "disabled"),
                  Output("res-table", "children"), Output("radar-img", "src"),
                  Output("downloads", "style"), Output("log", "children"),
                  Input("run", "n_clicks"),
                  State("ref-store", "data"), State("ref-label", "value"),
                  State({"type": "pred-store", "index": ALL}, "data"),
                  State({"type": "pred-store", "index": ALL}, "id"),
                  State({"type": "pred-label", "index": ALL}, "value"),
                  State({"type": "pred-label", "index": ALL}, "id"),
                  State({"type": "pred-color", "index": ALL}, "value"),
                  State({"type": "pred-color", "index": ALL}, "id"),
                  State("pred-order", "data"),
                  State("vars", "value"), State("thr", "value"), State("tmask", "value"),
                  State("mask", "value"),
                  State("norm", "value"), State("resamp", "value"), State("res", "value"),
                  State("rhos", "value"), State("rhow", "value"), State("grav", "value"),
                  State("epsg", "value"), State("outdir", "value"),
                  State("amax", "value"), State("tstep", "value"), State("coff", "value"),
                  State("title", "value"), State("legend", "value"), State("dpi", "value"),
                  prevent_initial_call=True)
    def start_run(_n, ref, ref_label, pstores, pstore_ids, plabels, plabel_ids, pcolors,
                  pcolor_ids, order, vars_, thr, tmask, mask, norm, resamp, res, rhos, rhow, grav,
                  epsg, outdir, amax, tstep, coff, title, legend, dpi):
        hide = {"display": "none"}
        fail = lambda msg: (no_update, True, html.Span(msg, style={"color": "#b42318"}),
                            hide, False, None, None, hide, "")
        if not ref or not ref.get("path"):
            return fail("⚠ Load the reference Maximums.nc first.")
        st = {i["index"]: s for i, s in zip(pstore_ids, pstores)}
        lb = {i["index"]: l for i, l in zip(plabel_ids, plabels)}
        cl = {i["index"]: c for i, c in zip(pcolor_ids, pcolors)}
        preds = [(st[i]["path"], (lb.get(i) or "").strip() or None,
                  (cl.get(i) or "").strip() or None)
                 for i in (order or []) if st.get(i) and st[i].get("path")]
        if not preds:
            return fail("⚠ Load at least one prediction.")
        if not vars_:
            return fail("⚠ Select at least one variable.")
        out = resolve_path(outdir) or None
        if out is None and str(ref["path"]).startswith(str(UPLOAD_ROOT)):
            out = str(SCRIPT_DIR / "Results_NRMSE")
        cc = chart_cfg(amax, tstep, coff, title, legend, dpi)
        cfg = Config(reference_path=ref["path"],
                     reference_label=(ref_label or "").strip() or None,
                     prediction_paths=[p for p, _, _ in preds],
                     prediction_labels=[l for _, l, _ in preds],
                     prediction_colors=[c for _, _, c in preds],
                     variables=[k for k in VARIABLES if k in vars_],
                     depth_threshold=_num(thr, DEPTH_THRESHOLD),
                     time_depth_mask="on" in (tmask or []),
                     mask_mode=mask or MASK_MODE, normalization=norm or NORMALIZATION,
                     resampling=resamp or RESAMPLING, grid_resolution=_num(res),
                     rho_solid=_num(rhos, RHO_SOLID), rho_water=_num(rhow, RHO_WATER),
                     gravity=_num(grav, GRAVITY), default_epsg=int(_num(epsg, DEFAULT_EPSG)),
                     axis_max=cc.axis_max, tick_step=cc.tick_step,
                     center_offset=cc.center_offset, chart_title=cc.chart_title,
                     show_legend=cc.show_legend, chart_dpi=cc.chart_dpi, output_dir=out)
        job_id = uuid.uuid4().hex
        job = {"pct": 0.0, "msg": "Starting…", "log": [], "done": False,
               "result": None, "error": None, "t0": time.time()}
        JOBS[job_id] = job

        def worker():
            def prog(f, msg):
                job["pct"], job["msg"] = f, msg
            try:
                job["result"] = run_nrmse(cfg, log=job["log"].append, progress=prog)
            except Exception as e:
                job["error"] = str(e)
                job["log"].append(traceback.format_exc())
            finally:
                job["done"] = True

        threading.Thread(target=worker, daemon=True).start()
        return (job_id, False, "Running…", {"display": "block"}, True, None, None, hide, "")

    # ---- poll progress ------------------------------------------------------------
    @app.callback(Output("progress-bar", "style"), Output("progress-text", "children"),
                  Output("log", "children", allow_duplicate=True),
                  Output("poll", "disabled", allow_duplicate=True),
                  Output("run", "disabled", allow_duplicate=True),
                  Output("run-status", "children", allow_duplicate=True),
                  Output("res-table", "children", allow_duplicate=True),
                  Output("radar-img", "src", allow_duplicate=True),
                  Output("result-store", "data"),
                  Output("downloads", "style", allow_duplicate=True),
                  Input("poll", "n_intervals"), State("job-store", "data"),
                  prevent_initial_call=True)
    def poll(_n, job_id):
        job = JOBS.get(job_id)
        if not job:
            return (no_update,) * 3 + (True,) + (no_update,) * 6
        pct = 100 * job["pct"]
        el = int(time.time() - job["t0"])
        bar = {"width": f"{pct:.0f}%", "height": "100%", "transition": "width 0.4s ease",
               "background": "#b42318" if job["error"] else "#0f7a4a"}
        text = f"{pct:.0f} % — {job['msg']} · {el // 60:02d}:{el % 60:02d} elapsed"
        log = "\n".join(job["log"])
        if not job["done"]:
            return (bar, text, log, False, True) + (no_update,) * 5
        if job["error"]:
            return (bar, f"Stopped at {pct:.0f} % — {job['msg']}", log, True, False,
                    html.Span(f"⚠ {job['error']}", style={"color": "#b42318"}),
                    None, None, None, {"display": "none"})
        r = job["result"]
        table = table_of(r["matrix"])
        status = html.Span(f"✓ Done in {el // 60:02d}:{el % 60:02d} — reference "
                           f"'{r['reference_label']}' · results in {r['out_dir']}",
                           style={"color": "#0f7a4a"})
        store = {k: r[k] for k in ("csv", "matrix_csv", "png", "pdf", "svg", "files",
                                   "out_dir", "labels")}
        JOBS.pop(job_id, None)
        img = "data:image/png;base64," + r["preview_b64"] if r["preview_b64"] else None
        return (bar, text, log, True, False, status, table, img, store,
                {"display": "block", "margin": "8px 0"})

    # ---- downloads ------------------------------------------------------------------
    for key, skey in (("csv", "csv"), ("mat", "matrix_csv"), ("png", "png"),
                      ("pdf", "pdf"), ("svg", "svg")):
        @app.callback(Output(f"dl-{key}", "data"), Input(f"dl-{key}-btn", "n_clicks"),
                      State("result-store", "data"), prevent_initial_call=True)
        def _dl(_n, store, skey=skey):
            return dcc.send_file(store[skey]) if store and store.get(skey) else no_update

    @app.callback(Output("dl-zip", "data"), Input("dl-zip-btn", "n_clicks"),
                  State("result-store", "data"), prevent_initial_call=True)
    def dl_zip(_n, store):
        if not store:
            return no_update
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in store["files"]:
                if Path(f).exists():
                    z.write(f, arcname=Path(f).name)
        return dcc.send_bytes(buf.getvalue(), "nrmse_results.zip")

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
        if not cfg.reference_path or not cfg.prediction_paths:
            sys.exit("Set REFERENCE_PATH and PREDICTION_PATHS (USER CONFIG) or use --config.")
        base = Path(args.config).resolve().parent if args.config else SCRIPT_DIR
        fix = lambda p: str(p if Path(p).is_absolute() else base / p)
        cfg.reference_path = fix(cfg.reference_path)
        cfg.prediction_paths = [fix(p) for p in cfg.prediction_paths]
        r = run_nrmse(cfg)
        with pd.option_context("display.width", 200, "display.max_columns", 20,
                               "display.float_format", "{:.2f}".format):
            print("\nNRMSE (%)\n" + r["matrix"].to_string())
        return
    app = build_app(cfg)
    print(f"NRMSE dashboard → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    sys.exit(main())
