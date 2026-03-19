#!/usr/bin/env python3
"""
Wind shelter index (simplified Winstral Sx) computed from a DEM.
Direction: 247.5° WSW (primary).

Shelter angle Sx = max over sample distances of arctan((z_upwind - z_pixel) / distance)
Positive = sheltered (upwind terrain higher), negative = exposed.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import rowcol as rio_rowcol, xy as rio_xy
from rasterio.features import geometry_mask
import geopandas as gpd
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
DEM_PATH          = "/Volumes/ivl/darbi/LF/ukti_dem.tif"
COMPARTMENTS_PATH = "/Users/ivo/Documents/darbam/LF/LF_ukri.shp"
TARGET_CRS        = "EPSG:3059"
DISTANCES_M       = [50, 100, 200, 400, 800]   # sample distances (m)
WSW_AZIMUTH       = 247.5                       # wind comes FROM WSW
OUTPUT_WSW        = "wsw_shelter.tif"
OUTPUT_FIG        = "wind_shelter_map.png"
CLIP_BUFFER_M     = 500   # mask/clip data outside compartments + this buffer
ZOOM_BUFFER_M     = 200   # axes zoom extent around compartments


# ── DEM loading ───────────────────────────────────────────────────────────────
def load_dem(path: str, target_crs: str):
    """Load DEM, reproject to target_crs if needed. Returns (array, transform, profile)."""
    with rasterio.open(path) as src:
        src_crs  = src.crs
        src_epsg = src_crs.to_epsg() if src_crs else None
        tgt_epsg = int(target_crs.split(":")[1])

        if src_epsg == tgt_epsg:
            data      = src.read(1).astype(np.float32)
            transform = src.transform
            profile   = src.profile.copy()
            nodata    = src.nodata
        else:
            transform, width, height = calculate_default_transform(
                src_crs, target_crs, src.width, src.height, *src.bounds
            )
            data = np.empty((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
            nodata  = src.nodata
            profile = src.profile.copy()
            profile.update(crs=target_crs, transform=transform,
                           width=width, height=height)

    if nodata is not None:
        data[data == nodata] = np.nan

    return data, transform, profile


# ── Shelter index ─────────────────────────────────────────────────────────────
def compute_shelter_index(
    dem: np.ndarray,
    transform,
    azimuth_deg: float,
    distances: list[int],
) -> np.ndarray:
    """
    Simplified Winstral Sx shelter index.

    Parameters
    ----------
    dem          : 2-D float32 array (NaN = nodata)
    transform    : rasterio Affine transform (must be in same CRS as distances)
    azimuth_deg  : direction wind comes FROM (degrees, clockwise from North)
    distances    : sample distances in metres

    Returns
    -------
    shelter : float32 array, same shape as dem, values in degrees.
              Positive = sheltered, negative = exposed, NaN = nodata.
    """
    nrows, ncols = dem.shape
    pixel_w = abs(transform.a)   # metres per pixel (x)
    pixel_h = abs(transform.e)   # metres per pixel (y)

    az_rad = np.deg2rad(azimuth_deg)
    # Pixel-space step per metre toward the upwind direction:
    #   East component  → col increases
    #   North component → row decreases  (rows go south)
    dcol_per_m =  np.sin(az_rad) / pixel_w
    drow_per_m = -np.cos(az_rad) / pixel_h

    # Base index grids
    rows_idx = np.arange(nrows, dtype=np.float64)
    cols_idx = np.arange(ncols, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(cols_idx, rows_idx)   # (nrows, ncols)

    valid_mask = np.isfinite(dem)

    # Fill NaN with 0 for interpolation; use in_bounds mask instead
    dem_filled = np.where(valid_mask, dem, 0.0).astype(np.float64)

    shelter     = np.full((nrows, ncols), -np.inf, dtype=np.float64)
    any_valid   = np.zeros((nrows, ncols), dtype=bool)

    for d in distances:
        s_rows = row_grid + drow_per_m * d   # sample row coords (float)
        s_cols = col_grid + dcol_per_m * d   # sample col coords (float)

        # Bounds check (conservative: require sample inside array)
        in_bounds = (
            (s_rows >= 0) & (s_rows <= nrows - 1) &
            (s_cols >= 0) & (s_cols <= ncols - 1)
        )

        z_sample = map_coordinates(
            dem_filled,
            [s_rows.ravel(), s_cols.ravel()],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        ).reshape(nrows, ncols)

        # Validity: source pixel valid, sample in-bounds, sample DEM valid
        # (interpolated value may mix with nodata fill; check with valid mask interp)
        valid_at_sample = map_coordinates(
            valid_mask.astype(np.float32),
            [s_rows.ravel(), s_cols.ravel()],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).reshape(nrows, ncols) > 0.9   # >0.9 means all neighbours were valid

        use = valid_mask & in_bounds & valid_at_sample

        dz    = z_sample - dem                    # upwind elev - current elev
        angle = np.arctan2(dz, float(d))          # = arctan(dz/d), range (-π/2, π/2)

        shelter   = np.where(use, np.maximum(shelter, angle), shelter)
        any_valid = any_valid | use

    # Pixels with no valid upwind sample → NaN
    shelter[~valid_mask] = np.nan
    shelter[~any_valid & valid_mask] = np.nan

    return np.rad2deg(shelter).astype(np.float32)


# ── Output helpers ─────────────────────────────────────────────────────────────
def save_geotiff(data: np.ndarray, profile: dict, path: str) -> None:
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data, 1)


def stats_within_compartments(shelter, transform, compartments):
    geoms = [g.__geo_interface__ for g in compartments.geometry]
    mask  = geometry_mask(geoms, transform=transform,
                          invert=True, out_shape=shelter.shape)
    vals  = shelter[mask & np.isfinite(shelter)]
    return float(vals.min()), float(vals.max()), float(vals.mean())


# ── Figure ─────────────────────────────────────────────────────────────────────
def make_figure(shelter, transform, compartments) -> None:
    bounds = compartments.total_bounds   # minx, miny, maxx, maxy

    # Clip/mask extent: compartments + 500 m
    nrows, ncols = shelter.shape
    bx0 = bounds[0] - CLIP_BUFFER_M
    by0 = bounds[1] - CLIP_BUFFER_M
    bx1 = bounds[2] + CLIP_BUFFER_M
    by1 = bounds[3] + CLIP_BUFFER_M

    r0, c0 = rio_rowcol(transform, bx0, by1)   # top-left corner
    r1, c1 = rio_rowcol(transform, bx1, by0)   # bottom-right corner
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(nrows, r1 + 1), min(ncols, c1 + 1)

    clip = shelter[r0:r1, c0:c1].copy()

    # Mask pixels outside compartments + 500 m buffer
    clip_transform = transform * transform.translation(c0, r0)   # shift to clip origin
    from rasterio.features import geometry_mask as _gmask
    buffer_geom = [compartments.unary_union.buffer(CLIP_BUFFER_M).__geo_interface__]
    outside = _gmask(buffer_geom, transform=clip_transform,
                     invert=False, out_shape=clip.shape)
    clip[outside] = np.nan

    # Exact geographic extent of the clipped pixel block (for imshow)
    left,  top    = rio_xy(transform, r0, c0)
    right, bottom = rio_xy(transform, r1, c1)
    extent = [left, right, bottom, top]

    # Zoom extent: compartments + 200 m
    zx0 = bounds[0] - ZOOM_BUFFER_M
    zy0 = bounds[1] - ZOOM_BUFFER_M
    zx1 = bounds[2] + ZOOM_BUFFER_M
    zy1 = bounds[3] + ZOOM_BUFFER_M

    finite = clip[np.isfinite(clip)]
    p2, p50, p98 = (np.percentile(finite, [2, 50, 98]) if finite.size
                    else (np.nan, np.nan, np.nan))
    print(f"  Shelter index within compartments+{CLIP_BUFFER_M}m buffer:")
    print(f"    p2={p2:.2f}°  p50={p50:.2f}°  p98={p98:.2f}°")

    vlim = np.nanpercentile(np.abs(clip), 98) if finite.size else 1.0
    vlim = max(vlim, 0.1)

    _, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(
        clip,
        extent=extent,
        origin="upper",
        cmap="coolwarm_r",
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
    )
    compartments.boundary.plot(ax=ax, edgecolor="yellow", linewidth=1.5, zorder=2)

    cb = plt.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("Shelter angle (°)", fontsize=10)

    ax.set_title("Wind shelter index — WSW (247.5°)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Easting (m)", fontsize=9)
    ax.set_ylabel("Northing (m)", fontsize=9)
    ax.set_xlim(zx0, zx1)
    ax.set_ylim(zy0, zy1)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.annotate(
        "Blue = exposed  |  Red = sheltered",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {OUTPUT_FIG}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading DEM …")
    dem, transform, profile = load_dem(DEM_PATH, TARGET_CRS)
    res = abs(transform.a)
    print(f"  shape={dem.shape}  CRS={profile['crs']}  resolution={res:.1f} m")

    print("Loading compartment boundaries …")
    compartments = gpd.read_file(COMPARTMENTS_PATH).to_crs(TARGET_CRS)
    print(f"  {len(compartments)} compartments")

    print(f"\nComputing shelter index — WSW ({WSW_AZIMUTH}°) …")
    shelter = compute_shelter_index(dem, transform, WSW_AZIMUTH, DISTANCES_M)

    save_geotiff(shelter, profile, OUTPUT_WSW)
    print(f"  Saved: {OUTPUT_WSW}")

    vmin, vmax, vmean = stats_within_compartments(shelter, transform, compartments)
    print(f"  Within compartments → min={vmin:.2f}°  max={vmax:.2f}°  mean={vmean:.2f}°")

    print("\nGenerating figure …")
    make_figure(shelter, transform, compartments)

    print("\nDone.")


if __name__ == "__main__":
    main()
