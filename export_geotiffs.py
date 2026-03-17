"""
Export annual NDVI composites (2019–2025) as GeoTIFFs to Google Drive.

Output folder : LF_NDVI  (created automatically in Drive root)
Filename      : LF_ukri_NDVI_{year}
Scale         : 10 m  (Sentinel-2 native)
CRS           : EPSG:4326
Clipped to    : LF_ukri compartment boundaries

Run:
    python export_geotiffs.py

Exports are asynchronous – the script submits all 7 tasks and prints their
status. Monitor progress at https://code.earthengine.google.com/tasks
"""

import ee
import geemap
import geopandas as gpd

# ── Init ───────────────────────────────────────────────────
ee.Initialize(project='gee-ivo')
print("GEE initialised with project 'gee-ivo'")

# ── Load shapefile → EE FeatureCollection ─────────────────
SHP_PATH = '/Users/ivo/Documents/darbam/LF/LF_ukri.shp'

gdf = gpd.read_file(SHP_PATH).to_crs('EPSG:4326')
print(f"Shapefile loaded: {len(gdf)} compartments")

compartments = geemap.gdf_to_ee(gdf)
clip_geom = compartments.geometry()

# Bounding box for the export region (ee_export_image_to_drive needs a region)
bounds_list = list(gdf.total_bounds)          # [W, S, E, N]
region = ee.Geometry.Rectangle(bounds_list)

# ── Cloud mask + NDVI ──────────────────────────────────────
def mask_ndvi(image):
    """Apply SCL mask (classes 4,5,6,11) and compute NDVI."""
    scl = image.select('SCL')
    mask = (scl.eq(4)
              .Or(scl.eq(5))
              .Or(scl.eq(6))
              .Or(scl.eq(11)))
    ndvi = image.updateMask(mask).normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ndvi

# ── Submit one export task per year ───────────────────────
YEARS = range(2019, 2026)

for year in YEARS:
    composite = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(clip_geom)
        .filterDate(f'{year}-06-01', f'{year}-08-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
        .map(mask_ndvi)
        .median()
        .clip(clip_geom)
    )

    task = geemap.ee_export_image_to_drive(
        image=composite,
        description=f'LF_ukri_NDVI_{year}',
        folder='LF_NDVI',
        fileNamePrefix=f'LF_ukri_NDVI_{year}',
        scale=10,
        region=region,
        crs='EPSG:4326',
        maxPixels=1e10,
    )

    print(f"  [{year}] task submitted – id: {task.id if hasattr(task, 'id') else 'see Tasks panel'}")

print(f"\nAll {len(list(YEARS))} tasks submitted.")
print("Track progress at: https://code.earthengine.google.com/tasks")
