import ee
import geemap
import geopandas as gpd
import folium

# ── 1. Initialise GEE ─────────────────────────────────────────────────────────
ee.Initialize(project='gee-ivo')
print("[1/7] GEE initialised with project 'gee-ivo'")

# ── 2. Load shapefile → GEE FeatureCollection ─────────────────────────────────
SHP_PATH  = '/Users/ivo/Documents/darbam/LF/LF_ukri.shp'
CSV_OUT   = '/Users/ivo/Documents/darbam/LF/LF_ukri_NDVI_2019_2025.csv'
MAP_OUT   = '/Users/ivo/Documents/darbam/LF/LF_ukri_NDVI_map.html'

gdf = gpd.read_file(SHP_PATH).to_crs('EPSG:4326')
print(f"[2/7] Shapefile loaded: {len(gdf)} compartments, CRS → EPSG:4326")

compartments = geemap.gdf_to_ee(gdf)
bounds = compartments.geometry()

# ── 3. Helper functions ────────────────────────────────────────────────────────
def mask_s2_scl(image):
    """Keep SCL classes 4 (vegetation), 5 (bare soil), 6 (water), 11 (snow/ice)."""
    scl = image.select('SCL')
    mask = (scl.eq(4)
              .Or(scl.eq(5))
              .Or(scl.eq(6))
              .Or(scl.eq(11)))
    return image.updateMask(mask).copyProperties(image, image.propertyNames())

def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# ── 4. Sentinel-2 SR L2A base collection ──────────────────────────────────────
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(bounds)
        .filterDate('2019-01-01', '2025-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
        .map(mask_s2_scl)
        .map(add_ndvi)
        .select('NDVI'))

print("[3/7] Sentinel-2 collection filtered, cloud-masked, NDVI computed")

# ── 5. Annual summer (Jun–Aug) median composites ───────────────────────────────
years = list(range(2019, 2026))

def summer_composite(year):
    start = ee.Date.fromYMD(year, 6, 1)
    end   = ee.Date.fromYMD(year, 8, 31)
    return (s2.filterDate(start, end)
              .median()
              .clip(bounds)
              .set('year', year))

composites = {year: summer_composite(year) for year in years}
print(f"[4/7] Annual summer composites built for years: {years}")

# ── 6. Zonal stats per compartment per year ────────────────────────────────────
print("[5/7] Running reduceRegions for each year (this may take a moment)...")

reducer = (
    ee.Reducer.mean()
      .combine(ee.Reducer.percentile([10, 90]), sharedInputs=True)
      .combine(ee.Reducer.stdDev(), sharedInputs=True)
)

all_features = []
for year, composite in composites.items():
    zonal = composite.reduceRegions(
        collection=compartments,
        reducer=reducer,
        scale=10,
        tileScale=4,
    )
    # GEE output band names: NDVI_mean, NDVI_p10, NDVI_p90, NDVI_stdDev
    zonal = zonal.map(lambda f: f.set({
        'mean_ndvi':   f.get('NDVI_mean'),
        'p10_ndvi':    f.get('NDVI_p10'),
        'p90_ndvi':    f.get('NDVI_p90'),
        'stddev_ndvi': f.get('NDVI_stdDev'),
        'year':        year,
    }))
    all_features.append(zonal)
    print(f"    • {year} done")

flat = ee.FeatureCollection(all_features).flatten()

# ── 7. Export CSV ──────────────────────────────────────────────────────────────
print(f"[6/7] Exporting CSV → {CSV_OUT}")

df_out = geemap.ee_to_df(flat)
print("    • Raw columns:", df_out.columns.tolist())
print(df_out.head(2))

# Map whatever GEE named the combined-reducer outputs to our target column names.
# Typical GEE names for a combined reducer on band 'NDVI':
#   mean → 'NDVI_mean', p10 → 'NDVI_p10', p90 → 'NDVI_p90', stdDev → 'NDVI_stdDev'
# The compartment ID column is usually 'fid' or 'system:index' depending on the FC.
df_out = df_out[['fid', 'year', 'mean', 'p10', 'p90', 'stdDev']].rename(columns={
    'fid':    'compartment_id',
    'mean':   'mean_ndvi',
    'p10':    'p10_ndvi',
    'p90':    'p90_ndvi',
    'stdDev': 'stddev_ndvi',
})
df_out['cv_ndvi'] = df_out['stddev_ndvi'] / df_out['mean_ndvi']
df_out.to_csv(CSV_OUT, index=False)

print(f"    • Shape: {df_out.shape}")
print(df_out.head())
print(f"    • CSV saved: {CSV_OUT}")

# ── 8. Interactive map ─────────────────────────────────────────────────────────
print(f"[7/7] Building map → {MAP_OUT}")

# Fixed vis_params shared across all years for comparability
ndvi_vis = {
    'min': 0.75,
    'max': 0.92,
    'palette': ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'],
}

# Compute centroid: reproject to EPSG:3857 for accurate centroid, then back to WGS84
centroid_3857 = gdf.to_crs('EPSG:3857').dissolve().centroid.iloc[0]
centroid_wgs84 = gpd.GeoSeries([centroid_3857], crs='EPSG:3857').to_crs('EPSG:4326').iloc[0]
lat, lon = centroid_wgs84.y, centroid_wgs84.x
print(f"    • Map centroid: lat={lat:.5f}, lon={lon:.5f}")

# Build folium map with Google Satellite basemap
m = folium.Map(
    location=[lat, lon],
    zoom_start=14,
    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr='Google Satellite',
)

# Add one tile layer per year; only 2024 visible by default
print("    • Fetching GEE tile URLs for all years...")
for year in years:
    tile_url = composites[year].getMapId(ndvi_vis)['tile_fetcher'].url_format
    folium.TileLayer(
        tiles=tile_url,
        attr='GEE',
        name=f'NDVI Summer {year}',
        overlay=True,
        show=(year == 2024),
    ).add_to(m)
    print(f"      – {year} tile URL fetched")

# Compartment outlines
folium.GeoJson(
    gdf,
    name='LF_ukri compartments',
    style_function=lambda _: {'color': 'white', 'weight': 0.8, 'fillOpacity': 0},
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

m.save(MAP_OUT)
print(f"    • Map saved: {MAP_OUT}")
print("\nDone.")
