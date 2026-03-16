import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV_PATH = '/Users/ivo/Documents/darbam/LF/LF_ukri_NDVI_2019_2025.csv'
SHP_PATH = '/Users/ivo/Documents/darbam/LF/LF_ukri.shp'
FIG_OUT  = '/Users/ivo/Documents/darbam/LF/LF_ukri_NDVI_timeseries.png'

# ── 1. Load data ───────────────────────────────────────────────────────────────
df  = pd.read_csv(CSV_PATH)
gdf = gpd.read_file(SHP_PATH)

print("── CSV columns ──────────────────────────────────")
print(df.columns.tolist())
print("\n── First 5 rows ─────────────────────────────────")
print(df.head())
print(f"\nRows: {len(df)}  |  Unique compartments: {df['compartment_id'].nunique()}  |  Years: {sorted(df['year'].unique())}")

# ── 2. Derived aggregates ──────────────────────────────────────────────────────
years = sorted(df['year'].unique())

# Per-year statistics across all compartments
annual = (df.groupby('year')['mean_ndvi']
            .agg(mean='mean',
                 p25=lambda x: np.percentile(x, 25),
                 p75=lambda x: np.percentile(x, 75))
            .reindex(years))

# Pivot: rows = compartment, columns = year
pivot = df.pivot(index='compartment_id', columns='year', values='mean_ndvi')

# Sort compartments by their overall mean NDVI (lowest at top → worst performers prominent)
pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]

# ── 3. Figure ──────────────────────────────────────────────────────────────────
fig, (ax_line, ax_heat) = plt.subplots(
    2, 1,
    figsize=(10, 11),
    gridspec_kw={'height_ratios': [1, 2.5], 'hspace': 0.35},
)

# ── Top panel: mean ± IQR line ─────────────────────────────────────────────────
ax_line.fill_between(
    annual.index,
    annual['p25'],
    annual['p75'],
    alpha=0.25,
    color='#4dac26',
    label='25th–75th percentile',
)
ax_line.plot(
    annual.index,
    annual['mean'],
    marker='o',
    linewidth=2,
    color='#1a6314',
    label='Mean NDVI',
)

ax_line.set_xlim(years[0] - 0.3, years[-1] + 0.3)
ax_line.set_xticks(years)
ax_line.set_xlabel('Year', fontsize=11)
ax_line.set_ylabel('NDVI', fontsize=11)
ax_line.set_title('Mean Summer NDVI across all compartments (Jun–Aug)', fontsize=12, fontweight='bold')
ax_line.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax_line.legend(fontsize=10)
ax_line.grid(axis='y', linestyle='--', alpha=0.4)

# ── Bottom panel: heatmap ──────────────────────────────────────────────────────
vmin = np.nanpercentile(pivot.values, 5)
vmax = np.nanpercentile(pivot.values, 95)

im = ax_heat.imshow(
    pivot.values,
    aspect='auto',
    cmap='RdYlGn',
    vmin=vmin,
    vmax=vmax,
    interpolation='nearest',
)

# Axes labels
ax_heat.set_xticks(range(len(pivot.columns)))
ax_heat.set_xticklabels(pivot.columns.astype(int), fontsize=10)
ax_heat.set_xlabel('Year', fontsize=11)

n_comp = len(pivot)
# Show compartment IDs on y-axis only if there are few enough to be readable
if n_comp <= 60:
    ax_heat.set_yticks(range(n_comp))
    ax_heat.set_yticklabels(pivot.index, fontsize=6)
else:
    ax_heat.set_yticks([])
    ax_heat.set_ylabel(f'Compartments (n={n_comp}, sorted by mean NDVI ↑)', fontsize=11)

ax_heat.set_title('Summer NDVI per compartment × year (sorted by mean NDVI, low → high)',
                  fontsize=12, fontweight='bold')

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
cbar.set_label('NDVI', fontsize=10)

# ── 4. Save ────────────────────────────────────────────────────────────────────
fig.savefig(FIG_OUT, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {FIG_OUT}")
