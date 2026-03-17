"""
Streamlit app – LF_ukri NDVI Explorer
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
import rasterio
import io
import base64
import matplotlib.image as mpimg

# ── Config ─────────────────────────────────────────────────
CSV_PATH = "LF_ukri_NDVI_2019_2025.csv"
SHP_PATH = "LF_ukri.shp"
GOOGLE_SAT = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
YEARS = list(range(2019, 2026))

st.set_page_config(page_title="LF NDVI pārlūks", layout="wide")
st.title("LF_ukri – NDVI pārlūks 2019–2025")

st.markdown("""
Rīks vizualizē Sentinel-2 satelītuzņēmumu vasaras (jūnijs–augusts) NDVI datus 2019.–2025. gadam bērzu stādījumu nogabaliem Ukru apkaimē. NDVI (normalizētais diferenciālais
veģetācijas indekss) atspoguļo koksnes veģetācijas vitalitāti – zemākas vērtības var liecināt par
stresa faktoriem kā sausums, kaitēkļi vai mehāniski bojājumi. Karte ļauj salīdzināt nogabalu
vitalitāti pa gadiem, savukārt laika rindas grafiks rāda katra nogabala individuālo dinamiku un
iekšējo neviendabīgumu (P10–P90 josla). Pikseļu līmeņa karte (lapas apakšā) ļauj novērtēt telpisko
sadalījumu 10 m izšķirtspējā – identificējot stresa perēkļus nogabalu iekšienē. Rīks izstrādāts kā
prototips dabā balstītu risinājumu plānošanas un monitoringa atbalstam.
""")

# ── TIF config & raster→overlay helper ─────────────────────
TIF_DIR = "tif"
TIF_CMAP = matplotlib.colormaps["RdYlGn"]
TIF_NORM = mcolors.Normalize(vmin=0.75, vmax=0.92)

@st.cache_data
def tif_to_overlay(year: int):
    """Read a local GeoTIFF, apply colormap, return (base64-PNG-url, bounds)."""
    path = f"{TIF_DIR}/LF_ukri_NDVI_{year}.tif"
    with rasterio.open(path) as src:
        ndvi = src.read(1).astype(float)
        nodata = src.nodata
        b = src.bounds  # left, bottom, right, top (EPSG:4326)

    # Build [[south, west], [north, east]] for folium
    img_bounds = [[b.bottom, b.left], [b.top, b.right]]

    # Mask nodata / NaN → transparent
    if nodata is not None:
        mask = (ndvi == nodata) | np.isnan(ndvi)
    else:
        mask = np.isnan(ndvi)

    rgba = TIF_CMAP(TIF_NORM(ndvi))          # (H, W, 4), float 0-1
    rgba[mask] = [0.0, 0.0, 0.0, 0.0]        # fully transparent

    buf = io.BytesIO()
    mpimg.imsave(buf, rgba, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    url = f"data:image/png;base64,{b64}"

    return url, img_bounds

# ── Load data ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["compartment_id"] = df["compartment_id"].astype(int)

    gdf = gpd.read_file(SHP_PATH)
    gdf["fid"] = gdf["fid"].astype(int)
    gdf = gdf[["fid", "geometry"]].rename(columns={"fid": "compartment_id"})
    gdf = gdf.to_crs(epsg=4326)

    # Overall mean NDVI per compartment (all years)
    overall = (
        df.groupby("compartment_id")["mean_ndvi"]
        .mean()
        .reset_index()
        .rename(columns={"mean_ndvi": "avg_ndvi_all"})
    )
    gdf = gdf.merge(overall, on="compartment_id", how="left")
    return df, gdf


df, gdf = load_data()

# ── Centroid ────────────────────────────────────────────────
centroid = gdf.to_crs(epsg=3857).geometry.union_all().centroid
centroid_wgs = gpd.GeoSeries([centroid], crs=3857).to_crs(4326).iloc[0]
MAP_LAT, MAP_LON = centroid_wgs.y, centroid_wgs.x

# ── Colour helpers ──────────────────────────────────────────
cmap = matplotlib.colormaps["RdYlGn"]

def make_norm(series):
    return mcolors.Normalize(
        vmin=float(series.quantile(0.05)),
        vmax=float(series.quantile(0.95)),
    )

def ndvi_to_hex(val, norm):
    if pd.isna(val):
        return "#888888"
    return mcolors.to_hex(cmap(norm(float(val))))

# ── Trend slopes ────────────────────────────────────────────
def _slope(grp):
    yrs = grp["year"].values
    ndvi = grp["mean_ndvi"].values
    return float(np.polyfit(yrs, ndvi, 1)[0]) if len(yrs) >= 2 else 0.0

slope_lookup = df.groupby("compartment_id").apply(_slope).to_dict()

def trend_arrow(cid):
    s = slope_lookup.get(cid, 0.0)
    if s < -0.003: return "↓"
    if s >  0.003: return "↑"
    return "→"

# ── Per-year NDVI lookup (for map recolouring) ──────────────
# pivot: index=compartment_id, columns=year, values=mean_ndvi
year_ndvi = df.pivot(index="compartment_id", columns="year", values="mean_ndvi")

# ── Layout ──────────────────────────────────────────────────
col_map, col_panel = st.columns([65, 35])

# ════════════════════════════════════════════════════════════
# LEFT – map
# ════════════════════════════════════════════════════════════
with col_map:
    selected_year = st.selectbox(
        "Atlasīt gadu",
        options=YEARS,
        index=YEARS.index(2025),
        key="year_sel",
    )

    # Determine fill colour: use selected year's NDVI if available, else all-years mean
    year_col = year_ndvi[selected_year] if selected_year in year_ndvi.columns else None

    if year_col is not None:
        fill_series = gdf["compartment_id"].map(year_col)
        norm = make_norm(fill_series.dropna())
        tooltip_year_label = str(selected_year)
    else:
        fill_series = gdf["avg_ndvi_all"]
        norm = make_norm(fill_series.dropna())
        tooltip_year_label = "all years"

    # Panel state: selected compartment comes from right column widget
    # We read it from session_state so both columns share it
    sel_id = st.session_state.get("sel_compartment")

    def build_map(fill_ser, norm_, sel):
        m = folium.Map(
            location=[MAP_LAT, MAP_LON],
            zoom_start=14,
            tiles=GOOGLE_SAT,
            attr="Google Satellite",
        )
        for _, row in gdf.iterrows():
            cid = row["compartment_id"]
            ndvi_val = fill_ser.iloc[row.name] if hasattr(fill_ser, "iloc") else fill_ser.get(cid)
            ndvi_label = f"{ndvi_val:.3f}" if ndvi_val is not None and not pd.isna(ndvi_val) else "N/A"

            is_sel = (cid == sel)
            if is_sel:
                style = {"fillColor": ndvi_to_hex(ndvi_val, norm_),
                         "color": "#ffffff", "weight": 3.5, "fillOpacity": 0.9}
            else:
                style = {"fillColor": ndvi_to_hex(ndvi_val, norm_),
                         "color": "white", "weight": 0.7, "fillOpacity": 0.65}

            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda _, s=style: s,
                tooltip=folium.Tooltip(
                    f"<b>Nogabals:</b> {cid}<br>"
                    f"<b>NDVI ({tooltip_year_label}):</b> {ndvi_label}"
                ),
            ).add_to(m)
        return m

    m = build_map(fill_series, norm, sel_id)
    st_folium(m, key="main_map", width="100%", height=580, returned_objects=[])

    # Colour scale legend
    vmin_leg = float(fill_series.dropna().quantile(0.05))
    vmax_leg = float(fill_series.dropna().quantile(0.95))
    gradient = ("background: linear-gradient(to right, "
                "#d73027,#fc8d59,#fee08b,#d9ef8b,#91cf60,#1a9850);")
    st.markdown(
        f'<div style="{gradient} height:12px; border-radius:3px; margin:4px 0;"></div>'
        f'<div style="display:flex;justify-content:space-between;">'
        f'<small>Zems {vmin_leg:.2f}</small>'
        f'<small style="text-align:center">NDVI {selected_year}</small>'
        f'<small>Augsts {vmax_leg:.2f}</small></div>',
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════
# RIGHT – selector + chart
# ════════════════════════════════════════════════════════════
with col_panel:
    all_ids = sorted(gdf["compartment_id"].tolist())
    avg_lookup = gdf.set_index("compartment_id")["avg_ndvi_all"].to_dict()

    def fmt(cid):
        if cid is None:
            return "— neviens —"
        avg = avg_lookup.get(cid)
        ndvi_str = f"{avg:.3f}" if avg is not None and not pd.isna(avg) else "N/A"
        return f"{cid}  {trend_arrow(cid)}  (NDVI {ndvi_str})"

    selected_id = st.selectbox(
        "Izvēlieties nogabalu",
        options=[None] + all_ids,
        format_func=fmt,
        key="sel_compartment",
    )

    if selected_id is None:
        st.info("Izvēlieties nogabalu augstāk, lai skatītu laika rindu.")
    else:
        ts = df[df["compartment_id"] == selected_id].sort_values("year")

        fig = go.Figure()
        if ts.empty:
            st.warning(f"Nav datu nogabalam {selected_id}.")
            fig = None
        else:
            # ── Time-series chart ───────────────────────────────

            # P10–P90 band
            fig.add_trace(go.Scatter(
                x=pd.concat([ts["year"], ts["year"][::-1]]),
                y=pd.concat([ts["p90_ndvi"], ts["p10_ndvi"][::-1]]),
                fill="toself",
                fillcolor="rgba(100,180,100,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name="P10–P90",
                hoverinfo="skip",
            ))

            # Mean NDVI line
            fig.add_trace(go.Scatter(
                x=ts["year"],
                y=ts["mean_ndvi"],
                mode="lines+markers",
                line=dict(color="#1a9641", width=2.5),
                marker=dict(size=7),
                name="Vidējais NDVI",
                hovertemplate="Gads %{x}: %{y:.3f}<extra></extra>",
            ))

            # Vertical dashed line for selected year
            yr_row = ts[ts["year"] == selected_year]
            yr_ndvi = float(yr_row["mean_ndvi"].iloc[0]) if not yr_row.empty else None
            fig.add_vline(
                x=selected_year,
                line=dict(color="rgba(60,60,200,0.55)", width=1.5, dash="dash"),
                annotation_text=str(selected_year),
                annotation_position="top right",
                annotation_font_size=11,
            )

            fig.update_layout(
                xaxis=dict(title="Gads", tickmode="array", tickvals=YEARS),
                yaxis=dict(title="NDVI", range=[0.5, 1.0]),
                legend=dict(orientation="h", y=1.08, x=0),
                margin=dict(l=40, r=10, t=30, b=40),
                height=310,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
            fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            # ── Stats table for selected year ───────────────────
            if not yr_row.empty:
                row_data = yr_row.iloc[0]
                stats = pd.DataFrame({
                    "Rādītājs": ["Vidējais NDVI", "P10", "P90", "Std novirze", "CV"],
                    f"{selected_year}": [
                        f"{row_data['mean_ndvi']:.3f}",
                        f"{row_data['p10_ndvi']:.3f}",
                        f"{row_data['p90_ndvi']:.3f}",
                        f"{row_data['stddev_ndvi']:.3f}",
                        f"{row_data['cv_ndvi']:.3f}",
                    ],
                }).set_index("Rādītājs")
                st.dataframe(stats, use_container_width=True)
            else:
                st.caption(f"Nav datu gadam {selected_year}.")

# ════════════════════════════════════════════════════════════
# PIXEL-LEVEL NDVI MAP – local GeoTIFFs, no GEE connection
# ════════════════════════════════════════════════════════════
with st.expander(f"Pikseļu līmeņa NDVI karte — {selected_year}", expanded=True):
    st.caption(
        "Sentinel-2 jūnijs–augusts mediānais NDVI kompozīts · apgriezts līdz nogabaliem · "
        "vizualizācija: 0.75–0.92 · sarkans→zaļš palete. Pārslēdziet gadus slāņu kontrolē. "
        "Baltas kontūras = nogabalu robežas."
    )

    m_tif = folium.Map(
        location=[MAP_LAT, MAP_LON],
        zoom_start=14,
        tiles=GOOGLE_SAT,
        attr="Google Satellite",
    )

    for yr in YEARS:
        try:
            url, img_bounds = tif_to_overlay(yr)
            folium.raster_layers.ImageOverlay(
                image=url,
                bounds=img_bounds,
                name=f"NDVI {yr}",
                opacity=0.8,
                show=(yr == selected_year),
                overlay=True,
            ).add_to(m_tif)
        except FileNotFoundError:
            pass  # skip years whose TIF hasn't been exported yet

    # Compartment outlines – added last so they sit on top
    folium.GeoJson(
        gdf[["compartment_id", "geometry"]].__geo_interface__,
        name="Nogabali",
        style_function=lambda _: {
            "fillColor": "none",
            "color": "white",
            "weight": 0.8,
            "fillOpacity": 0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["compartment_id"], aliases=["Nogabals:"]
        ),
    ).add_to(m_tif)

    folium.LayerControl(collapsed=False).add_to(m_tif)

    st_folium(m_tif, key="tif_map", width="100%", height=560, returned_objects=[])
